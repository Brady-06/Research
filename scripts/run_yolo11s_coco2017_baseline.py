"""Experiment 07: official YOLO11s baseline on COCO2017 val."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.downloads import attempt_download_asset

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now().astimezone().isoformat()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile

    def write(self, text):
        self.stream.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def count_lines(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_report(run_dir, info):
    metrics = info["metrics"]
    speed = info["speed_ms_per_image"]
    benchmark = info["benchmark"]
    lines = [
        "# 实验07：YOLO11s 在 COCO2017 上的基线",
        "",
        f"运行目录：`{run_dir.relative_to(ROOT).as_posix()}`。",
        "",
        "本实验直接验证 Ultralytics 官方 COCO 预训练 `yolo11s.pt`，没有再次训练，因此保存的 `baseline.pt` 是基线快照，不是新训练得到的 best.pt。这个检查点将作为后续敏感度分析和剪枝的统一起点。",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        f"| COCO2017 train 图片 | {info['dataset']['train_images']:,} |",
        f"| COCO2017 val 图片 | {info['dataset']['val_images']:,} |",
        f"| 参数量 | {info['parameters']:,} |",
        f"| GMAC（640） | {info['gmacs']:.4f} |",
        f"| mAP50 | {metrics['map50']:.4f} |",
        f"| mAP50-95 | {metrics['map50_95']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| 验证推理耗时/图 | {speed['inference']:.3f} ms |",
        f"| 纯网络前向中位耗时 | {benchmark['latency_median_ms']:.3f} ms |",
        "",
        "验证设置：COCO2017 `val2017`、5000张图片、640像素、batch=32、RTX 5060 Ti、FP32权重验证。纯网络测速采用融合模型、batch=1、640×640，预热30次并测量250次；不包含读图、预处理和NMS。",
        "",
        "完整终端日志保存在 `run.log`，环境、来源哈希、数据量、指标和耗时保存在 `run_info.json`，Ultralytics生成的曲线和预测图保存在 `validation`。",
        "",
        f"数据下载与验证总耗时：{info['elapsed_seconds']:.1f}秒。",
        "",
    ]
    text = "\n".join(lines)
    (run_dir / "report.md").write_text(text, encoding="utf-8")
    (ROOT / "reports/experiment07_yolo11s_coco2017_baseline.md").write_text(text, encoding="utf-8")


def run(args, run_dir, info):
    from greedy_evaluation import benchmark

    config = ROOT / "configs/coco2017.yaml"
    weights = ROOT / "weights/yolo11s.pt"
    free_before = shutil.disk_usage(ROOT).free
    if free_before < 35 * 1024**3:
        raise RuntimeError(f"COCO2017 下载前至少需要35GiB空闲空间，当前只有 {free_before / 1024**3:.1f}GiB")
    weights.parent.mkdir(parents=True, exist_ok=True)
    download_started = time.perf_counter()
    downloaded = Path(attempt_download_asset(weights)).resolve()
    if not downloaded.is_file():
        raise FileNotFoundError(f"权重下载失败：{downloaded}")
    print(f"WEIGHTS {downloaded}", flush=True)

    model = YOLO(str(downloaded))
    metrics = model.val(
        data=str(config), split="val", imgsz=640, batch=args.batch,
        device=args.device, workers=0, plots=True, verbose=True,
        seed=42, deterministic=True, project=str(run_dir), name="validation",
        exist_ok=False,
    )
    dataset_root = Path("C:/Users/22565/datasets/coco")
    train_list, val_list = dataset_root / "train2017.txt", dataset_root / "val2017.txt"
    if not train_list.is_file() or not val_list.is_file():
        raise FileNotFoundError("COCO2017 图片列表不存在，下载未完整完成")
    dataset = {
        "root": str(dataset_root),
        "train_images": count_lines(train_list),
        "val_images": count_lines(val_list),
        "train_directory_images": len(list((dataset_root / "images/train2017").glob("*.jpg"))),
        "val_directory_images": len(list((dataset_root / "images/val2017").glob("*.jpg"))),
    }
    if dataset["train_images"] != 118287 or dataset["val_images"] != 5000:
        raise RuntimeError(f"COCO2017 列表数量异常：{dataset}")
    if dataset["train_directory_images"] != 118287 or dataset["val_directory_images"] != 5000:
        raise RuntimeError(f"COCO2017 图片数量异常：{dataset}")

    baseline = run_dir / "weights/baseline.pt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, baseline)
    if sha256(downloaded) != sha256(baseline):
        raise RuntimeError("基线权重复制校验失败")

    raw_model = YOLO(str(baseline)).model.float().cuda(int(args.device)).eval()
    example = torch.zeros(1, 3, 640, 640, device=f"cuda:{args.device}")
    macs, _ = tp.utils.count_ops_and_params(copy.deepcopy(raw_model), example_inputs=example)
    parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    del raw_model, example
    torch.cuda.empty_cache()
    measured = benchmark({"yolo11s": baseline}, args.device)["yolo11s"]

    result = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    if not all(torch.isfinite(torch.tensor(value)).item() for value in result.values()):
        raise RuntimeError(f"验证指标不是有限数值：{result}")
    info.update(
        status="complete", finished_at=now(),
        elapsed_seconds=time.perf_counter() - info.pop("_timer"),
        download_and_validation_seconds=time.perf_counter() - download_started,
        model="yolo11s", source="official COCO-pretrained yolo11s.pt",
        source_weights=str(downloaded), source_sha256=sha256(downloaded),
        baseline_weights=str(baseline), baseline_sha256=sha256(baseline),
        config=str(config), config_sha256=sha256(config), dataset=dataset,
        metrics=result, speed_ms_per_image={key: float(value) for key, value in metrics.speed.items()},
        parameters=parameters, gmacs=float(macs) / 1e9, benchmark=measured,
        disk_free_before_bytes=free_before, disk_free_after_bytes=shutil.disk_usage(ROOT).free,
        environment={name: importlib.metadata.version(name) for name in ("torch", "ultralytics", "torch-pruning")},
        python=sys.version, platform=platform.platform(), gpu=torch.cuda.get_device_name(int(args.device)),
    )
    write_json(run_dir / "run_info.json", info)
    write_report(run_dir, info)
    print(f"COMPLETE mAP50-95={result['map50_95']:.6f} {run_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    os.chdir(ROOT)
    run_dir = ROOT / "runs/baseline/experiment07_yolo11s_coco2017" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    info = {"started_at": now(), "status": "starting", "configuration": vars(args), "_timer": time.perf_counter()}
    original_out, original_err = sys.stdout, sys.stderr
    with (run_dir / "run.log").open("w", encoding="utf-8", buffering=1) as logfile:
        sys.stdout, sys.stderr = Tee(original_out, logfile), Tee(original_err, logfile)
        handlers = [(handler, handler.stream) for handler in LOGGER.handlers if hasattr(handler, "stream")]
        for handler, _ in handlers:
            handler.setStream(sys.stdout)
        try:
            print(f"RUN_DIR {run_dir}", flush=True)
            run(args, run_dir, info)
        except BaseException as error:
            info.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=repr(error), finished_at=now())
            if "_timer" in info:
                info["elapsed_seconds"] = time.perf_counter() - info.pop("_timer")
            write_json(run_dir / "run_info.json", info)
            traceback.print_exc()
            raise
        finally:
            for handler, stream in handlers:
                handler.setStream(stream)
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()
