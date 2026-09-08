"""对实验 04 的三组结构化剪枝模型使用相同设置进行恢复微调。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer


PROFILES = ("light", "balanced", "strong")
FIELDS = [
    "profile",
    "epochs",
    "parameters",
    "parameter_reduction",
    "gmac_reduction",
    "raw_mAP50_95",
    "finetuned_mAP50",
    "finetuned_mAP50_95",
    "mAP50_95_drop_vs_baseline",
    "recovery_vs_raw",
    "inference_ms_per_image",
    "elapsed_seconds",
    "peak_gpu_memory_mb",
    "best_weights",
]


def freeze_batchnorm(trainer: DetectionTrainer) -> None:
    """小数据恢复训练时固定 BN 统计量，避免 102 张图片将其快速冲坏。"""
    for module in trainer.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def validate(weights: Path, data: Path, device: str, profile: str):
    metrics = YOLO(str(weights)).val(
        data=str(data),
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
        seed=42,
        deterministic=True,
        project="runs/prune/experiment04_independent/validation_finetuned",
        name=profile,
        exist_ok=True,
    )
    return float(metrics.box.map50), float(metrics.box.map), float(metrics.speed["inference"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/coco128_split.yaml")
    parser.add_argument("--raw-results", default="reports/experiment04_independent_comparison.csv")
    parser.add_argument("--root", default="runs/prune/experiment04_independent")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="reports/experiment04_independent_finetuned.csv")
    args = parser.parse_args()

    started_at = datetime.now().astimezone().isoformat()
    total_started = time.perf_counter()
    data = Path(args.data).resolve()
    raw_results_path = Path(args.raw_results).resolve()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    with raw_results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = {row["profile"]: row for row in csv.DictReader(handle)}
    baseline = json.loads((root / "run_info.json").read_text(encoding="utf-8"))["baseline"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    run_rows = []

    for profile_index, profile in enumerate(args.profiles):
        profile_started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        raw_weights = root / profile / "pruned_raw.pt"
        if not raw_weights.is_file():
            raise FileNotFoundError(raw_weights)
        pruned = YOLO(str(raw_weights))

        overrides = {
            "model": str(raw_weights),
            "data": str(data),
            "epochs": args.epochs,
            "imgsz": 640,
            "batch": 8,
            "device": args.device,
            "workers": 0,
            "optimizer": "AdamW",
            "lr0": 0.0001,
            "lrf": 0.1,
            "patience": args.epochs,
            "warmup_epochs": 0.0,
            "warmup_bias_lr": 0.0001,
            "weight_decay": 0.0001,
            "mosaic": 0.0,
            "fliplr": 0.0,
            "scale": 0.0,
            "translate": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "close_mosaic": 0,
            "amp": True,
            "plots": False,
            "seed": 42,
            "deterministic": True,
            "project": str(root / "finetune"),
            "name": profile,
            "exist_ok": True,
        }
        trainer = DetectionTrainer(overrides=overrides)
        trainer.model = pruned.model
        trainer.callbacks["on_pretrain_routine_end"].append(freeze_batchnorm)
        trainer.callbacks["on_train_epoch_start"].append(freeze_batchnorm)
        trainer.train()

        best = trainer.best if trainer.best.is_file() else trainer.last
        map50, map_value, speed = validate(best, data, args.device, profile)
        raw = raw_rows[profile]
        row = {
            "profile": profile,
            "epochs": args.epochs,
            "parameters": raw["parameters"],
            "parameter_reduction": raw["parameter_reduction"],
            "gmac_reduction": raw["gmac_reduction"],
            "raw_mAP50_95": raw["mAP50_95"],
            "finetuned_mAP50": map50,
            "finetuned_mAP50_95": map_value,
            "mAP50_95_drop_vs_baseline": float(baseline["mAP50_95"]) - map_value,
            "recovery_vs_raw": map_value - float(raw["mAP50_95"]),
            "inference_ms_per_image": speed,
            "elapsed_seconds": time.perf_counter() - profile_started,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
            ),
            "best_weights": str(best.resolve()),
        }
        run_rows.append(row)
        with output.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if profile_index == 0:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"[{profile}] 微调后 mAP50-95={map_value:.4f}, "
            f"恢复={row['recovery_vs_raw']:+.4f}, "
            f"较基线下降={row['mAP50_95_drop_vs_baseline']:+.4f}"
        )

        del trainer, pruned
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    info = {
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "total_elapsed_seconds": time.perf_counter() - total_started,
        "epochs_each": args.epochs,
        "profiles": args.profiles,
        "results": run_rows,
    }
    (root / "finetune_run_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"微调对比已保存：{output}")


if __name__ == "__main__":
    main()
