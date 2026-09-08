"""COCO128 上的 YOLO11n 敏感度引导独立结构化剪枝对比。

light / balanced / strong 三组实验均从实验 02 的同一个 best.pt 开始，
按照实验 03 的敏感度排序选取不同数量的低敏感层。Torch-Pruning
负责同步修改依赖层的通道，输出可重新加载的 Ultralytics 检查点。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch_pruning as tp
from ultralytics import YOLO


PROFILE_LAYER_COUNTS = {
    "light": 3,
    "balanced": 6,
    "strong": 9,
}

# 该残差输出层与相邻已剪层组合后，Torch-Pruning 1.6.1 未能完整更新
# 后续 Bottleneck 的输入通道，因此不作为本轮结构化剪枝根层。
UNSAFE_ROOT_LAYERS = {"model.13.m.0.cv2.conv"}

RESULT_FIELDS = [
    "profile",
    "selected_layers",
    "root_channels_removed",
    "parameters",
    "parameter_reduction",
    "gmacs",
    "gmac_reduction",
    "model_size_mb",
    "mAP50",
    "mAP50_drop",
    "mAP50_95",
    "mAP50_95_drop",
    "inference_ms_per_image",
    "elapsed_seconds",
    "peak_gpu_memory_mb",
    "weights",
]


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row
        for row in rows
        if not row["layer"].startswith(("model.0", "model.10", "model.23"))
        and row["layer"] not in UNSAFE_ROOT_LAYERS
        and int(row["out_channels"]) >= 64
        and float(row["mAP50_95_drop"]) <= 0.005
    ]
    return sorted(candidates, key=lambda row: float(row["mAP50_95_drop"]))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_gmacs(model: torch.nn.Module, example: torch.Tensor) -> float:
    macs, _ = tp.utils.count_ops_and_params(model, example_inputs=example)
    return float(macs) / 1e9


def choose_indices(conv: torch.nn.Conv2d, ratio: float, channel_multiple: int) -> list[int]:
    desired = max(1, round(conv.out_channels * ratio))
    if conv.out_channels >= channel_multiple * 2:
        desired = max(channel_multiple, round(desired / channel_multiple) * channel_multiple)
        desired = min(desired, conv.out_channels - channel_multiple)
    scores = conv.weight.detach().abs().flatten(1).mean(1)
    return torch.argsort(scores)[:desired].cpu().tolist()


def prune_one_layer(
    model: torch.nn.Module,
    layer_name: str,
    ratio: float,
    channel_multiple: int,
    example: torch.Tensor,
) -> dict[str, object]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    modules = dict(model.named_modules())
    conv = modules.get(layer_name)
    if not isinstance(conv, torch.nn.Conv2d):
        raise RuntimeError(f"找不到可剪卷积层：{layer_name}")

    before = conv.out_channels
    indices = choose_indices(conv, ratio, channel_multiple)
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    group = graph.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=indices)
    if not graph.check_pruning_group(group):
        raise RuntimeError(f"依赖图拒绝剪枝：{layer_name}")
    group.prune()

    with torch.no_grad():
        model(example)
    return {
        "layer": layer_name,
        "before_out_channels": before,
        "after_out_channels": conv.out_channels,
        "removed": before - conv.out_channels,
    }


def save_checkpoint(base_weights: Path, model: torch.nn.Module, output: Path) -> None:
    checkpoint = torch.load(base_weights, map_location="cpu", weights_only=False)
    checkpoint["model"] = copy.deepcopy(model).cpu().half()
    checkpoint["ema"] = None
    checkpoint["optimizer"] = None
    checkpoint["scaler"] = None
    checkpoint["updates"] = None
    checkpoint["epoch"] = -1
    checkpoint["best_fitness"] = None
    checkpoint["train_args"] = dict(checkpoint.get("train_args") or {})
    checkpoint["train_args"]["model"] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


def validate(weights: Path, data: Path, device: str, name: str) -> tuple[float, float, float]:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
        seed=42,
        deterministic=True,
        project="runs/prune/experiment04_independent/validation",
        name=name,
        exist_ok=True,
    )
    return (
        float(metrics.box.map50),
        float(metrics.box.map),
        float(metrics.speed["inference"]),
    )


def write_result(path: Path, row: dict[str, object], header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/experiment02_coco128/weights/best.pt")
    parser.add_argument("--data", default="configs/coco128_split.yaml")
    parser.add_argument("--sensitivity", default="reports/experiment03_sensitivity.csv")
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_LAYER_COUNTS, default=list(PROFILE_LAYER_COUNTS))
    parser.add_argument("--ratio", type=float, default=0.125)
    parser.add_argument("--channel-multiple", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", default="runs/prune/experiment04_independent")
    parser.add_argument("--results", default="reports/experiment04_independent_comparison.csv")
    parser.add_argument("--run-info", default="runs/prune/experiment04_independent/run_info.json")
    args = parser.parse_args()

    run_started_at = datetime.now().astimezone().isoformat()
    run_started = time.perf_counter()

    base_weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    sensitivity = Path(args.sensitivity).resolve()
    output_dir = Path(args.output_dir).resolve()
    results_path = Path(args.results).resolve()
    run_info_path = Path(args.run_info).resolve()
    for required in (base_weights, data, sensitivity):
        if not required.is_file():
            raise FileNotFoundError(required)

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    candidates = read_candidates(sensitivity)
    needed = max(PROFILE_LAYER_COUNTS[name] for name in args.profiles)
    if len(candidates) < needed:
        raise RuntimeError(f"低敏感候选层只有 {len(candidates)} 个，少于所需的 {needed} 个")

    example = torch.randn(1, 3, 640, 640, device=device)
    baseline_yolo = YOLO(str(base_weights))
    baseline_model = baseline_yolo.model.to(device).eval()
    baseline_params = count_parameters(baseline_model)
    baseline_gmacs = count_gmacs(baseline_model, example)
    baseline_map50, baseline_map, baseline_speed = validate(base_weights, data, args.device, "baseline")

    results_path.unlink(missing_ok=True)
    print(
        f"Baseline: params={baseline_params:,}, GMACs={baseline_gmacs:.4f}, "
        f"mAP50-95={baseline_map:.4f}, inference={baseline_speed:.3f} ms/image"
    )

    for profile_index, profile in enumerate(args.profiles):
        profile_started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        selected = candidates[: PROFILE_LAYER_COUNTS[profile]]
        yolo = YOLO(str(base_weights))
        model = yolo.model.to(device).eval()
        manifest = []

        print(f"\n[{profile}] 开始剪 {len(selected)} 个低敏感根层")
        for index, candidate in enumerate(selected, start=1):
            layer_started = time.perf_counter()
            item = prune_one_layer(
                model,
                candidate["layer"],
                args.ratio,
                args.channel_multiple,
                example,
            )
            item["sensitivity_drop"] = float(candidate["mAP50_95_drop"])
            item["elapsed_seconds"] = time.perf_counter() - layer_started
            manifest.append(item)
            print(
                f"  [{index}/{len(selected)}] {item['layer']}: "
                f"{item['before_out_channels']} -> {item['after_out_channels']}"
            )

        with torch.no_grad():
            model(example)
        params = count_parameters(model)
        gmacs = count_gmacs(model, example)

        profile_dir = output_dir / profile
        weights_path = profile_dir / "pruned_raw.pt"
        manifest_path = profile_dir / "pruning_manifest.json"
        save_checkpoint(base_weights, model, weights_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        map50, map_value, speed = validate(weights_path, data, args.device, profile)
        row = {
            "profile": profile,
            "selected_layers": len(selected),
            "root_channels_removed": sum(int(item["removed"]) for item in manifest),
            "parameters": params,
            "parameter_reduction": 1 - params / baseline_params,
            "gmacs": gmacs,
            "gmac_reduction": 1 - gmacs / baseline_gmacs,
            "model_size_mb": weights_path.stat().st_size / 1e6,
            "mAP50": map50,
            "mAP50_drop": baseline_map50 - map50,
            "mAP50_95": map_value,
            "mAP50_95_drop": baseline_map - map_value,
            "inference_ms_per_image": speed,
            "elapsed_seconds": time.perf_counter() - profile_started,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1e6
                if torch.cuda.is_available()
                else 0.0
            ),
            "weights": str(weights_path),
        }
        write_result(results_path, row, header=profile_index == 0)
        print(
            f"[{profile}] params -{row['parameter_reduction']:.2%}, "
            f"GMACs -{row['gmac_reduction']:.2%}, "
            f"mAP50-95={map_value:.4f} (drop {row['mAP50_95_drop']:+.4f})"
        )
        del model, yolo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_info = {
        "started_at": run_started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "total_elapsed_seconds": time.perf_counter() - run_started,
        "configuration": vars(args),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_pruning": getattr(tp, "__version__", "1.6.1"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        },
        "baseline": {
            "parameters": baseline_params,
            "gmacs": baseline_gmacs,
            "mAP50": baseline_map50,
            "mAP50_95": baseline_map,
            "inference_ms_per_image": baseline_speed,
        },
    }
    run_info_path.parent.mkdir(parents=True, exist_ok=True)
    run_info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n三组结果已保存：{results_path}")
    print(f"运行信息已保存：{run_info_path}")


if __name__ == "__main__":
    main()
