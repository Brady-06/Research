"""YOLO11n 全卷积层独立敏感度分析。

每次从同一个实验 02 best.pt 重新加载模型，只在一个卷积层中屏蔽
L1 重要性最低的 10% 输出通道，再用同一验证集测量精度变化。
该步骤用于制定后续结构化剪枝方案，本身不会生成结构化剪枝模型。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO


FIELDNAMES = [
    "layer_index",
    "total_layers",
    "layer",
    "in_channels",
    "out_channels",
    "kernel_size",
    "ratio",
    "masked_filters",
    "baseline_mAP50",
    "trial_mAP50",
    "mAP50_drop",
    "baseline_mAP50_95",
    "trial_mAP50_95",
    "mAP50_95_drop",
]


def find_candidate_layers(model: YOLO) -> list[tuple[str, torch.nn.Conv2d]]:
    """返回全部适合进行 10% 通道屏蔽测试的卷积层。"""
    return [
        (name, module)
        for name, module in model.model.named_modules()
        if isinstance(module, torch.nn.Conv2d) and module.out_channels >= 16
    ]


def mask_low_l1_filters(conv: torch.nn.Conv2d, ratio: float) -> int:
    """将一个卷积层中 L1 重要性最低的部分输出通道置零。"""
    count = max(1, round(conv.out_channels * ratio))
    count = min(count, conv.out_channels - 1)
    scores = conv.weight.detach().abs().flatten(1).mean(1)
    indices = torch.argsort(scores)[:count]
    with torch.no_grad():
        conv.weight[indices] = 0
        if conv.bias is not None:
            conv.bias[indices] = 0
    return count


def validate(model: YOLO, data: Path, device: str) -> tuple[float, float]:
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
        project="runs/analysis",
        name="experiment03_sensitivity_temp",
        exist_ok=True,
    )
    return float(metrics.box.map50), float(metrics.box.map)


def append_row(output: Path, row: dict[str, object], write_header: bool) -> None:
    with output.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="runs/train/experiment02_coco128/weights/best.pt",
    )
    parser.add_argument("--data", default="configs/coco128_split.yaml")
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="reports/experiment03_sensitivity.csv")
    args = parser.parse_args()

    if not 0 < args.ratio < 1:
        raise ValueError("--ratio 必须在 0 和 1 之间")

    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    output = Path(args.output).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"找不到权重：{weights}")
    if not data.is_file():
        raise FileNotFoundError(f"找不到数据配置：{data}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    baseline_model = YOLO(str(weights))
    candidates = find_candidate_layers(baseline_model)
    baseline_map50, baseline_map = validate(baseline_model, data, args.device)

    print(
        f"正式分析开始：{len(candidates)} 个卷积层，"
        f"baseline mAP50={baseline_map50:.4f}, mAP50-95={baseline_map:.4f}"
    )

    for index, (layer_name, layer_info) in enumerate(candidates, start=1):
        trial = YOLO(str(weights))
        trial_layer = dict(trial.model.named_modules())[layer_name]
        masked = mask_low_l1_filters(trial_layer, args.ratio)
        trial_map50, trial_map = validate(trial, data, args.device)

        row = {
            "layer_index": index,
            "total_layers": len(candidates),
            "layer": layer_name,
            "in_channels": layer_info.in_channels,
            "out_channels": layer_info.out_channels,
            "kernel_size": f"{layer_info.kernel_size[0]}x{layer_info.kernel_size[1]}",
            "ratio": args.ratio,
            "masked_filters": masked,
            "baseline_mAP50": baseline_map50,
            "trial_mAP50": trial_map50,
            "mAP50_drop": baseline_map50 - trial_map50,
            "baseline_mAP50_95": baseline_map,
            "trial_mAP50_95": trial_map,
            "mAP50_95_drop": baseline_map - trial_map,
        }
        append_row(output, row, write_header=index == 1)
        print(
            f"[{index:02d}/{len(candidates):02d}] {layer_name}: "
            f"mAP50-95={trial_map:.4f}, drop={baseline_map - trial_map:+.4f}"
        )

    print(f"全部完成，结果已保存：{output}")


if __name__ == "__main__":
    main()
