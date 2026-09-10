"""DINOv2 feature distillation for the best pruned YOLO11n model on COCO128.

The teacher is frozen DINOv2 ViT-S/14.  Its patch tokens are interpolated to
the student's P4 (stride-16) neck feature map, then compared after a learned
1x1 projection.  Detection loss remains the original Ultralytics loss.

The script writes a normal, inference-only YOLO ``weights/best.pt``.  A
separate ``best_training_wrapper.pt`` is retained only to resume the
distillation run (it contains DINOv2 and the projection head).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT = ROOT / "runs/prune/experiment05_greedy/20260908_161152/finetune/greedy/weights/best.pt"
DEFAULT_DATA = ROOT / "configs/coco128_split.yaml"
DEFAULT_RUN_ROOT = ROOT / "runs/distill/experiment12_dinov2_distill_coco128"
MULTISCALE_RUN_ROOT = ROOT / "runs/distill/experiment13_dinov2_multiscale_coco128"
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_batchnorm(trainer: DetectionTrainer) -> None:
    """Keep BN fixed: 102 training images are insufficient for stable updates."""
    for module in trainer.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


class FrozenBNTrainer(DetectionTrainer):
    def _freeze_runtime(self) -> None:
        distiller = getattr(self.model, "_dino_feature_distiller", None)
        if distiller is not None:
            distiller.teacher.eval()
            for parameter in distiller.teacher.parameters():
                parameter.requires_grad_(False)
        freeze_batchnorm(self)

    def build_optimizer(self, *args, **kwargs):
        # Ultralytics re-enables every floating parameter in _setup_train.
        # Re-freeze immediately before optimizer construction.
        self._freeze_runtime()
        return super().build_optimizer(*args, **kwargs)

    def _model_train(self):
        super()._model_train()
        self._freeze_runtime()

    def final_eval(self):
        # The training checkpoint contains the temporary DINO teacher.  main()
        # first converts it to a plain DetectionModel, then validates that file.
        return None


class DINOFeatureDistiller(nn.Module):
    """Frozen DINOv2 teacher plus the trainable feature-alignment head."""

    def __init__(self, student_channels: int, device: torch.device, dino_size: int, weight: float):
        super().__init__()
        if dino_size % 14:
            raise ValueError("--dino-size must be divisible by DINOv2 patch size 14")
        self.dino_size = dino_size
        self.base_weight = weight
        self.current_weight = 0.0
        # Official DINOv2 hub entry point; first run downloads code and ViT-S/14 weights.
        self.teacher = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.projector = nn.Conv2d(student_channels, 384, kernel_size=1, bias=False).to(device)
        nn.init.kaiming_normal_(self.projector.weight, mode="fan_out", nonlinearity="linear")
        self.register_buffer("mean", torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def teacher_feature(self, images: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        images = F.interpolate(images, size=(self.dino_size, self.dino_size), mode="bilinear", align_corners=False)
        tokens = self.teacher.forward_features((images - self.mean) / self.std)["x_norm_patchtokens"]
        grid = self.dino_size // 14
        tokens = tokens.transpose(1, 2).reshape(images.shape[0], 384, grid, grid)
        return F.interpolate(tokens, size=output_size, mode="bilinear", align_corners=False)

    def loss(self, student_feature: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        teacher_feature = self.teacher_feature(images, student_feature.shape[-2:])
        projected = F.normalize(self.projector(student_feature), dim=1)
        teacher_feature = F.normalize(teacher_feature, dim=1)
        # Cosine distance is normalized but remains O(1); a plain elementwise MSE
        # would be diluted by DINOv2's 384 channels (~0.005 at initialization).
        return 1.0 - (projected * teacher_feature).sum(dim=1).mean()


class MultiScaleDINOFeatureDistiller(nn.Module):
    """One-pass DINOv2 block 4/8/12 alignment to YOLO P3/P4/P5."""

    def __init__(self, student_channels: list[int], device: torch.device, dino_size: int, weight: float):
        super().__init__()
        if dino_size % 14:
            raise ValueError("--dino-size must be divisible by DINOv2 patch size 14")
        if len(student_channels) != 3:
            raise ValueError("P3/P4/P5 distillation requires exactly three student features")
        self.dino_size = dino_size
        self.base_weight = weight
        self.current_weight = 0.0
        self.block_indices = (3, 7, 11)  # zero-based: transformer blocks 4, 8 and 12
        self.level_weights = (0.25, 0.50, 0.25)
        self.teacher = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.projectors = nn.ModuleList(
            nn.Conv2d(channels, 384, kernel_size=1, bias=False) for channels in student_channels
        ).to(device)
        for projector in self.projectors:
            nn.init.kaiming_normal_(projector.weight, mode="fan_out", nonlinearity="linear")
        self.register_buffer("mean", torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def teacher_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        images = F.interpolate(images, size=(self.dino_size, self.dino_size), mode="bilinear", align_corners=False)
        normalized = (images - self.mean) / self.std
        return self.teacher.get_intermediate_layers(
            normalized, n=self.block_indices, reshape=True, norm=True
        )

    def loss(self, student_features: list[torch.Tensor], images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        teacher_features = self.teacher_features(images)
        level_losses = []
        for student, teacher, projector in zip(student_features, teacher_features, self.projectors):
            teacher = F.interpolate(teacher, size=student.shape[-2:], mode="bilinear", align_corners=False)
            projected = F.normalize(projector(student), dim=1)
            teacher = F.normalize(teacher, dim=1)
            level_losses.append(1.0 - (projected * teacher).sum(dim=1).mean())
        total = sum(level_weight * loss for level_weight, loss in zip(self.level_weights, level_losses))
        return total, level_losses


class CaptureP4:
    """Pickle-safe hook used by checkpoint/EMA serialization."""

    def __init__(self, features: dict[str, torch.Tensor]):
        self.features = features

    def __call__(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.features["p4"] = output

    def __getstate__(self):
        # A captured tensor can retain an autograd graph; never put it in a checkpoint.
        self.features.clear()
        return {"features": self.features}


class CaptureFeature:
    """Pickle-safe feature hook for one YOLO pyramid level."""

    def __init__(self, features: dict[str, torch.Tensor], key: str):
        self.features, self.key = features, key

    def __call__(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.features[self.key] = output

    def __getstate__(self):
        self.features.clear()
        return {"features": self.features, "key": self.key}


class DINOStudentModel(DetectionModel):
    """DetectionModel with an additional DINOv2 P4 feature loss."""

    def loss(self, batch: dict, preds=None):
        regular_loss, loss_items = super().loss(batch, preds)
        if not self.training:
            return regular_loss, loss_items
        feature = self._dino_student_features.get("p4")
        if feature is None:
            raise RuntimeError("P4 hook did not capture a student feature")
        dino_loss = self._dino_feature_distiller.loss(feature, batch["img"])
        loss_items["dino_loss"] = dino_loss.detach()
        # Ultralytics sums the returned components. Keep distillation as its own
        # component instead of broadcasting it into box/cls/dfl independently.
        scaled = dino_loss * self._dino_feature_distiller.current_weight * batch["img"].shape[0]
        return torch.cat((regular_loss, scaled.reshape(1))), loss_items


class MultiScaleDINOStudentModel(DetectionModel):
    """DetectionModel with DINOv2 P3/P4/P5 feature losses."""

    def loss(self, batch: dict, preds=None):
        regular_loss, loss_items = super().loss(batch, preds)
        if not self.training:
            return regular_loss, loss_items
        features = self._dino_student_features
        if any(level not in features for level in ("p3", "p4", "p5")):
            raise RuntimeError("P3/P4/P5 hooks did not capture all student features")
        dino_loss, level_losses = self._dino_feature_distiller.loss(
            [features["p3"], features["p4"], features["p5"]], batch["img"]
        )
        loss_items["dino_loss"] = dino_loss.detach()
        for level, level_loss in zip(("p3", "p4", "p5"), level_losses):
            loss_items[f"dino_{level}"] = level_loss.detach()
        scaled = dino_loss * self._dino_feature_distiller.current_weight * batch["img"].shape[0]
        return torch.cat((regular_loss, scaled.reshape(1))), loss_items


# Make temporary training checkpoints loadable when this file is run with
# ``python -m`` (whose runtime name is otherwise ``__main__``).
_SERIALIZATION_MODULE = "scripts.distill_dinov2_coco128"
sys.modules.setdefault(_SERIALIZATION_MODULE, sys.modules[__name__])
for _serializable_class in (DINOFeatureDistiller, MultiScaleDINOFeatureDistiller, CaptureP4,
                            CaptureFeature, DINOStudentModel, MultiScaleDINOStudentModel):
    _serializable_class.__module__ = _SERIALIZATION_MODULE


def _find_p4_layer(model: nn.Module) -> nn.Module:
    """Select the stride-16 input of Detect without relying on fixed layer numbers."""
    detect = model.model[-1]
    sources = getattr(detect, "f", None)
    if not isinstance(sources, (list, tuple)) or len(sources) < 2:
        raise RuntimeError("Could not identify Detect's P4 input layer")
    return model.model[sources[1]]


def _find_pyramid_layers(model: nn.Module) -> list[nn.Module]:
    detect = model.model[-1]
    sources = getattr(detect, "f", None)
    if not isinstance(sources, (list, tuple)) or len(sources) != 3:
        raise RuntimeError("Could not identify Detect's P3/P4/P5 input layers")
    return [model.model[index] for index in sources]


def attach_distillation(model: nn.Module, device: torch.device, dino_size: int, weight: float) -> None:
    captured: dict[str, torch.Tensor] = {}
    p4 = _find_p4_layer(model)
    with torch.no_grad():
        model.eval()
        p4_feature: list[torch.Tensor] = []
        def probe(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            p4_feature.append(output)
        handle = p4.register_forward_hook(probe)
        model(torch.zeros(1, 3, 640, 640, device=device))
        handle.remove()
        model.train()
    if len(p4_feature) != 1 or not isinstance(p4_feature[0], torch.Tensor):
        raise RuntimeError("P4 feature probe failed")
    model.add_module("_dino_feature_distiller", DINOFeatureDistiller(p4_feature[0].shape[1], device, dino_size, weight))
    model._dino_student_features = captured
    p4.register_forward_hook(CaptureP4(captured))
    model.__class__ = DINOStudentModel


def attach_multiscale_distillation(model: nn.Module, device: torch.device, dino_size: int, weight: float) -> None:
    captured: dict[str, torch.Tensor] = {}
    levels = ("p3", "p4", "p5")
    pyramid_layers = _find_pyramid_layers(model)
    probed: dict[str, torch.Tensor] = {}
    handles = []
    with torch.no_grad():
        model.eval()
        for level, layer in zip(levels, pyramid_layers):
            handles.append(layer.register_forward_hook(
                lambda _module, _inputs, output, key=level: probed.__setitem__(key, output)
            ))
        model(torch.zeros(1, 3, 640, 640, device=device))
        for handle in handles:
            handle.remove()
        model.train()
    if any(level not in probed or not isinstance(probed[level], torch.Tensor) for level in levels):
        raise RuntimeError("P3/P4/P5 feature probe failed")
    channels = [probed[level].shape[1] for level in levels]
    model.add_module("_dino_feature_distiller", MultiScaleDINOFeatureDistiller(channels, device, dino_size, weight))
    model._dino_student_features = captured
    for level, layer in zip(levels, pyramid_layers):
        layer.register_forward_hook(CaptureFeature(captured, level))
    model.__class__ = MultiScaleDINOStudentModel


def detach_for_inference(model: nn.Module) -> nn.Module:
    """Remove train-only teacher/projector and restore the stock YOLO loss method."""
    clean = copy.deepcopy(model).float().eval()
    for module in clean.modules():
        for hook_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, (CaptureP4, CaptureFeature)):
                del module._forward_hooks[hook_id]
    clean.__dict__.pop("_dino_student_features", None)
    if "_dino_feature_distiller" in clean._modules:
        del clean._modules["_dino_feature_distiller"]
    clean.__class__ = DetectionModel
    return clean


def validate(weights: Path, data: Path, device: str, project: Path, name: str) -> dict[str, float]:
    metrics = YOLO(str(weights)).val(data=str(data), imgsz=640, batch=8, device=device, workers=0,
                                    plots=True, verbose=False, seed=42, deterministic=True,
                                    project=str(project), name=name, exist_ok=True)
    return {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
            "inference_ms": float(metrics.speed["inference"])}


def training_overrides(student: Path, data: Path, run_dir: Path, label: str, args) -> dict:
    return {"model": str(student), "data": str(data), "epochs": args.epochs, "imgsz": 640,
            "batch": args.batch, "device": args.device, "workers": 0, "optimizer": "AdamW",
            "lr0": 0.0001, "lrf": 0.1, "patience": args.epochs, "warmup_epochs": 0.0,
            "warmup_bias_lr": 0.0001, "weight_decay": 0.0001, "mosaic": 0.0, "fliplr": 0.0,
            "scale": 0.0, "translate": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "close_mosaic": 0, "amp": True, "plots": True, "seed": 42, "deterministic": True,
            "project": str(run_dir / "finetune"), "name": label, "exist_ok": False}


def prepare_trainable_flags(model: nn.Module) -> None:
    """Avoid setup warnings; FrozenBNTrainer re-freezes BN/teacher before optimizer creation."""
    for parameter in model.parameters():
        if parameter.dtype.is_floating_point:
            parameter.requires_grad_(True)


def train_control(student: Path, data: Path, run_dir: Path, args) -> Path:
    trainer = FrozenBNTrainer(overrides=training_overrides(student, data, run_dir, "control", args))
    trainer.model = YOLO(str(student)).model.to(trainer.device)
    prepare_trainable_flags(trainer.model)
    trainer.callbacks["on_pretrain_routine_end"].append(freeze_batchnorm)
    trainer.callbacks["on_train_epoch_start"].append(freeze_batchnorm)
    trainer.train()
    best = Path(trainer.best).resolve()
    if not best.is_file():
        raise RuntimeError("Control training completed without best.pt")
    del trainer
    torch.cuda.empty_cache()
    return best


def _write_clean_checkpoint(wrapper_path: Path, clean_path: Path) -> None:
    checkpoint = torch.load(wrapper_path, map_location="cpu", weights_only=False)
    wrapped_ema = checkpoint.get("ema") or checkpoint.get("model")
    checkpoint["ema"] = detach_for_inference(wrapped_ema).half()
    checkpoint["model"] = None
    torch.save(checkpoint, clean_path)


def train_distill(student: Path, data: Path, run_dir: Path, args) -> tuple[Path, Path]:
    trainer = FrozenBNTrainer(overrides=training_overrides(student, data, run_dir, "distill", args))
    trainer.model = YOLO(str(student)).model.to(trainer.device)
    if args.multiscale:
        attach_multiscale_distillation(trainer.model, trainer.device, args.dino_size, args.weight)
    else:
        attach_distillation(trainer.model, trainer.device, args.dino_size, args.weight)
    prepare_trainable_flags(trainer.model)

    def prepare(tr):
        tr.model._dino_feature_distiller.current_weight = args.weight / max(1, args.warmup_epochs)
        tr._freeze_runtime()

    def set_weight(tr):
        fraction = min(1.0, (tr.epoch + 1) / max(1, args.warmup_epochs))
        tr.model._dino_feature_distiller.current_weight = args.weight * fraction
        tr._freeze_runtime()

    trainer.callbacks["on_pretrain_routine_end"].append(prepare)
    trainer.callbacks["on_train_epoch_start"].append(set_weight)
    trainer.train()

    wrapper_best = Path(trainer.best)
    if not wrapper_best.is_file():
        raise RuntimeError("Distillation completed without best.pt")
    resume_best = wrapper_best.with_name("best_training_wrapper.pt")
    wrapper_best.replace(resume_best)
    _write_clean_checkpoint(resume_best, wrapper_best)
    wrapper_last = Path(trainer.last)
    resume_last = wrapper_last.with_name("last_training_wrapper.pt")
    wrapper_last.replace(resume_last)
    _write_clean_checkpoint(resume_last, wrapper_last)
    del trainer
    torch.cuda.empty_cache()
    return wrapper_best.resolve(), resume_best.resolve()


def write_report(run_dir: Path, results: dict, settings: dict) -> Path:
    source, control, distill = results["source"], results["control"], results["distill"]
    gain_source = distill["map50_95"] - source["map50_95"]
    gain_control = distill["map50_95"] - control["map50_95"]
    conclusion = (
        "本轮 DINOv2 蒸馏优于同轮普通微调。" if gain_control > 0
        else "本轮 DINOv2 蒸馏未超过同轮普通微调，不能认定蒸馏有效。"
    )
    text = f"""# 实验12：DINOv2 蒸馏剪枝 YOLO11n（COCO128）

运行目录：`{run_dir.relative_to(ROOT).as_posix()}`

## 结论

{conclusion}

蒸馏模型相对输入模型 mAP50-95 变化 `{gain_source:+.4f}`，相对相同 {settings['epochs']} 轮设置的普通微调对照变化 `{gain_control:+.4f}`。验证集仅 26 张图片，因此这里只作为小规模方法验证。

## 结果

| 方案 | mAP50 | mAP50-95 | 验证推理耗时/图 |
|---|---:|---:|---:|
| 输入剪枝模型 | {source['map50']:.4f} | {source['map50_95']:.4f} | {source['inference_ms']:.3f} ms |
| 普通微调 control | {control['map50']:.4f} | {control['map50_95']:.4f} | {control['inference_ms']:.3f} ms |
| DINOv2 蒸馏 | {distill['map50']:.4f} | {distill['map50_95']:.4f} | {distill['inference_ms']:.3f} ms |

## 方法与输入输出

- 输入学生：`{Path(settings['student']).relative_to(ROOT).as_posix()}`
- 数据：`{Path(settings['data']).relative_to(ROOT).as_posix()}`（102 train / 26 val）
- 教师：冻结的 DINOv2 ViT-S/14；对齐 YOLO P4（stride 16），1x1 投影，cosine distance。
- 总损失：YOLO 检测损失 + DINO 特征损失；最终权重 {settings['weight']}，前 {settings['warmup_epochs']} 轮线性 warm-up。
- 训练：{settings['epochs']} epochs、batch {settings['batch']}、640、AdamW、lr0=1e-4、seed=42、冻结 BN、关闭强增强。
- 部署权重：`{Path(settings['distill_best']).relative_to(ROOT).as_posix()}`
- 完整日志：`{(run_dir / 'run.log').relative_to(ROOT).as_posix()}`
- 结构化结果：`{(run_dir / 'comparison.csv').relative_to(ROOT).as_posix()}`、`{(run_dir / 'run_info.json').relative_to(ROOT).as_posix()}`
"""
    local = run_dir / "report.md"
    local.write_text(text, encoding="utf-8")
    summary = ROOT / "reports/experiment12_dinov2_distillation_coco128.md"
    summary.write_text(text, encoding="utf-8")
    return summary


def write_multiscale_report(run_dir: Path, results: dict, settings: dict) -> Path:
    source, control, distill = results["source"], results["control"], results["distill"]
    gain_source = distill["map50_95"] - source["map50_95"]
    gain_control = distill["map50_95"] - control["map50_95"]
    conclusion = "优于同设置普通微调对照" if gain_control > 0 else "未超过同设置普通微调对照"
    text = f"""# 实验13：多尺度 DINOv2 蒸馏剪枝 YOLO11n（COCO128）

运行目录：`{run_dir.relative_to(ROOT).as_posix()}`

## 结论

本轮 P3/P4/P5 多尺度 DINOv2 特征蒸馏{conclusion}。相对输入模型的 mAP50-95 变化为 `{gain_source:+.4f}`，相对普通微调对照变化为 `{gain_control:+.4f}`。验证集仅 26 张图片，本结果只作为小规模方法验证，不能代替完整 COCO 实验。

## 最优权重结果

| 方案 | mAP50 | mAP50-95 | 验证推理耗时/图 |
|---|---:|---:|---:|
| 输入剪枝模型 | {source['map50']:.4f} | {source['map50_95']:.4f} | {source['inference_ms']:.3f} ms |
| 普通微调 control | {control['map50']:.4f} | {control['map50_95']:.4f} | {control['inference_ms']:.3f} ms |
| 多尺度 DINOv2 蒸馏 | {distill['map50']:.4f} | {distill['map50_95']:.4f} | {distill['inference_ms']:.3f} ms |

## 方法与输入输出

- 输入学生：`{Path(settings['student']).relative_to(ROOT).as_posix()}`
- 数据：`{Path(settings['data']).relative_to(ROOT).as_posix()}`（102 train / 26 val）
- 教师：冻结的 DINOv2 ViT-S/14，一次前向提取第 4/8/12 个 Transformer Block（代码索引 3/7/11）。
- 对齐：分别对齐 YOLO P3/P4/P5；三个独立 1×1 投影；层权重 0.25/0.50/0.25；cosine distance。
- 总损失：YOLO 检测损失 + DINO 特征损失；最终权重 {settings['weight']}，前 {settings['warmup_epochs']} 轮线性 warm-up。
- 训练：{settings['epochs']} epochs、batch {settings['batch']}、640、AdamW、lr0=1e-4、seed=42、冻结 BN、关闭强增强。
- 部署权重：`{Path(settings['distill_best']).relative_to(ROOT).as_posix()}`
- 完整日志：`{(run_dir / 'run.log').relative_to(ROOT).as_posix()}`
- 结构化结果：`{(run_dir / 'comparison.csv').relative_to(ROOT).as_posix()}`、`{(run_dir / 'run_info.json').relative_to(ROOT).as_posix()}`
"""
    local = run_dir / "report.md"
    local.write_text(text, encoding="utf-8")
    summary = ROOT / "reports/experiment13_dinov2_multiscale_coco128.md"
    summary.write_text(text, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=None, help="Exact run directory; default uses a timestamp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--weight", type=float, default=0.5, help="Final DINO loss weight")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--dino-size", type=int, default=644, help="Must be a multiple of 14")
    parser.add_argument("--multiscale", action="store_true", help="Align DINO blocks 4/8/12 to YOLO P3/P4/P5")
    args = parser.parse_args()
    student, data = args.student.resolve(), args.data.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_root = MULTISCALE_RUN_ROOT if args.multiscale else DEFAULT_RUN_ROOT
    output = args.output.resolve() if args.output else default_root / stamp
    if not student.is_file() or not data.is_file() or args.epochs < 1 or args.batch < 1:
        raise ValueError("Student/data must exist; epochs and batch must be positive")
    output.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    feature = ("DINO blocks 4/8/12 -> YOLO P3/P4/P5; level weights 0.25/0.50/0.25"
               if args.multiscale else "YOLO Detect P4 (stride 16)")
    settings = {"student": str(student), "data": str(data), "teacher": "dinov2_vits14",
                "feature": feature, "multiscale": args.multiscale,
                "dino_block_indices_zero_based": [3, 7, 11] if args.multiscale else [11],
                "pyramid_weights": [0.25, 0.50, 0.25] if args.multiscale else [1.0],
                "epochs": args.epochs, "batch": args.batch,
                "imgsz": 640, "optimizer": "AdamW", "lr0": 0.0001, "weight": args.weight,
                "warmup_epochs": args.warmup_epochs, "dino_size": args.dino_size, "seed": 42,
                "device": args.device}
    (output / "train_config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    original_out, original_err = sys.stdout, sys.stderr
    with (output / "run.log").open("w", encoding="utf-8", buffering=1) as logfile:
        sys.stdout, sys.stderr = Tee(original_out, logfile), Tee(original_err, logfile)
        try:
            print(f"RUN_DIR {output}", flush=True)
            source_metrics = validate(student, data, args.device, output / "validation", "source")
            control_best = train_control(student, data, output, args)
            control_metrics = validate(control_best, data, args.device, output / "validation", "control")
            control_last = control_best.with_name("last.pt")
            control_last_metrics = validate(control_last, data, args.device, output / "validation", "control_last")
            distill_best, resume_best = train_distill(student, data, output, args)
            distill_metrics = validate(distill_best, data, args.device, output / "validation", "distill")
            distill_last = distill_best.with_name("last.pt")
            distill_last_metrics = validate(distill_last, data, args.device, output / "validation", "distill_last")
            results = {"source": source_metrics, "control": control_metrics, "distill": distill_metrics,
                       "control_last": control_last_metrics, "distill_last": distill_last_metrics}
            with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["label", "map50", "map50_95", "inference_ms", "weights"])
                writer.writeheader()
                for label, metrics, weights in (("source", source_metrics, student),
                                                ("control", control_metrics, control_best),
                                                ("distill", distill_metrics, distill_best),
                                                ("control_last", control_last_metrics, control_last),
                                                ("distill_last", distill_last_metrics, distill_last)):
                    writer.writerow({"label": label, **metrics, "weights": str(weights)})
            settings.update({"control_best": str(control_best), "control_last": str(control_last),
                             "distill_best": str(distill_best), "distill_last": str(distill_last),
                             "training_wrapper": str(resume_best)})
            run_info = {"started_at": started_at, "status": "complete",
                        "elapsed_seconds": time.perf_counter() - started, "configuration": settings,
                        "results": results, "sources": {str(student): sha256(student), str(data): sha256(data),
                        str(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
                        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "ultralytics": importlib.metadata.version("ultralytics"),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
            (output / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
            report_path = (write_multiscale_report(output, results, settings) if args.multiscale
                           else write_report(output, results, settings))
            print(json.dumps(run_info, ensure_ascii=False, indent=2), flush=True)
            print(f"REPORT {report_path}", flush=True)
        except BaseException as error:
            failure = {"status": "failed", "error": repr(error),
                       "elapsed_seconds": time.perf_counter() - started, "configuration": settings}
            (output / "run_info.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()
