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
import json
import math
import time
import types
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT = ROOT / "runs/prune/experiment05_greedy/20260908_161152/finetune/greedy/weights/best.pt"
DEFAULT_DATA = ROOT / "configs/coco128_split.yaml"
DEFAULT_RUN = ROOT / "runs/distill/experiment07_dinov2_coco128"
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def freeze_batchnorm(trainer: DetectionTrainer) -> None:
    """Keep BN fixed: 102 training images are insufficient for stable updates."""
    for module in trainer.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


class FrozenBNTrainer(DetectionTrainer):
    def _model_train(self):
        super()._model_train()
        freeze_batchnorm(self)


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
        self.projector = nn.Conv2d(student_channels, 384, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.projector.weight, mode="fan_out", nonlinearity="linear")
        self.register_buffer("mean", torch.tensor(DINO_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(DINO_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def teacher_feature(self, images: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        images = F.interpolate(images, size=(self.dino_size, self.dino_size), mode="bilinear", align_corners=False)
        tokens = self.teacher.forward_features((images - self.mean) / self.std)["x_norm_patchtokens"]
        grid = self.dino_size // 14
        tokens = tokens.transpose(1, 2).reshape(images.shape[0], 384, grid, grid)
        return F.interpolate(tokens, size=output_size, mode="bilinear", align_corners=False)

    def loss(self, student_feature: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        teacher_feature = self.teacher_feature(images, student_feature.shape[-2:])
        projected = self.projector(student_feature)
        # Channel L2 normalization prevents raw feature scale from dominating YOLO loss.
        return F.mse_loss(F.normalize(projected, dim=1), F.normalize(teacher_feature, dim=1))


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


def _find_p4_layer(model: nn.Module) -> nn.Module:
    """Select the stride-16 input of Detect without relying on fixed layer numbers."""
    detect = model.model[-1]
    sources = getattr(detect, "f", None)
    if not isinstance(sources, (list, tuple)) or len(sources) < 2:
        raise RuntimeError("Could not identify Detect's P4 input layer")
    return model.model[sources[1]]


def distillation_loss(self: nn.Module, batch: dict, preds=None):
    """Replacement for DetectionModel.loss; called after student forward hooks fire."""
    regular_loss, loss_items = self._original_detection_loss(batch, preds)
    if not self.training:
        return regular_loss, loss_items
    feature = self._dino_student_features.get("p4")
    if feature is None:
        raise RuntimeError("P4 hook did not capture a student feature")
    dino_loss = self._dino_feature_distiller.loss(feature, batch["img"])
    loss_items["dino_loss"] = dino_loss.detach()
    # Ultralytics detection loss is batch-summed; scale the mean feature loss alike.
    return regular_loss + dino_loss * self._dino_feature_distiller.current_weight * batch["img"].shape[0], loss_items


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
    model._original_detection_loss = model.loss
    model.loss = types.MethodType(distillation_loss, model)


def detach_for_inference(model: nn.Module) -> nn.Module:
    """Remove train-only teacher/projector and restore the stock YOLO loss method."""
    clean = copy.deepcopy(model).float().eval()
    for module in clean.modules():
        for hook_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, CaptureP4):
                del module._forward_hooks[hook_id]
    clean.__dict__.pop("_dino_student_features", None)
    clean.__dict__.pop("_original_detection_loss", None)
    clean.__dict__.pop("loss", None)
    if "_dino_feature_distiller" in clean._modules:
        del clean._modules["_dino_feature_distiller"]
    return clean


def validate(weights: Path, data: Path, device: str, project: Path) -> dict[str, float]:
    metrics = YOLO(str(weights)).val(data=str(data), imgsz=640, batch=8, device=device, workers=0,
                                    plots=False, verbose=False, seed=42, deterministic=True,
                                    project=str(project), name="validation", exist_ok=True)
    return {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
            "inference_ms": float(metrics.speed["inference"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--weight", type=float, default=0.5, help="Final DINO loss weight")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--dino-size", type=int, default=644, help="Must be a multiple of 14")
    args = parser.parse_args()
    student, data, output = args.student.resolve(), args.data.resolve(), args.output.resolve()
    if not student.is_file() or not data.is_file() or args.epochs < 1 or args.batch < 1:
        raise ValueError("Student/data must exist; epochs and batch must be positive")
    output.mkdir(parents=True, exist_ok=True)

    overrides = {"model": str(student), "data": str(data), "epochs": args.epochs, "imgsz": 640,
                 "batch": args.batch, "device": args.device, "workers": 0, "optimizer": "AdamW",
                 "lr0": 0.0001, "lrf": 0.1, "patience": args.epochs, "warmup_epochs": 0.0,
                 "warmup_bias_lr": 0.0001, "weight_decay": 0.0001, "mosaic": 0.0, "fliplr": 0.0,
                 "scale": 0.0, "translate": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
                 "close_mosaic": 0, "amp": True, "plots": False, "seed": 42, "deterministic": True,
                 "project": str(output.parent), "name": output.name, "exist_ok": True}
    trainer = FrozenBNTrainer(overrides=overrides)
    yolo = YOLO(str(student))
    trainer.model = yolo.model.to(trainer.device)
    attach_distillation(trainer.model, trainer.device, args.dino_size, args.weight)

    def set_weight(tr):
        fraction = min(1.0, (tr.epoch + 1) / max(1, args.warmup_epochs))
        tr.model._dino_feature_distiller.current_weight = args.weight * fraction
        tr.model._dino_feature_distiller.teacher.eval()
        freeze_batchnorm(tr)

    trainer.callbacks["on_pretrain_routine_end"].append(set_weight)
    trainer.callbacks["on_train_epoch_start"].append(set_weight)
    started = time.perf_counter()
    trainer.train()

    # Preserve resume checkpoint, then replace best.pt with a normal deployable YOLO checkpoint.
    wrapper_best = Path(trainer.best)
    resume_best = wrapper_best.with_name("best_training_wrapper.pt")
    wrapper_best.replace(resume_best)
    checkpoint = torch.load(resume_best, map_location="cpu", weights_only=False)
    wrapped_ema = checkpoint.get("ema") or checkpoint.get("model")
    checkpoint["ema"] = detach_for_inference(wrapped_ema).half()
    checkpoint["model"] = None
    torch.save(checkpoint, wrapper_best)
    result = validate(wrapper_best, data, args.device, output)
    report = {"started_at": datetime.now().astimezone().isoformat(), "elapsed_seconds": time.perf_counter() - started,
              "student": str(student), "teacher": "dinov2_vits14", "feature": "YOLO Detect P4 (stride 16)",
              "dino_size": args.dino_size, "final_weight": args.weight, "warmup_epochs": args.warmup_epochs,
              "epochs": args.epochs, "best_weights": str(wrapper_best.resolve()), "resume_weights": str(resume_best.resolve()),
              "validation": result}
    (output / "run_info.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
