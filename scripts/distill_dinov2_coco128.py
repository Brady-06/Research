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
        projected = F.normalize(self.projector(student_feature), dim=1)
        teacher_feature = F.normalize(teacher_feature, dim=1)
        # Cosine distance is normalized but remains O(1); a plain elementwise MSE
        # would be diluted by DINOv2's 384 channels (~0.005 at initialization).
        return 1.0 - (projected * teacher_feature).sum(dim=1).mean()


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


# Make temporary training checkpoints loadable when this file is run with
# ``python -m`` (whose runtime name is otherwise ``__main__``).
_SERIALIZATION_MODULE = "scripts.distill_dinov2_coco128"
sys.modules.setdefault(_SERIALIZATION_MODULE, sys.modules[__name__])
for _serializable_class in (DINOFeatureDistiller, CaptureP4, DINOStudentModel):
    _serializable_class.__module__ = _SERIALIZATION_MODULE


def _find_p4_layer(model: nn.Module) -> nn.Module:
    """Select the stride-16 input of Detect without relying on fixed layer numbers."""
    detect = model.model[-1]
    sources = getattr(detect, "f", None)
    if not isinstance(sources, (list, tuple)) or len(sources) < 2:
        raise RuntimeError("Could not identify Detect's P4 input layer")
    return model.model[sources[1]]


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


def detach_for_inference(model: nn.Module) -> nn.Module:
    """Remove train-only teacher/projector and restore the stock YOLO loss method."""
    clean = copy.deepcopy(model).float().eval()
    for module in clean.modules():
        for hook_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, CaptureP4):
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
    args = parser.parse_args()
    student, data = args.student.resolve(), args.data.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output.resolve() if args.output else DEFAULT_RUN_ROOT / stamp
    if not student.is_file() or not data.is_file() or args.epochs < 1 or args.batch < 1:
        raise ValueError("Student/data must exist; epochs and batch must be positive")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    settings = {"student": str(student), "data": str(data), "teacher": "dinov2_vits14",
                "feature": "YOLO Detect P4 (stride 16)", "epochs": args.epochs, "batch": args.batch,
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
            distill_best, resume_best = train_distill(student, data, output, args)
            distill_metrics = validate(distill_best, data, args.device, output / "validation", "distill")
            results = {"source": source_metrics, "control": control_metrics, "distill": distill_metrics}
            with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["label", "map50", "map50_95", "inference_ms", "weights"])
                writer.writeheader()
                for label, metrics, weights in (("source", source_metrics, student),
                                                ("control", control_metrics, control_best),
                                                ("distill", distill_metrics, distill_best)):
                    writer.writerow({"label": label, **metrics, "weights": str(weights)})
            settings.update({"control_best": str(control_best), "distill_best": str(distill_best),
                             "training_wrapper": str(resume_best)})
            run_info = {"started_at": datetime.now().astimezone().isoformat(), "status": "complete",
                        "elapsed_seconds": time.perf_counter() - started, "configuration": settings,
                        "results": results, "sources": {str(student): sha256(student), str(data): sha256(data),
                        str(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
                        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "ultralytics": importlib.metadata.version("ultralytics"),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
            (output / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
            report_path = write_report(output, results, settings)
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
