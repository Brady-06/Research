"""Shared, checked recovery training and evaluation for experiment 05.

Only load trusted checkpoints produced locally by this research project.
Validation receives a separately loaded model because Ultralytics fuses it.
"""

from __future__ import annotations

import copy
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer


def _checkpoint(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"Expected an existing trusted local .pt checkpoint: {path}")
    return path


def _shapes(model: nn.Module) -> dict:
    """Structural widths and tensor dimensions, excluding dynamic anchor caches."""
    return {
        name: (
            type(module).__name__,
            tuple((key, tuple(value.shape)) for key, value in module.state_dict().items()),
            getattr(module, "groups", None),
        )
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d))
    }


def _bn_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{key}": value.detach().cpu().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        for key, value in module.state_dict().items()
    }


def _freeze_bn(trainer: DetectionTrainer) -> None:
    for module in trainer.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


class FrozenBNTrainer(DetectionTrainer):
    def _model_train(self):
        # The epoch-start callback precedes _model_train in Ultralytics 8.4.142.
        # Freeze after super() so model.train() cannot turn BN updates back on.
        super()._model_train()
        _freeze_bn(self)


def run_finetune(
    weights: Path, data: Path, output_dir: Path, device: str, epochs: int = 10
) -> Path:
    """Train the supplied pruned architecture directly; assert BN and shape invariants."""
    weights = _checkpoint(weights)
    data, output_dir = Path(data).resolve(), Path(output_dir).resolve()
    if not data.is_file() or epochs < 1:
        raise ValueError("An existing data config and positive epoch count are required")
    pruned = YOLO(str(weights))
    expected_shapes = _shapes(pruned.model)
    overrides = {
        "model": str(weights), "data": str(data), "epochs": epochs,
        "imgsz": 640, "batch": 8, "device": device, "workers": 0,
        "optimizer": "AdamW", "lr0": 0.0001, "lrf": 0.1,
        "patience": epochs, "warmup_epochs": 0.0, "warmup_bias_lr": 0.0001,
        "weight_decay": 0.0001, "mosaic": 0.0, "fliplr": 0.0,
        "scale": 0.0, "translate": 0.0, "hsv_h": 0.0, "hsv_s": 0.0,
        "hsv_v": 0.0, "close_mosaic": 0, "amp": True, "plots": False,
        "seed": 42, "deterministic": True,
        "project": str(output_dir.parent), "name": output_dir.name,
        "exist_ok": False,
    }
    trainer = FrozenBNTrainer(overrides=overrides)
    # Passing the module avoids get_model() rebuilding the original YAML widths.
    trainer.model = pruned.model
    reference: dict[str, torch.Tensor] = {}

    def prepare(tr):
        _freeze_bn(tr)
        reference.update(_bn_state(tr.model))
        if not reference:
            raise RuntimeError("Expected unfused BatchNorm layers for recovery training")

    def check_batch(tr):
        if tr.loss is None or not torch.isfinite(tr.loss.detach()).all().item():
            raise RuntimeError("Recovery training produced missing or non-finite loss")
        if any(m.training for m in tr.model.modules() if isinstance(m, nn.BatchNorm2d)):
            raise RuntimeError("BatchNorm statistics unexpectedly enabled during training")

    def check_final(tr):
        actual = _bn_state(tr.model)
        if actual.keys() != reference.keys():
            raise RuntimeError("BatchNorm architecture changed during recovery")
        changed = [key for key in reference if not torch.equal(reference[key], actual[key])]
        if changed:
            raise RuntimeError(f"Frozen BatchNorm buffers/affine parameters changed: {changed[:5]}")
        if _shapes(tr.model) != expected_shapes:
            raise RuntimeError("Pruned architecture changed during recovery")
        print("Verified: finite training losses, unchanged BN state and pruned widths.")

    trainer.callbacks["on_pretrain_routine_end"].append(prepare)
    trainer.callbacks["on_train_batch_end"].append(check_batch)
    trainer.callbacks["on_train_end"].append(check_final)
    trainer.train()
    best = Path(trainer.best)
    if not best.is_file():
        raise RuntimeError("Recovery completed without best.pt")
    reloaded = YOLO(str(best))
    if _shapes(reloaded.model) != expected_shapes:
        raise RuntimeError("Reloaded best.pt does not preserve the pruned architecture")
    del trainer, pruned, reloaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best.resolve()


def evaluate(weights: Path, data: Path, output_dir: Path, device: str) -> dict:
    """Validate a fresh checkpoint instance, never the mutable search model."""
    weights = _checkpoint(weights)
    output_dir = Path(output_dir).resolve()
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(Path(data).resolve()), imgsz=640, batch=8, device=device,
        workers=0, plots=False, verbose=False, seed=42, deterministic=True,
        project=str(output_dir.parent), name=output_dir.name, exist_ok=True,
    )
    result = {
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "validation_inference_ms": float(metrics.speed["inference"]),
    }
    if not all(torch.isfinite(torch.tensor(v)).item() for v in result.values()):
        raise RuntimeError(f"Non-finite validation metrics for {weights}")
    del model, metrics
    return result


def benchmark(weights: dict[str, Path], device: str) -> dict:
    """FP32 fused batch-1 network forward only; excludes preprocessing and NMS.

    Warm each model for 30 forwards, then rotate model order across five rounds
    of 50 synchronized samples per model. No inference-speed claim is inferred.
    GMACs/parameters use the unfused architecture, matching experiment 04.
    """
    if not weights:
        return {}
    target = torch.device("cpu" if device == "cpu" else (
        str(device) if str(device).startswith("cuda:") else f"cuda:{device}"
    ))
    example = torch.zeros(1, 3, 640, 640, device=target, dtype=torch.float32)
    models, results, samples = {}, {}, {}
    for label, path in weights.items():
        model = YOLO(str(_checkpoint(path))).model.to(target).float().eval()
        parameters = sum(p.numel() for p in model.parameters())
        # Op counter installs hooks; keep even that operation off the timed model.
        macs, _ = tp.utils.count_ops_and_params(copy.deepcopy(model), example_inputs=example)
        results[label] = {"parameters": parameters, "gmacs": float(macs) / 1e9,
                          "round_medians_ms": []}
        models[label] = model.fuse(verbose=False).eval()
        samples[label] = []

    def synchronize():
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    labels = list(models)
    with torch.inference_mode():
        for model in models.values():
            for _ in range(30):
                model(example)
        synchronize()
        for round_index in range(5):
            shift = round_index % len(labels)
            order = labels[shift:] + labels[:shift]
            for label in order:
                model = models[label]
                # Briefly warm on each switch so an idle model is not penalized.
                for _ in range(5):
                    model(example)
                synchronize()
                values = []
                for _ in range(50):
                    if target.type == "cuda":
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        model(example)
                        end.record()
                        synchronize()
                        elapsed = start.elapsed_time(end)
                    else:
                        started = time.perf_counter()
                        model(example)
                        elapsed = (time.perf_counter() - started) * 1000
                    values.append(elapsed)
                samples[label].extend(values)
                results[label]["round_medians_ms"].append(statistics.median(values))
    for label, values in samples.items():
        results[label].update({
            "latency_median_ms": statistics.median(values),
            "latency_p90_ms": float(torch.quantile(torch.tensor(values), 0.9)),
            "sample_count": len(values), "batch": 1, "imgsz": 640,
            "precision": "float32", "fused": True,
            "scope": "network_forward_excluding_preprocess_and_nms",
        })
    del models, model, example
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return results
