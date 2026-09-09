"""Train YOLO11s on the complete COCO2017 dataset for Experiment 08."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile

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


def main():
    started = time.perf_counter()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ROOT / "runs/train/experiment08_yolo11s_coco2017" / stamp
    root.mkdir(parents=True, exist_ok=False)
    config = ROOT / "configs/coco2017.yaml"
    weights = ROOT / "weights/yolo11s.pt"
    settings = {
        "model": str(weights), "data": str(config), "epochs": 30,
        "imgsz": 512, "batch": 64, "device": 0, "workers": 2,
        "seed": 42, "deterministic": False, "pretrained": True, "patience": 10,
    }
    (root / "train_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    original_out, original_err = sys.stdout, sys.stderr
    with (root / "run.log").open("w", encoding="utf-8", buffering=1) as logfile:
        sys.stdout = Tee(original_out, logfile)
        sys.stderr = Tee(original_err, logfile)
        try:
            print(f"RUN_DIR {root}", flush=True)
            print(json.dumps(settings, ensure_ascii=False), flush=True)
            model = YOLO(str(weights))
            model.train(
                data=str(config), epochs=settings["epochs"], imgsz=settings["imgsz"],
                batch=settings["batch"], device=settings["device"], workers=settings["workers"],
                seed=settings["seed"], deterministic=settings["deterministic"],
                pretrained=settings["pretrained"], patience=settings["patience"],
                project=str(root.parent), name=stamp, exist_ok=True, plots=True, verbose=True,
            )
            best = root / "weights/best.pt"
            last = root / "weights/last.pt"
            status = {"status": "complete", "best": str(best), "last": str(last),
                      "elapsed_seconds": time.perf_counter() - started}
            (root / "train_info.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            (ROOT / "reports/experiment08_yolo11s_coco2017.md").write_text(
                "# 实验08：YOLO11s 在 COCO2017 上训练\n\n"
                f"训练目录：`{root.relative_to(ROOT).as_posix()}`。\n\n"
                "从官方 `yolo11s.pt` 开始，在完整 COCO2017 train2017 上训练，使用 val2017 按验证综合指标保存 `best.pt`。\n\n"
                f"训练配置见 `{(root / 'train_config.json').relative_to(ROOT).as_posix()}`，完整日志见 `{(root / 'run.log').relative_to(ROOT).as_posix()}`。\n",
                encoding="utf-8",
            )
            print(f"COMPLETE best={best}", flush=True)
        except BaseException as error:
            failure = {"status": "failed", "error": repr(error),
                       "elapsed_seconds": time.perf_counter() - started}
            (root / "train_info.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()
