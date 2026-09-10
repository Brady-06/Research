# YOLO11s COCO2017 structural pruning

Search uses COCO val2017 for model selection; results are development-set metrics.
Sensitivity masking is a heuristic, not exact channel-removal equivalence.
GMAC reduction is measured on unfused models with the same counter.
Stop: max_steps
Compute reduction: 1.19%
Raw mAP50-95: 0.437372

See run_info.json, steps.json and comparison.csv for full results.