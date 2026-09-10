# 实验12：DINOv2 蒸馏剪枝 YOLO11n（COCO128）

运行目录：`runs/distill/smoke_experiment12_structure`

## 结论

本轮 DINOv2 蒸馏未超过同轮普通微调，不能认定蒸馏有效。

蒸馏模型相对输入模型 mAP50-95 变化 `+0.0000`，相对相同 30 轮设置的普通微调对照变化 `+0.0000`。验证集仅 26 张图片，因此这里只作为小规模方法验证。

## 结果

| 方案 | mAP50 | mAP50-95 | 验证推理耗时/图 |
|---|---:|---:|---:|
| 输入剪枝模型 | 0.7156 | 0.5962 | 10.444 ms |
| 普通微调 control | 0.7156 | 0.5962 | 5.338 ms |
| DINOv2 蒸馏 | 0.7156 | 0.5962 | 3.985 ms |

## 方法与输入输出

- 输入学生：`runs/prune/experiment05_greedy/20260908_161152/finetune/greedy/weights/best.pt`
- 数据：`configs/coco128_split.yaml`（102 train / 26 val）
- 教师：冻结的 DINOv2 ViT-S/14；对齐 YOLO P4（stride 16），1x1 投影，cosine distance。
- 总损失：YOLO 检测损失 + DINO 特征损失；最终权重 0.5，前 2 轮线性 warm-up。
- 训练：1 epochs、batch 8、640、AdamW、lr0=1e-4、seed=42、冻结 BN、关闭强增强。
- 部署权重：`runs/distill/smoke_experiment12_structure/finetune/distill/weights/best.pt`
- 完整日志：`runs/distill/smoke_experiment12_structure/run.log`
- 结构化结果：`runs/distill/smoke_experiment12_structure/comparison.csv`、`runs/distill/smoke_experiment12_structure/run_info.json`
