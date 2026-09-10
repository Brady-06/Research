# 实验12：DINOv2 蒸馏剪枝 YOLO11n（COCO128）

运行目录：`runs/distill/experiment12_dinov2_distill_coco128/20260910_181026`

## 结论

本轮 DINOv2 蒸馏未超过同轮普通微调，不能认定蒸馏有效。

蒸馏模型相对输入模型 mAP50-95 变化 `+0.0000`，相对相同 30 轮设置的普通微调对照变化 `+0.0000`。验证集仅 26 张图片，因此这里只作为小规模方法验证。

训练确实更新了权重：control 与 distill 的 `last.pt` 均有 90 个公共张量相对输入发生变化。但所有训练 epoch 都没有超过起点，因而两组 `best.pt` 在 checkpoint 精度下均与输入模型的 499 个公共张量完全相同。最后一轮 control / distill 的 mAP50-95 分别为 0.4828 / 0.5040；DINOv2 对后期退化有一定缓解，但本设置下没有形成可保留的精度收益。

## 结果

| 方案 | mAP50 | mAP50-95 | 验证推理耗时/图 |
|---|---:|---:|---:|
| 输入剪枝模型 | 0.7156 | 0.5962 | 9.878 ms |
| 普通微调 control | 0.7156 | 0.5962 | 3.247 ms |
| DINOv2 蒸馏 | 0.7156 | 0.5962 | 4.088 ms |
| 普通微调最后一轮 | 0.6359 | 0.4828 | 4.638 ms |
| DINOv2 蒸馏最后一轮 | 0.6279 | 0.5040 | 4.622 ms |

各次验证的推理耗时存在明显预热和轮间波动，本表不能用于宣称蒸馏带来推理加速；三组可部署模型结构相同。

## 方法与输入输出

- 输入学生：`runs/prune/experiment05_greedy/20260908_161152/finetune/greedy/weights/best.pt`
- 数据：`configs/coco128_split.yaml`（102 train / 26 val）
- 教师：冻结的 DINOv2 ViT-S/14；对齐 YOLO P4（stride 16），1x1 投影，cosine distance。
- 总损失：YOLO 检测损失 + DINO 特征损失；最终权重 0.5，前 2 轮线性 warm-up。
- 训练：30 epochs、batch 8、640、AdamW、lr0=1e-4、seed=42、冻结 BN、关闭强增强。
- 部署权重：`runs/distill/experiment12_dinov2_distill_coco128/20260910_181026/finetune/distill/weights/best.pt`
- 完整日志：`runs/distill/experiment12_dinov2_distill_coco128/20260910_181026/run.log`
- 结构化结果：`runs/distill/experiment12_dinov2_distill_coco128/20260910_181026/comparison.csv`、`runs/distill/experiment12_dinov2_distill_coco128/20260910_181026/run_info.json`
