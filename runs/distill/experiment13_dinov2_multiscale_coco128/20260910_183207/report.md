# 实验13：多尺度 DINOv2 蒸馏剪枝 YOLO11n（COCO128）

运行目录：`runs/distill/experiment13_dinov2_multiscale_coco128/20260910_183207`

## 结论

本轮多尺度蒸馏没有刷新输入模型的最优精度，也没有优于实验 12 的 P4 单尺度方案。

训练确实发生：多尺度末轮 mAP50-95 为 `0.4957`，比普通微调末轮 `0.4828` 高 `+0.0128`，说明 DINOv2 特征约束缓解了部分后期退化；但它比 P4 单尺度蒸馏末轮 `0.5040` 低 `-0.0084`。因此在本次 COCO128 小样本设置中，增加 P3/P5 对齐没有带来额外收益。

验证集只有 26 张图片，这一结论属于小规模可行性证据，不能代替完整 COCO2017 实验。

## 结果对照

| 方案 | best mAP50 | best mAP50-95 | last mAP50 | last mAP50-95 |
|---|---:|---:|---:|---:|
| 输入剪枝模型 | 0.7156 | 0.5962 | — | — |
| 普通微调 control | 0.7156 | 0.5962 | 0.6359 | 0.4828 |
| 实验12：P4 单尺度 DINOv2 | 0.7156 | 0.5962 | 0.6279 | **0.5040** |
| 实验13：P3/P4/P5 多尺度 DINOv2 | 0.7156 | 0.5962 | 0.6248 | 0.4957 |

所有 best 都与输入模型相同，是因为 30 个训练 epoch 均未超过训练起点；并非训练未执行。实验 13 的 `last.pt` 有 90 个共同张量相对输入权重发生改变，且三层 DINO loss 均下降。

## 方法

- 学生模型：`runs/prune/experiment05_greedy/20260908_161152/finetune/greedy/weights/best.pt`
- 数据：`configs/coco128_split.yaml`（102 train / 26 val）
- 教师：冻结 DINOv2 ViT-S/14
- 教师特征：一次前向提取第 4、8、12 个 Transformer Block（代码索引 `3/7/11`）
- 学生特征：YOLO P3/P4/P5，实际形状分别为 `40×80×80`、`128×40×40`、`256×20×20`
- 对齐：三个独立 1×1 投影，cosine distance，层权重 `0.25/0.50/0.25`
- 总蒸馏权重：`0.5`，前 2 epochs 线性 warm-up
- 训练：30 epochs、batch 8、imgsz 640、AdamW、lr0=1e-4、seed 42、冻结 BN、关闭强增强
- 耗时：168.58 秒（约 2 分 49 秒，含 source/control/distill 训练与 best 验证）

## 损失变化

| 指标 | epoch 1 | epoch 30 |
|---|---:|---:|
| 总 DINO loss | 0.9811 | 0.8986 |
| P3 loss | 1.0125 | 0.9814 |
| P4 loss | 0.9568 | 0.8722 |
| P5 loss | 0.9983 | 0.8687 |

三层损失均下降，说明多尺度监督实际参与并被优化；P3 的下降幅度最小，可能是多尺度没有超过单尺度的原因之一。

## 输出与验证

- 推荐部署权重：`runs/distill/experiment13_dinov2_multiscale_coco128/20260910_183207/finetune/distill/weights/best.pt`
- 末轮研究权重：`runs/distill/experiment13_dinov2_multiscale_coco128/20260910_183207/finetune/distill/weights/last.pt`
- 训练恢复权重：同目录下 `best_training_wrapper.pt` 和 `last_training_wrapper.pt`
- 日志：`runs/distill/experiment13_dinov2_multiscale_coco128/20260910_183207/run.log`
- 结构化结果：同目录下 `comparison.csv`、`run_info.json`
- 曲线：同目录下 `finetune/control/results.png` 与 `finetune/distill/results.png`

部署版 best/last 均可由 Ultralytics 独立加载，数值全部有限，并且不包含 DINOv2 教师或投影头。推理耗时受首次 CUDA 预热影响较大，本实验不据此作速度结论。

