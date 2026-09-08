# 实验02：YOLO11n 在 COCO128 子集上的训练验证

- 日期：2026-09-07
- 目的：在比 COCO8 更大的数据上验证训练流程
- 数据：COCO128，固定随机种子 42 划分为 102 张训练图、26 张验证图
- 模型：YOLO11n，初始权重 `models/yolo11n.pt`
- 设置：3 epochs，imgsz=640，batch=8，RTX 5060 Ti
- 结果目录：`runs/train/experiment02_coco128/`

## 验证结果

| 模型 | Precision | Recall | mAP50 | mAP50-95 |   50 55 60 等等平均的
|---|---:|---:|---:|---:|
| 原始 YOLO11n | 0.642 | 0.741 | 0.767 | 0.615 |
| 微调后 YOLO11n | 0.644 | 0.701 | 0.788 | 0.627 |

## 结论

微调后 mAP50 和 mAP50-95 略有提升，Precision 基本不变，Recall 有所下降。说明在独立验证集上流程能够运行，但 128 张图片和 3 轮训练仍然太少，不能作为正式科研结论。后续剪枝实验应继续使用更大的数据集和相同验证集。

关键文件：

- 最佳权重：`runs/train/experiment02_coco128/weights/best.pt`
- 数据配置：`configs/coco128_split.yaml`
- 划分脚本：`scripts/split_coco128.py`
