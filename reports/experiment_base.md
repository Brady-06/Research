# 实验记录

## 实验01：YOLO11n 基线流程验证

- 日期：2026-09-07
- 目的：验证环境、GPU、数据读取、训练和验证流程
- 模型：YOLO11n，初始权重 `models/yolo11n.pt`
- 数据：COCO8（仅用于流程验证）
- 设置：3 epochs，imgsz=640，batch=8，RTX 5060 Ti
- 结果目录：`runs/train/coco8_baseline/`

### 验证结果

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 原始 YOLO11n | 0.568 | 0.850 | 0.846 | 0.630 |
| 微调后 YOLO11n | 0.560 | 0.850 | 0.877 | 0.634 |

### 结论

微调后 mAP50 和 mAP50-95 略有提升，说明训练流程正常。由于 COCO8 数据量很小，本结果仅用于流程验证，不能作为正式科研结论。

关键文件：

- 最佳权重：`runs/train/coco8_baseline/weights/best.pt`
- 训练曲线：`runs/train/coco8_baseline/results.png`
- 指标表：`runs/train/coco8_baseline/results.csv`

## 后续实验模板

```text
实验编号与名称：
目的：
模型与数据：
关键设置：
Precision / Recall / mAP50 / mAP50-95：
模型大小与速度：
结果目录：
结论：
```
