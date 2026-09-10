# YOLO 结构化剪枝：方法原理与代码对照速查

> 本笔记是剪枝方法的学习型总结，配套本轮在 YOLO11n + coco128 上跑通的一整套实验。
> 实验脚本在临时目录 `C:\Users\22565\AppData\Local\Temp\yolo_coco128_impr\`（结果见 `results.jsonl`）。
> 本仓库现有脚本（未改动）：`scripts/prune_greedy.py`、`scripts/prune_independent_compare.py`、`scripts/greedy_evaluation.py`。

---

## 一、剪枝的本质：用精度换体积/速度

剪枝的**产出不是更高的 mAP**，而是更小的参数量 / 更少的 GMACs / 更快的推理，mAP 只会持平或略降。
衡量一次剪枝好坏的正确口径是：**同样剪幅下掉多少精度**，或**同样精度下能剪多少**。

- **非结构化剪枝**：把不重要的单个权重置零。只减体积，几乎不加速（稀疏矩阵无法利用）。
- **结构化剪枝**：删除整条卷积输出通道（`out_channels` 变小）。体积、GMACs、速度同时受益，是本笔记讨论的对象。

YOLO11n 是 nano 模型（~2.6M 参数），通道冗余本就很小；能"剪了不崩"的安全层只有 9 个，总剪幅被锁在 ~3.6% 参数附近。这是本轮所有方法"提升不明显"的根本原因，而非方法本身不行。

---

## 二、重要性准则：剪谁？（同结构，换评分函数即换方法）

所有准则都在"候选卷积层"上给每个**输出通道**打一个重要性分数，剪掉分数最低的一批。
打分函数对应代码：

| 准则 | 原理 | 代码位置 |
|---|---|---|
| L1 | 卷积核绝对值均值小 → 不重要 | `pruner.py: score_l1` |
| L2 | 卷积核 L2 范数小 → 不重要 | `pruner.py: score_l2` |
| BN-γ | BN 缩放因子 \|γ\| 小 → 不重要 | `pruner.py: score_bn` |
| FPGM | 离同类通道的几何中位数近 → 冗余 | `pruner.py: score_fpgm` |
| Taylor | \|梯度 × 激活\| 小 → 该通道对损失不敏感 | `pruner.py: score_taylor` |
| HRank | 输出特征图矩阵秩低 → 信息少 | `e5_sfp_hrank.py: hrank_main` |

本轮结论：**Taylor > FPGM > L1 ≈ L2 ≫ BN-γ**。Taylor 一次性剪枝 raw mAP 最高（0.336 vs L1 的 0.314），
说明"看梯度敏感度"比"看幅值"更准。差异只在安全层上较小；换到高剪枝率或更敏感的层会拉大。

---

## 三、结构化剪枝的依赖问题（为什么"一剪就崩"）

删除一个 conv 的输出通道，会连锁影响下游模块（BN 通道数、相连 conv 的输入通道、C2f 的 split/concat、残差相加）。
不处理依赖就会报 `expected 96 channels but got 91` 之类错误。

两个关键工程点（都踩过坑）：

1. **依赖图（DepGraph）**：用 Torch-Pruning 的 `prune_conv_out_channels` 自动联动删除下游通道。
2. **requires_grad 修复**：ultralytics 加载 `.pt` 时所有参数 `requires_grad=False`，导致 Torch-Pruning 的
   前向追踪找到 0 个可剪模块。剪枝前必须：
   ```python
   for p in model.parameters():
       p.requires_grad_(True)
   ```
   见 `pruner.py: prune_layer_out`。

代码对照：`pruner.py` 里 `candidate_layers`（9 个安全层）、`uniform_plan`（生成删除通道计划）、
`apply_plan`（应用并处理依赖）、`prune_layer_out`（真正剪一层）。

---

## 四、恢复手段：剪完怎么把精度补回来

| 手段 | 原理 | 结果 | 代码 |
|---|---|---|---|
| 微调（finetune） | 剪完继续训，让模型适应 | 标准做法，raw 0.314→0.340 | `pruner.py: finetune` |
| 稀疏训练 Network Slimming | 先给 BN-γ 加 L1 惩罚逼稀疏，再按 γ 剪 | **对 YOLO11n 失败**（γ 压不稀疏，剪完灾难） | `e3_sparse.py` |
| 软剪枝 SFP | 先归零通道让模型适应，再硬剪 | 本轮最佳（0.357，剪幅最小） | `e5_sfp_hrank.py: sfp_main` |
| 蒸馏 KD | 剪后模型向未剪教师模型对齐 | 本轮无增益（0.333 vs 普通微调 0.340） | `e6_kd.py` |
| 量化 INT8 | 权重/激活转低精度 | 未跑（缺 onnx/tensorrt） | — |

### 微调的关键写法（避免结构被重建）

ultralytics 的 `YOLO().train()` 会调用 `get_model()` 从 YAML 重建模型，**丢失已剪的结构**。
正确做法是用 `DetectionTrainer` 并直接塞入已剪模型，绕过重建：

```python
from ultralytics.models.yolo.detect.train import DetectionTrainer

trainer = DetectionTrainer(overrides=overrides)   # overrides 含 model/data/epochs/...
trainer.model = pruned.model                      # 关键：直接赋值，保留剪枝后的通道宽度
trainer.train()
```

`setup_model()` 检测到 `self.model` 已是 `nn.Module` 会直接返回，因此此写法有效（`pruner.py: finetune` 已验证）。

---

## 五、本轮结果速查（YOLO11n，held-out val2017，基准 0.386）

| 方法 | raw | 微调后 val2017 | 参数降幅 |
|---|---:|---:|---:|
| L1 | 0.314 | 0.340 | 3.7% |
| L2 | 0.315 | 0.340 | 3.7% |
| FPGM | 0.326 | 0.337 | 3.7% |
| **Taylor** | **0.336** | **0.344** | 3.7% |
| BN-γ（无稀疏） | 0.012 | 0.195 | 3.7% |
| Network Slimming | 0.004 | 0.096 | 3.7% |
| 迭代剪枝 | — | 0.329 | 4.0% |
| **SFP** | — | **0.357** | **2.6%** |
| HRank | — | 0.319 | 3.7% |
| KD 蒸馏 | — | 0.333 | 3.7% |

> 评测口径：必须用 held-out val2017。coco128 的 26-val 会系统性高估 ~0.24 且低估剪枝损失，
> 本仓库早先"independent > greedy"的结论就是被这个口径污染。

---

## 六、真正能打开剪幅上限的方向（按杠杆排序）

1. **量化 INT8**：体积 −75%、速度 +2~3×，几乎不掉点——最大的免费杠杆。
2. **对 C2f split/concat 做成对结构化剪枝**：大头全在这些耦合层，需要保证 concat 两侧通道一致，剪幅可从 3.6% 提到 30%+。
3. **高剪枝率 + 完整 COCO2017 长微调**：让 Taylor/SFP 的优势真正显现，恢复上限更高。
4. **换更大模型（YOLO11s/m）**：大模型冗余多，剪枝收益更明显。

---

## 边界与局限

- 数据仅 coco128（128 训练 / 26 验证），精度天花板 ~0.386；微调预算仅 50ep，恢复不充分。
- 仅剪 9 个"安全层"（mAP50-95 下降 ≤ 0.005），未触及更敏感但占体积更大的层。
- 结果在同一划分上评测，属开发集对比，不能据此宣称泛化提升。

方法依赖：[Torch-Pruning](https://github.com/VainF/Torch-Pruning)、[ultralytics](https://github.com/ultralytics/ultralytics)。
