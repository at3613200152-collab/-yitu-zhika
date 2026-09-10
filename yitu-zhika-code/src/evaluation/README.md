# 评估工具模块

阶段一生成器质量评估 + 阶段二多任务模型评估。

> **当前状态（2026-09-10 审计）**：本文件早期版本的 `19.29 dB / 0.799 / MAE 25.38 kcal / MAPE 16.97% / 29.92% / R² 0.9403`
> 均来自早期模型、早期划分与不同的 MAPE 口径，**已作废**（其中热量/重量两列当时还对调过）。
> 当前对外数字见下表，口径逐条注明。

## 模块清单

| 文件 | 描述 | 指标 |
|------|------|------|
| [eval_generator.py](eval_generator.py) | 阶段一生成器评估（7 层 `models.nir_generator.UNetGenerator`） | PSNR, SSIM, L1 |
| [eval_multitask.py](eval_multitask.py) | 阶段二多任务评估（4 层 `models.generator.UNetGenerator` 供 NIR 分支） | MAE, MAPE, RMSE, R² |
| [eval_multitask_full.py](eval_multitask_full.py) | 阶段二全量/冻结测试集评估 | 同上 |

## 阶段一评估口径（**项目内有两种 PSNR 口径，不要混用**）

| 口径 | 公式 | 用在哪些数字上 |
|------|------|----------------|
| **主口径** | `10·log10(1/MSE)`，`[0,1]` 域，**逐图 PSNR 再平均** | 本文档表中全部数字：26.674 / 26.654 / 22.84 / 23.33 / 18.928 / 21.062 / 14.25 |
| 批 MSE 口径 | `10·log10(1/MSE)`，`[0,1]` 域，先全局平均 MSE | `test_metrics.json` 的 `test_psnr_batch_mse_db`（25.39） |
| 遗留口径 | `10·log10(4/MSE)`，`[-1,1]` 域 | 仅 `train_generator.py`（7 层）与 `eval_generator.py`；与主口径相差一个常数 |

主口径实现在 `src/training/train_hsi_full_v3.py::metrics()`（`src/training/train_hsi_official.py`、
`artifacts/eval_generators_same_test.py`、`scripts/make_nir_preview.py` 同口径），因此跨脚本的数字可直接比较。

> ⚠️ **已知口径瑕疵（2026-09-11 定位）**：`train_hsi_full_v3.py` 的 test 阶段只做 `torch.set_grad_enabled(False)`，
> **没有调用 `model.eval()`**，因此上表数字是在 **BN 使用 batch 统计**（batch=8）下测得的。
> 阶段二推理走的是 `NIR = generator(...)` 且显式 `.eval()`，同一权重在 eval 模式下为 **27.27 dB**（批 MSE 口径 25.98）。
> 影响：绝对值偏低约 0.6 dB，方向与量级结论不受影响（锚点消融三次运行同一口径）。
> 复现脚本：`scripts/analyze_nir_domain_gap.py`（同时打印两种模式的 PSNR 作自校验）。

- **SSIM**：11×11 高斯窗，`range = 1`（主口径）；旧脚本用 `data_range=2.0` 时不可与上表直接比较。
- **L1**：像素级绝对误差（另报 `l1_01`，即 `[0,1]` 域的 L1）。

### 当前结果（全量数据，固定缩放，327 张独立测试样本）

| 版本 | 训练样本 | PSNR | SSIM | 说明 |
|------|----------|------|------|------|
| `full_seed42` | 2772 | 26.654 dB | 0.9093 | — |
| `full_seed43` | 2772 | 26.626 dB | 0.9081 | — |
| `full_seed44` | 2772 | 26.741 dB | 0.9078 | — |
| 三种子均值 | 2772 | **26.674 dB** | **0.9084** | 跨种子极差仅 0.116 dB / 0.0015 |
| 小数据版（路径 A） | 93 | 22.84 dB | 0.837 | 与全量版**同一批 327 张 test**，故可直接比较 |
| 小数据版（自测集） | 93 | 23.33 dB | 0.812 | 仅在自己 33 张 test 上 |
| 路径 B 队友实现 | 2772 | 14.25 dB | 0.622 | NIR 与 RGB 差 90° 未对齐（见下） |
| 论文报告 | — | 30.61 dB | 0.865 | **口径不同**（论文仅背景区域），不得直接比较 |

### 锚点消融（单因素，seed 42，其余设置与 `full_seed42` 完全一致）

| 消融 | PSNR | SSIM | Δ 相对 `full_seed42` |
|------|------|------|----------------------|
| 基线 `full_seed42` | 26.654 dB | 0.9093 | — |
| 把目标 NIR 旋转 90°（`--extra-rot 1`） | 18.928 dB | 0.7160 | **−7.73 dB / −0.193** |
| 逐图百分位归一化替代训练集固定缩放（`--target-norm per_image`） | 21.062 dB | 0.8373 | **−5.59 dB / −0.072** |

两次消融的落差分别是跨种子极差（0.116 dB）的 **67 倍 / 48 倍**，因此锚点影响的判定不受运行噪声干扰。
这也解释了队友那版为什么只有 14.25 dB：它是「朝向错位 + 不同结构/损失/轮数」的混合结果，不能当作单因素结论引用。

## 阶段二评估口径

- 冻结 **507** 测试划分，只评估一次（选模完全依赖验证集）。
- 未知标签 `mask=0` 不参与损失与 MAE，真实 0 保留；
- **MAPE 只在非零子集（n=506）上有意义**，必须标注，不能与论文的 MAPE 直接相减；
- 保留负预测原值（不裁零），异常值在接口层标记为「待确认」。

| 指标 | RGB（内部对照） | RGB + 预测 NIR |
|------|-----------------|----------------|
| 热量 MAE | 56.55 kcal | 56.21 kcal |
| 热量 RMSE | 85.13 | 84.90 |
| R² | 0.8388 | 0.8397 |
| 非零 MAPE (n=506) | 48.07% | 51.02% |
| 重量 MAE | 36.70 g | 35.64 g |
| 粗分类准确率 | 72.78% | 73.18% |

逐餐盘配对 bootstrap（10000 次重采样）：热量 Δ = +2.32 / −2.98 / +3.19 kcal（符号翻转、CI 全跨零）；
换用全量生成器后 Δ = −1.74 / −0.48 / −0.01 kcal（均值 −0.74，CI 仍全跨零）。
**结论：未观察到预测 NIR 的稳定增益。** 评估脚本：`scripts/analyze_paired_nir.py`、`scripts/analyze_ms3.py`。

## 评估产物

- `results/<run>/test_metrics.json` — 单次运行的测试指标 + 检查点 SHA
- `results/<run>/epochs.json` — 逐 epoch 曲线
- `results/course_report_plots/` — 散点图与混淆矩阵（`calorie_scatter_3models.png`、`confusion_matrix_{rgb,nir}.png`、`paired_delta_histogram.png`）
- `results/nir_preview/` — 生成器可视化对照（`nir_three_versions_0.png`）

## 论文参考数字（**不同口径**）

| 论文 | 本项目对应量 | 为什么不能直接比较 |
|------|--------------|--------------------|
| PSNR 30.61 dB / SSIM 0.865 | 26.674 dB / 0.9084 | 论文只在背景区域计算 PSNR，测试数据为作者自采 |
| 热量 MAPE 12.13% | 48.07%（RGB，非零子集 n=506） | MAPE 零值/负值处理不同 |
| 识别率 98.24% | 粗分类 72.78%（11 类） | 类别集合与统计方式不同 |
