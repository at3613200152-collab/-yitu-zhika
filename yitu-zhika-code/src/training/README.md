# 训练管线模块

Phase1 RGB→NIR 生成器训练 + Phase2 多任务（RGB / RGB+NIR 对照）训练 + 检查点管理。

## 当前状态（2026-09-10）

> 本文件中的协议、指标、数据集与路线图最后一次审计于 **2026-09-10**。
> 早前文档中的 `nirscene1` / `capsicum` 数据源、"123 样本 × 5 权重"、`19.29 dB`、`0.799`、`16.97%`、`29.92%`、`R² 0.9403` 等数字均已作废，请以下文为准。
> **诚实汇报口径**：NIR 分支目前**没有**统计显著的增益；与论文数字**不可直接相减**比较；系统**不是**可交付给真实用户的状态；"蔬菜偏置"问题**尚未解决**。

## 模块清单

| 文件 | 描述 | 阶段 |
|------|------|------|
| [train_generator.py](train_generator.py) | RGB→NIR 生成器训练 | Phase1 |
| [train_hsi_full_v3.py](train_hsi_full_v3.py) | 全量数据生成器训练（`data/hsi_full_v3/` memmap） | Phase1 |
| [train_multitask.py](train_multitask.py) / [train_multitask_v2.py](train_multitask_v2.py) | 多任务 ResNet50 训练 | Phase2 |
| [train_meal_nir_official.py](train_meal_nir_official.py) | RGB+NIR 4 通道对照实验（正式跑法） | Phase2 |
| [train_meal_official.py](train_meal_official.py) | RGB 单臂对照实验 | Phase2 |
| [train_meal_macros.py](train_meal_macros.py) | 宏量营养素回归（在线模型 `v1_expanded`） | Phase2 / 在线 |
| [train_meal_ablation.py](train_meal_ablation.py) / [meal_ablation_core.py](meal_ablation_core.py) | 消融实验（B 非负输出、C 类别加权 CE、锚点实验） | 消融 |
| [checkpoint.py](checkpoint.py) | 检查点保存/加载/恢复 | 通用 |
| [train_capsicum.py](train_capsicum.py) | 早期 capsicum 实验（**历史遗留，非当前路线**） | 历史 |

## Phase1 训练流程（生成器，RGB→NIR）

**当前协议（2026-09-10 审计）**

1. 数据：HSIFoodIngr-64 的 RGB/NIR 配对扫描，NIR 取 **860 nm** 波段；NIR 需 `np.rot90(nir, 3)` + 仿射标定对齐到 RGB（见 `src/data/README.md`）
2. 归一化：**仅用训练集统计量的固定缩放** low=0.17514 / high=1.67553，输入输出统一在 `[-1,1]`
3. 损失：**L1**（`[-1,1]` 空间）。早前文档里描述的 PatchGAN 对抗损失 / λ=100 组合不属于当前审计协议
4. 优化器：Adam，lr 2e-4，betas (0.5, 0.999)
5. 调度：ReduceLROnPlateau，factor 0.5，patience 5
6. 模型选择：验证集**逐图 L1** 最优
7. 全量数据训练由 `scripts/build_hsi_full_dataset.py` 先构建 memmap 数据集，再由 `train_hsi_full_v3.py` 训练（`--seed` / `--epochs` / `--batch` / `--target-norm fixed`）

**训练步数对比**：小数据 96 × 12 ≈ **1150** 步；全量数据 28 × 347 ≈ **9700** 步（约 **8.4×**）。

## Phase1 结果（小数据版与全量版都在同一批 327 张测试样本上比较）

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| 小数据模型（144 扫描协议：train 93 / val 18 / test 33） | 23.33 dB | 0.812 | 评估于**自己的 33 张**测试集 |
| 同一个小数据模型 | 22.84 dB | 0.837 | 评估于**全量 test 327 张**（与全量版同口径，故可直接比较） |
| **全量数据生成器（train 2772，3 个种子）** | **26.674 dB** | **0.9084** | 同一批 327 张测试；跨种子极差 0.116 dB / 0.0015 |
| 队友自训生成器 | 14.25 dB | 0.622 | 混合差异，原因见下 |

> 注意：并不存在"小数据模型重训 327 张"这一配置 —— 327 是**全量数据集的 test 划分**，
> 小数据模型与全量模型都在它上面评估，这正是两者可比的依据。

- 全量数据相对小数据提升约 **+3.8 dB**（22.84 → 26.65，同一批 327 张测试），对应 MSE 降至原先的约 **41.6%**
- 与队友生成器差距的原因已定位：**其 NIR 相对 RGB 旋转了 90°**，配对目标本身错位；对齐后平均绝对差 0.0029、相关系数 0.99965、标定 R² = 0.9990
  （单因素消融见 `src/evaluation/README.md`：仅旋转 90° 就损失 **7.73 dB**）
- **论文参考值 30.61 dB / 0.865 采用不同协议（仅背景区域）测量，不可与本表数字直接相减比较**

## Phase2 训练流程（4 通道 ResNet50 多任务）

1. 数据：Nutrition5k，2188 训练 / 567 验证 / **507 冻结测试** 菜品；回归目标统计量**仅由训练集**计算
2. 输入：RGB 单臂（3 通道）或 RGB+生成 NIR（4 通道）；对照两臂除输入通道外配置一致
3. 模型：ResNet50 多任务头 —— 分类（CrossEntropy）+ 卡路里/重量回归（**掩码归一化 L1**）
4. 训练：30 epochs，batch 8，AdamW（backbone lr 1e-5、heads lr 1e-4），weight decay 1e-4，bf16 autocast，梯度裁剪 1.0，早停 patience 8
5. 模型选择：验证集**归一化 L1** 最优

### Phase2 结果（3 个种子，冻结测试集）

| 实验臂 | 卡路里 MAPE | 卡路里 MAPE（非零子集，n=506） | 重量 MAPE | R² |
|--------|-------------|-------------------------------|-----------|----|
| RGB 单臂 | 56.55% | 48.07% | 85.13% | 0.8388 |
| RGB+NIR（小数据 144 扫描生成器） | 56.21% | 51.02% | 84.90% | 0.8397 |
| RGB+NIR（全量数据生成器 26.674 dB） | 见下方配对检验 | — | — | — |

配对检验（逐菜品 bootstrap，10000 次重采样）：

| 对比 | 卡路里 Δ（每种子） | 重量 Δ（每种子） | 结论 |
|------|-------------------|------------------|------|
| RGB vs RGB+NIR（小数据生成器） | +2.32 / −2.98 / +3.19 kcal（均值 +0.84） | −0.15 / −2.87 / +1.31 | **置信区间全部跨零且符号翻转 → 小数据生成器无稳定增益** |
| RGB 臂 `results/meal_exp_ms2_rgb_seed{42,43,44}` vs RGB+NIR 臂 `results/meal_exp_ms3_rgbnir_seed{42,43,44}` | −1.74 / −0.48 / −0.01 kcal（均值 −0.74，区间 [−1.74, −0.01]） | −0.82 / −1.73 / +0.59（均值 −0.65，符号翻转） | 3/3 种子方向一致为**非正**，但配对 CI 仍跨零，幅度 ≈ 1 kcal 噪声底 → **方向一致，仍不显著，不得宣称 NIR 有显著增益** |

- 运行间噪声底实测约 **1 kcal**
- 因此对外表述只能是"方向倾向有利、统计上不显著"，不能写成"稳定增益"

### 在线模型指标

以下数字均为**冻结 507 测试集上的 MAE**（不是 MAPE；MAPE 只在非零子集 n=506 上有意义，见上文两臂对照表）。

| 模型 | 指标 | 数值 |
|------|------|------|
| `v1_expanded`（在线回归/宏量，3390 行 / 11 类） | 热量 MAE | **58.50 kcal** |
| `v1_expanded` | 重量 MAE | 36.14 g |
| `v1_expanded` | 蛋白 / 碳水 / 脂肪 MAE | 5.71 / 6.10 / 4.32 g |
| `v1_expanded` | 粗分类准确率 | 73.77% |
| `v3_corrected_seed42`（在线类别模型，3390 行 / 12 类） | 粗分类准确率 | 69.03% |

> 在线系统把两者解耦：热量/宏量走 `v1_expanded`（11 类标签组），类别走 `v3_corrected_seed42`（12 类，新增「水果」）。
> 12 类标签体系让水果类 P 0.809 / R 0.867 / F1 0.837，但同配方下热量 MAE 上升约 1.9 kcal（三种子区间不重叠），故不合并。

## 消融实验

| 实验 | 内容 | 结果 | 是否采用 |
|------|------|------|----------|
| A（锚点/健全性） | RGB 分支是否被改动 | 144 张扫描上逐像素平均绝对差**恰为 0.0000** → RGB 分支未被触碰 | 验证通过 |
| A（锚点/健全性） | NIR 对齐校验 | 旋转 90° → `np.rot90(nir, 3)` + 仿射标定，对齐后平均绝对差 0.0029 / 相关 0.99965 / R² 0.9990 | 验证通过 |
| B | 非负输出参数化（softplus） | 负预测 0/0/0/0/0（5 个目标全为 0），但回归**更差**：热量 MAE 62.98 kcal（+4.5）、重量 MAE 44.49 g（+8.4） | **不上线** |
| C | 类别加权 CE（`--class-weight-ce`） | 平衡召回 0.5150 → 0.5384，Macro-F1 0.5283 → 0.5371，混合类召回 0.316 → 0.526，总体准确率 0.7357 → 0.7318（略降） | 对稀有类有帮助，作为**可选开关**，**非默认** |

**分类指标口径**：冻结测试划分中 `soup_stew`（测试集 0 样本）、`dessert`（验证集 0 样本）、`sauce_condiment`（测试集 0 样本）无样本，故 **Macro-F1 仅在 9 个受支持类别上报告**。

## 关键参数（当前）

| 参数 | 当前值 | 说明 |
|------|--------|------|
| Phase1 epochs | 28（全量数据） | 早前的 200 轮不是现行配置 |
| Phase2 epochs | 30 | 早前的 100 轮不是现行配置 |
| batch_size | 8 | 两个阶段一致 |
| Phase1 优化器 | Adam, lr 2e-4, betas (0.5, 0.999) | ReduceLROnPlateau 0.5 / patience 5 |
| Phase1 选择指标 | 验证集逐图 L1 | — |
| Phase2 优化器 | AdamW, backbone 1e-5 / heads 1e-4, wd 1e-4 | bf16 autocast, 梯度裁剪 1.0 |
| Phase2 选择指标 | 验证集归一化 L1 | early stop patience 8 |
| 归一化缩放 | 固定 low=0.17514 / high=1.67553 | 仅由训练集统计，验证/测试复用 |

## 技术栈与环境

- Python 3.11 + PyTorch 2.x + CUDA，单卡 **RTX 5060 Laptop 8GB**
- 在线服务：Flask (`app/inference_service.py`) + 微信小程序 (`miniprogram/`)；Gradio 仅作为历史内部 Demo，**不再是产品入口**
- 数据采集：SQLite `data/collection_v1.sqlite3`
- 训练 / 评估脚本位于 `src/training/` 与 `src/evaluation/`，实验汇总脚本位于 `scripts/`

## 启动方式

```powershell
# Phase1：构建全量数据 memmap 数据集
python scripts/build_hsi_full_dataset.py

# Phase1：全量数据生成器训练（示例：seed 42）
python src/training/train_hsi_full_v3.py --tag full_seed42 --seed 42 --epochs 28 --batch 8 --target-norm fixed

# Phase2：RGB 单臂 / RGB+NIR 对照实验
python src/training/train_meal_official.py
python src/training/train_meal_nir_official.py

# 汇总与配对检验
python scripts/summarize_phase1.py
python scripts/analyze_paired_nir.py
```

## 路线图（2026-09-10）

- [x] Phase1 生成器全量数据训练 + 三种子评估（26.674 dB / 0.9084）
- [x] Phase2 RGB / RGB+NIR 对照 + 配对 bootstrap 检验
- [x] 消融：非负输出(B)、类别加权 CE(C)、旋转/归一化锚点实验
- [x] 在线 Flask 推理服务 + 微信小程序接入
- [ ] 报告与答辩材料
- [ ] 额外种子复跑（B/C 实验）
- [ ] 侧视角 / hold-out 后续实验

**明确不做**：真机部署与公网发布、侧视角几何矫正、第五阶段。
