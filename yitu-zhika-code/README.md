# 一图知卡 — 多光谱食物识别与卡路里估计平台

> 课程设计复现：*Multi-Spectral Food Classification and Caloric Estimation Using Predicted Images*（Foods 2024, 13(4):551）
> **最终产品形态是微信小程序**，后端是 Flask 推理服务；`app/gradio_demo.py` 只是历史内部 Demo，不是产品入口。

## 当前状态（2026-09-10 审计）

本 README 的指标已于 2026-09-10 重新核对，作废的数字包括 `19.29 dB / 0.799 / 16.97% / 29.92% / R² 0.9403`
（早期版本、不同划分与不同 MAPE 口径的产物）。当前口径如下：

| 阶段 | 当前结果 | 证据 |
|------|----------|------|
| 阶段一 RGB→NIR（小数据） | PSNR 22.84 dB / SSIM 0.837（训练 93 / 测试 33 扫描） | `results/` 阶段一汇总 |
| 阶段一 RGB→NIR（全量，三种子） | **PSNR 26.674 dB / SSIM 0.9084**（2772 / 290 / 327） | 跨种子极差 0.116 dB / 0.0015 |
| 阶段一 队友实现（对照） | 14.25 dB / 0.622 —— NIR 与 RGB 差 90°，对齐后相关 0.99965 | `scripts/audit_hsi_alignment.py` |
| 阶段二 RGB vs RGB+NIR | 热量 MAE 56.55 → 56.21 kcal，**配对 bootstrap CI 跨零** | `scripts/analyze_paired_nir.py` |
| 阶段二 全量生成器复核（ms3） | Δ = −1.74 / −0.48 / −0.01 kcal（均值 −0.74），**CI 仍全跨零** | `scripts/analyze_ms3.py` |
| 在线系统 | 微信小程序 + Flask；回归/宏量 `v1_expanded`，类别 `v3_corrected_seed42`（解耦） | `scripts/check_inference_consistency.py` |

四类指标口径（论文协议不同，**不得直接相减**）：生成器（论文 30.61 dB / 0.865，仅背景区域）、
回归（论文 MAPE 12.13%，零值处理不同）、分类（论文 98.24%，类别集合不同）。

## 项目结构

### 核心代码 `src/`

| 模块 | 描述 | 入口文件 |
|------|------|----------|
| [数据加载器](src/data/README.md) | HSI 波段/清单、Nutrition5k 与扩展清单、标签体系构建 | `src/data/hsi_dataset.py` |
| [模型定义](src/models/README.md) | 阶段一 U-Net 生成器与判别器、阶段二多任务 ResNet50、五目标宏量网络 | `src/models/resnet_multitask.py` |
| [训练管线](src/training/README.md) | 阶段一训练、阶段二多任务/宏量/消融训练、检查点管理 | `src/training/train_hsi_full_v3.py` |
| [评估工具](src/evaluation/README.md) | 阶段一 PSNR/SSIM、阶段二 MAPE/R²/散点图 | `src/evaluation/eval_generator.py` |

### 在线服务 `app/`

| 模块 | 描述 |
|------|------|
| `inference_service.py` | Flask 服务：`/predict`、`/record`、`/model-info`、`/merchant/*`、`/auth/wx-login`；X-API-Key 鉴权、限流、EXIF 剥离 |
| `experiment_pipeline.py` | 实验推理编排：ImageNet 食物过滤 → RGB/NIR 两目标对照 → 五目标回归 → 类别模型 |
| `contracts.py` | 请求/响应契约与溯源字段（`source_model`+SHA、`category_model`、`label_schema`、`abnormal_fields`） |
| `collection.py` | SQLite 采集与记录（`data/collection_v1.sqlite3`），训练授权与普通记录分离 |
| `cloud_pipeline.py` | `YITU_RUNTIME=cloud_primary` 时切换的云端运行时 |

### 小程序 `miniprogram/`

三 tab：今天（拍照→结果→修正→历史）、饮食计划（TDEE + 7 天食谱）、我的。
结果页展示热量/重量/蛋白/碳水/脂肪/类别与置信度分布，异常或负值显示「待确认」（不裁零、不伪造）。

### 实验与统计脚本 `scripts/`

`build_hsi_full_dataset.py`（全量 memmap 数据集）、`analyze_ms3.py` / `analyze_paired_nir.py`（配对检验）、
`summarize_phase1.py` / `summarize_ms2.py`（汇总）、`audit_*.py`（各类审计）、`make_nir_preview.py`（配图）。

### 资源目录

| 目录 | 描述 |
|------|------|
| `configs/` | 默认配置 |
| `data/` | 数据集与 memmap（`data/hsi_full_v3/`）、SQLite 采集库 |
| `checkpoints/` | 模型检查点 |
| `results/` | 评估产物（`meal_exp_ms2_rgb_seed*`、`meal_exp_ms3_rgbnir_seed*`、散点图、预览图） |
| `docs/` | 报告数据包、论文对比、各阶段结论与审计 |
| `artifacts/` | 答辩 PPT、PPT 内容 JSON 与构建脚本、锚点量化等产物 |
| `attic/` | 已裁剪内容归档（侧视角、公网部署） |

## 快速开始

```bash
conda create -n yitu python=3.11
conda activate yitu
pip install -r requirements.txt
```

### 在线服务（当前主入口）

```bash
python app/inference_service.py          # Flask 推理服务
# YITU_RUNTIME=cloud_primary python app/inference_service.py   # 切云端运行时
python app/gradio_demo.py                # 仅历史内部 Demo，非产品入口
```

### 阶段一：RGB→NIR 生成器

```bash
# 全量数据集构建（队友 h5 → memmap，含 90° 旋转与仿射标定）
python scripts/build_hsi_full_dataset.py

# 全量训练（拒绝覆盖非空输出目录，.tmp 原子写）
python src/training/train_hsi_full_v3.py --out results/hsi_full_seed42 --seed 42

# 小数据协议训练 / 官方子集训练
python src/training/train_generator.py
python src/training/train_hsi_official.py
```

### 阶段二：多任务与消融

```bash
python src/training/train_meal_official.py            # RGB / RGB+NIR 对照（--generator 指定生成器）
python src/training/train_multispectral_experiment.py --generator <path>
python src/training/train_meal_macros.py              # 五目标（--nonneg-output / --class-weight-ce 为消融开关）
python src/training/train_meal_ablation.py            # B/C 消融
python src/evaluation/eval_multitask_full.py          # 评估
```

### 答辩 PPT 构建

```powershell
$env:PYTHONPATH = "<repo>\.pptx_libs"     # 仓库内 vendored python-pptx（cp314）
& "C:\Users\user\AppData\Local\Python\pythoncore-3.14-64\python.exe" artifacts\build_defense_ppt.py
```

## 当前指标

### 阶段一：RGB→NIR 生成器

| 版本 | 训练样本 | PSNR | SSIM |
|------|----------|------|------|
| 路径 A 小数据 | 93 | 22.84 dB | 0.837 |
| 路径 A 全量（三种子均值） | 2772 | **26.67 dB** | **0.9084** |
| 路径 B 队友（朝向错位） | 2772 | 14.25 dB | 0.622 |
| 论文报告（口径不同，未直接比较） | — | 30.61 dB | 0.865 |

> 口径：小数据模型与全量模型都在**同一批 327 张全量 test** 上评估（小数据模型在自己 33 张测试集上另有 23.33 dB / 0.812），
> 因此前两行可直接比较。

优化步数 1150 → 9700（8.4×）只换来 +3.8 dB（MSE 降至约 41.6%）；**锚点决定能不能学到，数据量决定学得多细**。

| 锚点单因素消融（seed 42，结构/损失/轮数/种子完全一致） | PSNR | SSIM | Δ |
|--------------------------------------------------------|------|------|---|
| 基线 `full_seed42`（固定缩放 + 朝向对齐） | 26.654 dB | 0.9093 | — |
| 目标 NIR 旋转 90°（`--extra-rot 1`） | 18.928 dB | 0.7160 | **−7.73 dB / −0.193** |
| 逐图百分位归一化替代固定缩放（`--target-norm per_image`） | 21.062 dB | 0.8373 | **−5.59 dB / −0.072** |

两次落差分别是跨种子极差（0.116 dB）的 **67 倍 / 48 倍**；队友那版的 14.25 dB 是「朝向错位 + 不同结构/损失/轮数」
的混合结果，**不能当作单因素结论引用**。

> 评测口径：PSNR = `10·log10(4/MSE)`（值域 `[-1,1]`，逐图平均，与训练日志同口径）；SSIM 用 11×11 高斯窗。
> 每次运行都记录清单 SHA、代码 SHA、标定参数与消融开关，`results/hsi_full_v3/<tag>/test_metrics.json` 存检查点 SHA。

### 阶段二：多通道多任务（冻结 507 测试集）

| 模型 | 热量 MAE | 热量 RMSE | R² | 非零 MAPE (n=506) | 重量 MAE | 粗分类 |
|------|----------|-----------|----|-------------------|----------|--------|
| RGB（内部对照） | 56.55 kcal | 85.13 | 0.8388 | 48.07% | 36.70 g | 72.78% |
| RGB + 预测 NIR（小数据生成器） | 56.21 kcal | 84.90 | 0.8397 | 51.02% | 35.64 g | 73.18% |
| CalorieCLIP 基线（仅热量） | 58.74 kcal | 91.29 | 0.8146 | 35.36% | 不支持 | 不支持 |

| 配对检验 | 热量 Δ（三种子） | 结论 |
|----------|------------------|------|
| 小数据生成器 | +2.32 / −2.98 / +3.19 kcal | 符号翻转、CI 全跨零 → 无稳定增益 |
| 全量生成器（ms3） | −1.74 / −0.48 / −0.01 kcal（均值 −0.74） | 方向一致偏 NIR，但 CI 仍全跨零、幅度 ≈ 1 kcal 噪声底 → **仍不显著** |

### 五目标在线模型（冻结 507 测试集，MAE）

| 模型 | 热量 | 重量 | 蛋白 | 碳水 | 脂肪 | 粗分类 |
|------|------|------|------|------|------|--------|
| `v1_expanded`（在线回归/宏量，11 类） | 58.50 kcal | 36.14 g | 5.71 g | 6.10 g | 4.32 g | 73.77% |
| `v3_corrected_seed42`（在线类别，12 类） | 60.16 kcal | 37.63 g | 5.68 g | 6.16 g | 4.44 g | 69.03% |

### 消融实验

**阶段二（冻结 507 测试集）**

| 实验 | 内容 | 结果 | 是否采用 |
|------|------|------|----------|
| B | 非负输出（softplus） | 负预测 0/0/0/0/0，但热量 MAE 62.98 kcal（+4.5）、重量 44.49 g（+8.4） | **不上线** |
| C | 类别加权 CE | 平衡召回 0.5150 → 0.5384，Macro-F1 0.5283 → 0.5371，总体准确率 0.7357 → 0.7318 | 可选开关，**非默认** |
| — | RGB vs RGB+NIR × 3 种子 | 逐餐盘配对 bootstrap，置信区间全跨零（含全量生成器复核） | **不为 NIR 增加部署复杂度** |

**阶段一（生成器锚点，单因素，seed 42）**

| 实验 | 内容 | 结果 |
|------|------|------|
| A1 | RGB 分支是否被改动 | 144 张扫描上逐像素平均绝对差**恰为 0.0000** → NIR 实验不污染 RGB 对照 |
| A2 | NIR 对齐校验 | 旋转 90° → `np.rot90(nir, 3)` + 仿射标定，对齐后平均绝对差 0.0029 / 相关 0.99965 / R² 0.9990 |
| A3 | NIR 目标朝向 | PSNR 26.654 → 18.928 dB（−7.73）、SSIM 0.9093 → 0.7160 |
| A4 | NIR 目标归一化 | 逐图归一化：PSNR → 21.062 dB（−5.59）、SSIM → 0.8373 |

分类必须同时报 Macro-F1 / 平衡召回 / 逐类支持度：`soup_stew`、`sauce_condiment` 测试样本为 0，`dessert` 验证样本为 0，
**Macro-F1 只在 9 个受支持类别上报告**。NIR 臂准确率略升（0.7278 → 0.7318）但 Macro-F1 与平衡召回下降。

## 技术栈

- Python 3.11 + PyTorch 2.x + CUDA 12.x，NVIDIA RTX 5060 Laptop (8GB)
- 在线服务：Flask + 微信小程序；SQLite 采集库
- 阶段一：4 层 U-Net + L1（在 [-1,1] 域）；Adam 2e-4 β(.5,.999)；ReduceLROnPlateau 0.5/5；选模 = 验证集逐图 L1
- 阶段二：ResNet50 4 通道（第 4 通道用 RGB 均值初始化），掩码归一化 L1 + 0.2×CE；AdamW 1e-5/1e-4，wd 1e-4，bf16，grad clip 1.0，早停 patience 8

## 数据集

| 数据集 | 用途 | 规模 |
|--------|------|------|
| HSIFoodIngr-64 | 阶段一生成器 | 小数据协议 144 对（93 / 18 / 33）；全量 3389 对 = 2772 / 290 / **327** |
| Nutrition5k | 阶段二多任务 | 2188 train / 567 val / 507 冻结 test；扩展清单 3390 |
| FoodData Central | 食谱营养库 | — |

## 项目路线

- [x] 阶段一 生成器：小数据协议跑通 → 全量数据重训（26.674 dB / 0.9084，三种子）
- [x] 锚点实验：旋转 90°（−7.73 dB）与逐图归一化（−5.59 dB）的单因素量化证据
- [x] 阶段二 RGB / RGB+NIR 对照 + 逐餐盘配对 bootstrap（10000 次）
- [x] 全量生成器复核 NIR 臂（ms3，三种子）
- [x] 消融 B（非负输出）/ C（类别加权 CE）+ 在线解耦决策
- [x] 五目标扩展与 12 类标签体系修正
- [x] 在线 Flask 服务 + 微信小程序 + 一致性校验（`all_pass: true`）
- [x] 答辩 PPT（18 页，含备注与配图）
- [ ] 课程设计报告正文（数据包已备齐）
- [ ] B/C 消融补 2 个种子复核
- [ ] 训练权重打包上传（约 1.6 GB / 6 个文件）
- [ ] 后续：显式确定性设置、多角度与真机评测集

**明确不做**：真机部署与公网发布、侧视角几何矫正、产品化加密/防抄设计（第五阶段）。

## 诚实边界

- 不得声称「NIR 显著提升」；不得声称「适合真实用户」；不得把论文数字直接相减。
- MAPE 只在非零子集（n=506）上成立，必须标注口径。
- 「蔬菜类偏置」未解决；无测试支持的类别不得宣称验收通过。

## 参考

- 论文：*Multi-Spectral Food Classification and Caloric Estimation Using Predicted Images*（Foods 2024, 13(4):551, PMC10887625）
- 数据：HSIFoodIngr-64、Nutrition5k、FoodData Central
- 基线：CalorieCLIP（本地权重 strict 加载，仅输出热量 → 降级为参考行）
