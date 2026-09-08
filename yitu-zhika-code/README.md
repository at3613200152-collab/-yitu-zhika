# 一图知卡 — 基于近红外光谱的实时食物热量分析平台

> **最终产品：微信小程序。** Gradio Demo 仅承担模型展示与测评，不作为最终应用交付。小程序端及独立推理接口待实现，见 [交付范围](docs/delivery_scope.md)。当前三模型训练、指标审计与 Demo 上传联调已完成，对比见 `results/experiment_comparison_v1.md`。

> **当前入口（2026-09-06）**：请先阅读 [最新验收进度](docs/progress_2026-09-06.md)。新 HSI 生成器和官方 ID 子集 RGB/NIR 对照已完成并审计；新增 128 餐盘 / 384 侧视帧已在 D 盘核验。Demo 已改接新权重并显示 RGB 对比。CalorieCLIP 类基线状态见 `results/calorieclip_official_v1/status.json`，三模型审计完成后自动生成 `results/experiment_comparison_v1.md`。下方旧架构与指标说明保留作历史记录，不代表当前默认模型。

> 2026-09-04 晚更新：已复现并修复第 16 轮的 AMP 溢出处理缺陷，完成真实故障轮重放与 21 项回归检查。旧 Phase2 入口改为 v2 兼容入口，RGB+NIR 模式必须显式提供生成器；详见 [训练诊断与操作边界](docs/training_diagnosis_2026-09-04.md)。扩库尚未下载，见 [数据来源核查](docs/dataset_expansion_sources.md)。

> 2026-09-04 重要勘误：本地 Nutrition5k 原转换表的热量/重量列已查实对调，现已生成核验标签并修正 Demo。旧验证集的正确热量 MAPE 为 29.92%，重量 MAPE 为 16.97%；不是官方测试集成绩。请先阅读 [本轮进度与指标勘误](docs/progress_2026-09-04.md)。当前已加载的生成器是 4 层 U-Net；下方 7 层架构是项目中另一个实现，不代表当前权重。当前历史权重不提供可靠食物分类。

> 两阶段框架：Phase1 RGB→NIR 生成器（7层 Deep U-Net + PatchGAN）+ Phase2 RGB+NIR 4ch 多任务 ResNet50（卡路里 + 重量 + 分类）

## 项目结构

### engineering/ — 核心代码 `src/`

| 模块 | 描述 | 入口文件 |
|------|------|----------|
| [数据加载器](src/data/README.md) | HSI ENVI格式 + Pix2PixHD格式 + Nutrition5k CSV 多源数据加载 | [pix2pix_dataset.py](src/data/pix2pix_dataset.py) |
| [模型定义](src/models/README.md) | Phase1 7层U-Net生成器/判别器 + Phase2 多任务ResNet50 + 损失函数 | [nir_generator.py](src/models/nir_generator.py) |
| [训练管线](src/training/README.md) | Phase1 Pix2Pix训练 + Phase2 多任务训练 + 检查点管理 | [train_generator.py](src/training/train_generator.py) |
| [评估工具](src/evaluation/README.md) | Phase1 PSNR/SSIM评估 + Phase2 MAPE/R²/散点图 | [eval_generator.py](src/evaluation/eval_generator.py) |

### in-progress/ — 测试与管线验证 `src/`

| 模块 | 描述 | 入口文件 |
|------|------|----------|
| Phase2全链路测试 | 7环节管线验证脚本 | [test_phase2_pipeline.py](src/test_phase2_pipeline.py) |
| 测试后训练流程 | 全链路测试→训练一体化 | [run_test_then_train.py](src/run_test_then_train.py) |

### deprecated/ — 废弃文件归档 `archive/`

| 说明 | 详情 |
|------|------|
| 旧版启动脚本 | auto_train.py, launch_bg.py, launch_v2.py |
| 旧版批处理 | start_training.bat, run_phase1_new.bat, start_download.bat 等 |
| 旧版日志 | download_data.log, phase2_err.log, phase2_run.log |

详见 [archive/README.md](archive/README.md)

### 资源目录

| 目录 | 描述 |
|------|------|
| [configs/](configs/) | 默认配置文件（default.yaml） |
| [data/](data/) | 数据集目录（nirscene1_x10, HSIFoodIngr-64, capsicum） |
| [checkpoints/](checkpoints/) | 模型检查点（phase1/, multitask/） |
| [results/](results/) | 评估产物（metrics, scatter_plot, training_curves） |
| [logs/](logs/) | TensorBoard训练日志 |

### 根目录文件

| 文件 | 用途 |
|------|------|
| `run_phase1_bg.py` | Phase1 一键启动脚本（下载→解压→训练） |
| `verify_data.py` | 数据完整性验证脚本 |
| `.gitignore` | Git忽略规则 |
| `requirements.txt` | Python依赖清单 |

## 快速开始

### 环境准备

```bash
# 创建conda环境
conda create -n yitu python=3.11
conda activate yitu
pip install -r requirements.txt
```

### Phase1: RGB→NIR 生成器训练

```bash
# 一键启动（自动下载数据→解压→加载预训练→训练）
pythonw run_phase1_bg.py

# 手动启动训练（数据已就绪时）
python src/training/train_generator.py --pretrained <pretrained_weights.pth>

# 评估生成器质量
python src/evaluation/eval_generator.py --checkpoint checkpoints/phase1/final_model.pth
```

### Phase2: 多任务 ResNet50 训练

```bash
# 训练多任务模型（卡路里+重量+分类）
python src/training/train_multitask_v2.py --config configs/default.yaml --epochs 100 --generator_ckpt checkpoints/phase1/final_model.pth

# 评估多任务模型
python src/evaluation/eval_multitask.py --checkpoint checkpoints/multitask/best_model.pt
```

## 当前指标

### Phase1: RGB→NIR 生成器

| 指标 | 旧模型(4层U-Net) | 新模型(7层U-Net) | 论文参考 |
|------|-----------------|-----------------|----------|
| PSNR | 19.29±2.10 dB | 🔄 重训中 | 30.61 dB |
| SSIM | 0.799±0.040 | 🔄 重训中 | 0.865 |

> Phase1重训采用队友7层Deep U-Net架构 + 预训练权重(PSNR 25.8)，修复NIR归一化bug后PSNR从-27dB恢复至16dB+

### Phase2: 多任务预测

| 指标 | 当前结果 | 论文参考 |
|------|----------|----------|
| 卡路里 MAPE | 16.97% | 12.13% |
| 卡路里 R² | 0.9403 | — |
| 重量 MAPE | 29.92% | — |
| 重量 R² | 0.9036 | — |

## 技术栈

- **Phase1 生成器**: 7层 Deep U-Net（46.9M参数），支持预训练权重fine-tune
- **Phase1 判别器**: PatchGAN Discriminator（2.8M参数）
- **Phase2 模型**: ResNet50 4通道输入（RGB+NIR），多任务头（卡路里回归+重量回归+分类）
- **数据集**: deepNIR nirscene1 (2597对) + HSIFoodIngr-64 (123样本×5权重) + capsicum (592对)
- **GPU**: NVIDIA RTX 5060 Laptop (8GB GDDR7, Blackwell)

## 参考

- 论文：Multi-Spectral Food Classification and Caloric Estimation Using Predicted Images (PMC10887625)
- 数据集：deepNIR (nirscene1 + capsicum), HSIFoodIngr-64, Nutrition5k
- 预训练模型：队友7层U-Net，PSNR 25.8dB（epoch 27）
