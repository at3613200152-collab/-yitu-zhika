# 一图知卡 — 基于近红外光谱的实时食物热量分析

> RGB -> NIR 生成 + 多任务营养估计的端到端食物热量分析平台

## 项目概述

两阶段框架：
1. **Phase1**: RGB->NIR 生成器 — 7层Deep U-Net + PatchGAN判别器
2. **Phase2**: RGB+NIR 4通道多任务ResNet50 — 同时预测食物类别、卡路里、重量

## 模块导航

### 核心引擎 (engineering)

| 模块 | 描述 | 文档 |
|------|------|------|
| [app/](app/) | Gradio Web应用 + 推理管线 | [README](app/README.md) |
| [models/generator/](models/generator/) | NIR生成器 (7层/4层U-Net + PatchGAN) | [README](models/generator/README.md) |
| [models/multitask/](models/multitask/) | 多任务营养估计网络 (ResNet50 4ch) | [README](models/multitask/README.md) |
| [data/](data/) | 数据加载器 (Nutrition5k + HSIFoodIngr) | [README](data/README.md) |
| [training/](training/) | 训练脚本 (Phase1 GAN + Phase2 多任务) | [README](training/README.md) |
| [evaluation/](evaluation/) | 评估工具 (PSNR/SSIM/MAE/MAPE) | [README](evaluation/README.md) |
| [configs/](configs/) | 超参数配置 | [README](configs/README.md) |

### 用户工具 (productivity)

| 模块 | 描述 | 文档 |
|------|------|------|
| [recipe/](recipe/) | TDEE计算 + 食谱推荐 + 食物数据库 | [README](recipe/README.md) |

### 实验性功能 (in-progress)

| 模块 | 描述 | 文档 |
|------|------|------|
| [optimizations/](optimizations/) | 注意力机制/多波段/分割等实验方案 | [README](optimizations/README.md) |

## 快速开始

### 环境要求
- Python 3.11+
- PyTorch 2.x + CUDA
- GPU: 8GB+ VRAM (训练), CPU也可推理

### 安装
```bash
pip install -r requirements.txt
```

### 启动应用
```bash
# 确保checkpoint文件在以下位置:
#   checkpoints/phase1/final_model.pth  (NIR生成器)
#   checkpoints/multitask/best_model.pt (多任务网络)
python app/gradio_demo.py
# 访问 http://localhost:7860
```

### 训练
```bash
# Phase1: RGB->NIR 生成器
python training/train_generator.py --epochs 200 --batch_size 8

# Phase2: 多任务营养估计
python training/train_multitask.py --config configs/default.yaml
```

## 模型架构

### Phase1: NIR生成器
- **nir_generator.py**: 7层Deep U-Net (46.9M参数), 编码器256->2, 解码器1->256
- **generator_v1.py**: 4层U-Net (早期版本, checkpoint兼容)
- **discriminator.py**: 70x70 PatchGAN条件判别器 (2.8M参数)
- 输入: RGB [B,3,256,256] -> 输出: NIR [B,1,256,256]

### Phase2: 多任务网络
- **resnet_multitask.py**: ResNet50改编, 4通道输入 (RGB+NIR)
- 输出: 61类食物分类 + 卡路里回归 + 重量回归

## 当前性能

| 指标 | Phase1 (NIR生成) | Phase2 (营养估计) | 论文参考 |
|------|------------------|-------------------|----------|
| PSNR | 19.29 dB | — | 30.61 dB |
| SSIM | 0.799 | — | 0.865 |
| 卡路里MAPE | — | 16.97% | 12.13% |
| 重量MAPE | — | 29.92% | — |
| R2 (卡路里) | — | 0.940 | — |
| R2 (重量) | — | 0.904 | — |

## 技术栈

- PyTorch 2.7.1 + CUDA 12.8
- Gradio (Web界面)
- torchvision (ResNet50骨干)
- numpy, PIL (图像处理)

## 数据集

- **nirscene1**: 真实近红外RGB-NIR配对图像 (2597 train + 320 test)
- **Nutrition5k**: 食物营养标注数据 (Phase2训练)
- **HSIFoodIngr-64**: 高光谱食材数据 (辅助训练)

## 项目路线

- [x] Phase1 生成器训练 (4层U-Net, 200 epochs)
- [x] Phase2 多任务训练 (100 epochs)
- [x] 端到端推理管线
- [x] Gradio Web界面
- [x] 食谱推荐系统
- [ ] Phase1 重训 (7层Deep U-Net, 进行中)
- [ ] 分类标签完善 (当前category=unknown)
- [ ] 消融实验
- [ ] 双路径改造
