# 一图知卡 (Yitu Zhika)

基于近红外光谱预测的实时食物热量分析平台

> 核心论文：*Multi-Spectral Food Classification and Caloric Estimation Using Predicted Images*

## 项目简介

**一图知卡** 通过手机拍摄的食物照片，利用生成模型预测其近红外(NIR)光谱图像，再结合RGB+NIR多通道输入进行多任务学习，同时实现食物分类与热量/重量回归估计，最终为用户生成个性化周食谱。

### 核心流程

```
食物RGB图像 → NIR图像生成器(U-Net/Pix2Pix) → 拼接RGB+NIR → 多任务ResNet → {类别, 卡路里, 重量}
                                                                                        ↓
                                                                   用户信息(TDEE) → 规则引擎+LLM润色 → 周食谱
```

## 项目结构

```
yitu-zhika/
├── README.md                    # 项目说明
├── requirements.txt             # Python依赖
├── configs/
│   ├── default.yaml             # 默认超参配置
│   └── kaggle.yaml              # Kaggle环境配置
├── data/
│   ├── __init__.py
│   ├── hsifoodingr_loader.py    # HSIFoodIngr-64 数据加载（HDF5格式）
│   └── nutrition5k_loader.py    # Nutrition5k 数据加载
├── models/
│   ├── __init__.py
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── unet.py              # U-Net生成器
│   │   ├── pix2pix.py           # Pix2Pix生成器+判别器
│   │   └── losses.py            # 生成器loss（GAN+L1+感知loss）
│   ├── multitask/
│   │   ├── __init__.py
│   │   ├── resnet_multitask.py  # 改造ResNet多任务网络
│   │   └── losses.py            # CrossEntropy + L1 + MAPE
│   └── baseline/
│       ├── __init__.py
│       └── calorieclip_wrapper.py  # CalorieCLIP基线封装
├── optimizations/
│   ├── __init__.py
│   ├── advanced_generator.py    # 优化1: Pix2PixHD/条件扩散替代Pix2Pix
│   ├── multi_band.py            # 优化2: 多波段NIR预测
│   ├── attention.py             # 优化3: 空间注意力聚焦食物区域
│   ├── cooking_head.py          # 优化4: 烹饪方式辅助分类头
│   ├── food_segmention.py       # 优化5: 食物分割前置
│   └── user_interaction.py      # 优化6: 用户交互（圈选/修正）
├── evaluation/
│   ├── __init__.py
│   ├── image_metrics.py         # PSNR, SSIM
│   ├── nutrition_metrics.py     # MAPE, RMSE, MAE
│   └── ablation.py              # 消融实验脚本
├── recipe/
│   ├── __init__.py
│   ├── food_database.py         # 食物营养数据库（USDA+中国食物成分表）
│   ├── rule_engine.py           # 路线A: 规则引擎（线性规划+约束满足）
│   ├── llm_polish.py            # 路线B: LLM润色
│   ├── tdee_estimator.py        # TDEE估算
│   └── weekly_planner.py        # 整合A+B，生成完整周食谱
├── training/
│   ├── train_generator.py       # 阶段一训练脚本
│   ├── train_multitask.py       # 阶段二训练脚本
│   ├── checkpoint.py            # checkpoint保存/续训逻辑
│   └── trainer.py               # 通用训练器
├── app/
│   ├── __init__.py
│   ├── gradio_demo.py           # Gradio演示界面
│   └── pipeline.py              # 推理管线
└── notebooks/
    └── kaggle_train.ipynb       # Kaggle训练notebook模板
```

## 两阶段训练

### 阶段一：NIR图像生成
- 输入：RGB图像 → 输出：预测的NIR图像
- 模型：U-Net + PatchGAN判别器 (Pix2Pix框架)
- 损失：L1重建损失 + GAN对抗损失 + 感知损失

### 阶段二：多任务营养估计
- 输入：RGB(3ch) + NIR(1ch) = 4通道拼接
- 模型：改造ResNet50多任务网络
- 输出：食物分类 + 卡路里回归 + 重量回归
- 损失：CrossEntropy + L1 + MAPE

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练阶段一（NIR生成器）
python training/train_generator.py --config configs/default.yaml

# 训练阶段二（多任务网络）
python training/train_multitask.py --config configs/default.yaml

# 启动Gradio演示
python app/gradio_demo.py
```

## 数据集

| 数据集 | 用途 | 格式 |
|--------|------|------|
| HSIFoodIngr-64 | NIR生成训练 | HDF5 (hsi, rgb, mask, meta) |
| Nutrition5k | 多任务训练 | RGB图 + CSV metadata |

## 食谱推荐

系统提供两条路线生成周食谱：
- **路线A（规则引擎）**：基于线性规划，严格遵守营养约束
- **路线B（LLM润色）**：在路线A骨架基础上，用大模型生成做法步骤和口味描述

## 许可证

MIT License
