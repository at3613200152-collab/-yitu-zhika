# 食物卡路里估计系统 - 多光谱图像方法

## 项目简介

本项目复现论文 **Multi-Spectral Food Classification and Caloric Estimation Using Predicted Images** 的算法流程，实现基于多光谱图像的食物分类与卡路里估计系统。

### 算法流程

1. **第一阶段：NIR图像生成**
   - 在 HSIFoodIngr-64 高光谱数据集上训练 U-Net / Pix2Pix 生成器
   - 将 RGB 图像映射为近红外（NIR）光谱图像
   - 使用 PSNR 和 SSIM 评估生成质量

2. **第二阶段：多任务学习**
   - 在 Nutrition5k 数据集上，将 RGB 图像与预测的 NIR 通道拼接
   - 修改 ResNet 构建多任务网络：
     - 食物分类任务（Cross-Entropy Loss）
     - 卡路里估计任务（L1 Loss）
     - 重量估计任务（L1 Loss）

3. **消融实验对比**
   - 基线方法：纯 RGB 输入的 ResNet
   - 多光谱方法：RGB + 预测 NIR 输入
   - 对比指标：MAPE、RMSE、分类准确率

## 项目结构

```
food_calorie_estimation/
├── config.yaml                 # 配置文件
├── requirements.txt            # 依赖包
├── data/
│   ├── __init__.py
│   └── dataset.py             # 数据集加载
├── models/
│   ├── __init__.py
│   ├── nir_generator.py       # NIR生成器 (U-Net / Pix2Pix)
│   ├── multitask_net.py       # 多任务网络
│   └── baseline.py            # 基线模型
├── training/
│   ├── __init__.py
│   ├── train_nir_generator.py # NIR生成器训练脚本
│   └── train_multitask.py     # 多任务网络训练脚本
├── evaluation/
│   ├── __init__.py
│   └── metrics.py             # 评估指标 (PSNR, SSIM, MAPE, RMSE)
├── inference/
│   ├── __init__.py
│   └── estimator.py           # 推理封装类
├── app/
│   └── server.py              # Flask后端API
├── frontend/
│   ├── index.html             # 前端页面
│   ├── css/style.css          # 样式文件
│   └── js/app.js              # 前端逻辑
├── checkpoints/               # 模型权重保存目录
├── results/                   # 结果保存目录
└── logs/                      # 训练日志
```

## 环境安装

```bash
cd food_calorie_estimation
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备

将数据集放入对应目录：
- HSIFoodIngr-64: `data/HSIFoodIngr-64/RGB/` 和 `data/HSIFoodIngr-64/NIR/`
- Nutrition5k: `data/Nutrition5k/images/` 和 `data/Nutrition5k/labels.csv`

> 注意：首次运行时，如无数据会自动生成模拟数据用于测试。

### 2. 训练 NIR 生成器

```bash
python training/train_nir_generator.py --config config.yaml
```

训练完成后，最佳模型保存在 `checkpoints/nir_generator/best.pth`

### 3. 训练多任务网络

使用多光谱输入（RGB + 预测NIR）：

```bash
python training/train_multitask.py --config config.yaml \
    --nir_ckpt checkpoints/nir_generator/best.pth
```

训练基线模型（纯RGB）：

```bash
python training/train_multitask.py --config config.yaml --baseline
```

### 4. 启动演示应用

```bash
python app/server.py
```

然后在浏览器中打开 http://localhost:5000

## API 接口

### POST /api/predict
上传图像，返回完整预测结果

请求：
- `image`: 图像文件

响应：
```json
{
  "success": true,
  "data": {
    "rgb_image": "base64...",
    "nir_image": "base64...",
    "multispectral": {
      "food_class": "apple",
      "class_probability": 0.95,
      "calories": 95.5,
      "weight": 180.2
    },
    "baseline": {
      "food_class": "apple",
      "class_probability": 0.88,
      "calories": 105.3,
      "weight": 190.1
    }
  }
}
```

### POST /api/predict-nir
仅预测 NIR 图像

### POST /api/predict-calories
预测卡路里和分类

## 评估指标

| 指标 | 说明 | 用途 |
|------|------|------|
| PSNR | 峰值信噪比 | NIR图像生成质量 |
| SSIM | 结构相似性 | NIR图像生成质量 |
| MAPE | 平均绝对百分比误差 | 卡路里/重量估计 |
| RMSE | 均方根误差 | 卡路里/重量估计 |
| Accuracy | 分类准确率 | 食物分类任务 |

## 课程设计提交物

1. **源代码仓库**: 本项目所有代码
2. **训练好的模型权重**: `checkpoints/` 目录
3. **课程设计报告**: 包含算法复现与对比实验分析

## 论文参考

> Multi-Spectral Food Classification and Caloric Estimation Using Predicted Images
