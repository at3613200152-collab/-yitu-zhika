---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 151082346487440_0-drive/220826202915330793/yitu-zhika_CLAUDE.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 151082346487440#1788361454527
    ReservedCode2: ""
---
# 一图知卡 — 近红外光谱食物热量分析平台

## 项目概述
用 RGB 图像生成近红外(NIR)图像，再结合 RGB+NIR 双路径做多任务热量估计（卡路里+重量回归+食物分类）。
完整链路：RGB→NIR 生成器(Phase1) → 多任务营养估计网络(Phase2) → 食谱推荐 → Gradio 演示界面。

## 技术栈
- **框架**: PyTorch 2.5.1+cu121, Python 3.11 (Miniconda 虚拟环境名: yitu)
- **生成器架构**: U-Net + Pix2Pix GAN + 感知损失, 保存格式为 .pth state_dict
- **多任务网络**: 改造 ResNet50, 4通道输入(RGB 3ch + NIR 1ch), 输出分类+卡路里回归+重量回归
- **数据集**: HSIFoodIngr-64 (RGB-NIR配对, ~2GB) + Nutrition5k (营养标注, ~10GB+)
- **硬件**: 本地 RTX 5060 笔记本版 (8GB GDDR7), 正式长训练用 Kaggle T4/P100

## 当前进度 (2026-09-02)

### Phase 1 — RGB→NIR 生成器 ✅ 已完成
- 200 epoch 全量训练完成
- 21个验证样本评估: PSNR 均值19.29dB, SSIM 均值0.799, L1 均值0.1718
- 最终权重: `checkpoints/phase1/final_model.pth` (16.6M参数)
- 评估脚本: `src/evaluation/eval_generator.py`
- 可视化: `results/phase1_eval/samples_grid.png`

### Phase 2 — 多任务热量估计 ✅ 已完成
- 100 epoch 全量训练完成
- 卡路里: MAPE 16.97%, R²=0.94
- 重量: MAPE 29.92%, R²=0.90

### 当前阶段 — 扩充数据集重训 Phase 1
- 目标: 扩充 RGB-NIR 配对数据集，重训生成器提升 PSNR/SSIM
- 已新增 Pix2PixHD 格式适配加载器
- HSIFoodIngr-64 已加载 123 个有效样本

## 代码结构
```
yitu-zhika/
├── src/
│   ├── data/           # 双数据集加载器 (HSIFoodIngr-64 + Nutrition5k)
│   ├── models/         # U-Net生成器, Pix2Pix GAN, ResNet50多任务网络
│   ├── training/       # 分阶段训练脚本 (带自动续训/Checkpoint恢复)
│   ├── evaluation/     # PSNR/SSIM/L1 + MAPE/RMSE/MAE + 7组消融实验
│   ├── optimization/   # 6大优化模块 (高级生成器/多波段NIR/空间注意力等)
│   └── inference/      # 端到端推理管线 + Gradio界面
├── checkpoints/        # 模型权重
├── results/            # 评估结果和可视化
├── data/               # 数据集目录 (需手动下载)
└── docs/               # 项目文档
```

## 关键参数
- 生成器输入: RGB 图像 (3通道)
- 生成器输出: NIR 图像 (单通道)
- 多任务网络输入: RGB+NIR (4通道)
- batch_size 推荐: 4-8 (本地8GB显存约束)

## 待办
- [ ] 下载 Nutrition5k 数据集 (~10GB+)
- [ ] 扩充 RGB-NIR 配对数据集
- [ ] 重训 Phase 1 生成器
- [ ] 配置 Kaggle GPU Notebook 环境用于长耗时训练
- [ ] 推送代码到 GitHub (用户名: at3613200152-collab, 仓库名: -yitu-zhika)

## 注意事项
- 数据集尚未完整下载，HSIFoodIngr-64 已有123个有效样本在用
- Nutrition5k 数据集还未下载
- 本地显存有限，batch_size 不要超过8
- 训练脚本支持自动续训，中断后可从最近 checkpoint 恢复

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
