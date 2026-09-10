# 模型定义模块

阶段一 RGB→NIR 生成器与判别器 + 阶段二多任务 ResNet50 与五目标宏量网络。

> **当前状态（2026-09-10）**：阶段一当前使用的是 [`generator.py`](generator.py) 的 **4 层 `UNetGenerator`**
> （对齐论文的 U-Net 路线，`train_hsi_full_v3.py` / `train_hsi_official.py` 均用它）。
> [`nir_generator.py`](nir_generator.py) 是队友那套 **7 层 Deep U-Net**（`base_channels=64`），
> 只在 `src/training/train_generator.py` 与 `src/evaluation/eval_generator.py` 中使用，作为路径 B 的对照实现。
> 早期 README 写的「4 层 = 旧版 PSNR 19.29、7 层 = 当前使用」已作废。

## 模块清单

| 文件 | 描述 | 用途 |
|------|------|------|
| [generator.py](generator.py) | **4 层 U-Net 生成器**（`UNetGenerator(in_channels=3, out_channels=1, base_filters=64)`） | 阶段一：RGB→NIR，**当前使用**（全量三种子 26.674 dB / SSIM 0.9084） |
| [nir_generator.py](nir_generator.py) | 7 层 Deep U-Net（46.9M 参数） | 阶段一：路径 B 对照实现（队友架构） |
| [discriminator.py](discriminator.py) | PatchGAN 判别器（70×70 感受野） | 阶段一：仅在路径 B（对抗训练）中使用 |
| [resnet_multitask.py](resnet_multitask.py) | 4 通道 ResNet50 多任务网络 | 阶段二：热量 + 重量 + 分类 |
| [meal_macros_net.py](meal_macros_net.py) | 五目标宏量网络（`MealMacrosNet`，可选非负参数化） | 阶段二扩展：热量/重量/蛋白/碳水/脂肪 |
| [meal_ablation.py](meal_ablation.py) | 消融用网络包装 | 消融 B/C（非负输出、类别加权） |
| [multitask_losses.py](multitask_losses.py) | 多任务损失（掩码归一化 L1 + 0.2×CE） | 阶段二 |
| [checkpoint_io.py](checkpoint_io.py) | 检查点读写（`best.pt` 可 `weights_only` 加载） | 通用 |
| [clip_food_classifier.py](clip_food_classifier.py) | CalorieCLIP 基线探针 | 开源基线对照（仅热量） |

## 架构说明

### 阶段一（当前）：4 层 U-Net + L1

- **生成器**（`generator.py`）：`UNetGenerator`，4 层下采样、`base_filters=64`、输入 3 通道 → 输出 1 通道
  - 输入 RGB `[3,H,W]`，输出 NIR `[1,H,W]`，值域 `[-1,1]`
  - 损失：**L1**（在 `[-1,1]` 域计算），不使用对抗损失
- **训练协议**：目标为 860 nm 单波段；缩放只用训练集统计的固定值（`low=0.17514 / high=1.67553`）；
  原生 HSI 顺时针 90° 对齐 RGB；Adam `lr=2e-4, β=(0.5,0.999)`；`ReduceLROnPlateau(0.5, patience=5)`；
  选模 = 验证集逐图 L1；bf16 前向 + fp32 损失；梯度裁剪 1.0
- **锚点消融（单因素，seed 42）**：把目标旋转 90° → **−7.73 dB**（26.654 → 18.928，SSIM 0.9093 → 0.7160）；
  逐图归一化替代固定缩放 → **−5.59 dB**（→ 21.062，SSIM 0.8373）

### 阶段一（路径 B 对照）：7 层 Deep U-Net + PatchGAN

- **生成器**（`nir_generator.py`）：7 层编码器-解码器 + skip connection（256→…→2 → 1→…→256），约 46.9M 参数
- **判别器**（`discriminator.py`）：PatchGAN 70×70，输入 RGB+NIR 拼接 `[4,H,W]` → 真伪概率图（约 2.8M 参数）
- 损失：L1（λ=100）+ 对抗损失
- ⚠️ 这条路径**不是当前报告数字的来源**，且队友那份数据因 NIR 与 RGB 差 90° 未对齐，只有 14.25 dB / 0.622

### 阶段二：4 通道多任务 ResNet50

- **输入**：RGB `[3,H,W]` 与预测 NIR `[1,H,W]` 拼接为 4 通道
- **第 4 通道初始化**：前 3 通道沿用 ImageNet 预训练权重，第 4 通道用 RGB 通道均值初始化并缩放
  （保证与纯 RGB 模型**只差一个通道**，这是 NIR 对照实验成立的前提）
- **骨干**：ResNet50（改写 `first_conv` 接收 4 通道）
- **输出头**：`calorie`（1 维回归）、`weight`（1 维回归）、`category`（分类）
- `forward` 返回 dict：`{'calorie': tensor, 'weight': tensor, 'category': tensor}`
- **损失**：带 mask 的逐目标归一化 L1 + `0.2 × CE`（分类作辅助正则）；目标归一化统计量只用训练集

### 阶段二扩展：五目标宏量网络

- `MealMacrosNet(manifest, pretrained, input_channels, nonneg=False)`，在 4 通道 ResNet50 之上加五个回归头：
  calories(kcal)、mass(g)、protein/carbohydrate/fat(g)
- `--nonneg-output` 打开 softplus 参数化（消融 B：负值清零但热量 MAE +4.5 kcal → 不上线）
- 在线回归/宏量模型为 `v1_expanded`（冻结 507 测试集：热量 MAE 58.50 kcal、重量 36.14 g、粗分类 73.77%）
