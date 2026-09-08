# 模型定义模块

Phase1 生成对抗网络 + Phase2 多任务 ResNet50。

## 模块清单

| 文件 | 描述 | 用途 |
|------|------|------|
| [nir_generator.py](nir_generator.py) | **7层 Deep U-Net 生成器** | Phase1: RGB→NIR 图像生成（当前使用） |
| [generator.py](generator.py) | 4层 U-Net 生成器（备份） | Phase1: 旧版生成器，PSNR 19.29 |
| [discriminator.py](discriminator.py) | PatchGAN 判别器 | Phase1: 对抗训练判别真伪 NIR |
| [resnet_multitask.py](resnet_multitask.py) | 4通道 ResNet50 多任务网络 | Phase2: 卡路里+重量+分类预测 |
| [multitask_losses.py](multitask_losses.py) | 多任务损失函数 | Phase2: L1+CrossEntropy加权组合 |

## 架构说明

### Phase1: Pix2Pix（7层 Deep U-Net）

- **生成器**（nir_generator.py）：7层编码器-解码器，跳跃连接
  - 编码: 256→128→64→32→16→8→4→2（7层下采样）
  - 解码: 1→2→4→8→16→32→64→128→256（7层上采样 + skip connection）
  - 参数量: 46,872,321（~47M）
  - 输入: RGB[3,256,256] → 输出: NIR[1,256,256]，值域 [-1,1]
  - 支持加载预训练权重（队友PSNR 25.8模型）
- **旧生成器**（generator.py）：4层U-Net，参数量更少，PSNR 19.29（已废弃）
- **判别器**：PatchGAN 70×70 感受野，输入 RGB+NIR 拼接 [4,256,256] → 真伪概率图
  - 参数量: 2,766,720（~2.8M）
- 损失：L1（λ=100）+ 对抗损失（GAN loss）

### Phase2: 多任务 ResNet50

- **输入**：RGB[3,H,W] + NIR[1,H,W] 拼接为 4 通道
- **骨干**：ResNet50（修改 first_conv 接收 4ch）
- **输出头**：
  - `calorie`：回归（1维），卡路里预测
  - `weight`：回归（1维），重量预测
  - `category`：分类，食物类别
- forward 返回 dict: `{'calorie': tensor, 'weight': tensor, 'category': tensor}`
