# Generator 模块

NIR图像生成器及相关组件。

## 模块列表

| 文件 | 描述 |
|------|------|
| [nir_generator.py](nir_generator.py) | 7层Deep U-Net生成器 (RGB→NIR)，当前主力架构 |
| [generator_v1.py](generator_v1.py) | 4层U-Net生成器 (RGB→NIR)，早期版本，checkpoint兼容 |
| [discriminator.py](discriminator.py) | 70×70 PatchGAN条件判别器 |
| [unet.py](unet.py) | 团队版7层U-Net (备选，解码器更宽) |
| [pix2pix.py](pix2pix.py) | Pix2Pix模型 |
| [losses.py](losses.py) | 生成器损失函数 |

## 使用说明

- 推理时使用 `nir_generator.py` 或 `generator_v1.py`，根据checkpoint自动选择
- 训练时使用 `discriminator.py` 作为GAN判别器
- `unet.py` 为团队版本，架构与nir_generator不兼容，需独立训练
