"""
优化1: 高级生成器 — Pix2PixHD / 条件扩散模型替代Pix2Pix
==========================================================

思路:
    基线Pix2Pix使用U-Net+PatchGAN，生成的NIR图分辨率和细节有限。
    本模块提供两种进阶方案:

    方案A - Pix2PixHD:
        使用多尺度生成器和多尺度判别器，支持高分辨率(512x512+)生成。
        全局生成器处理整体结构，局部增强网络补充细节纹理。

    方案B - 条件扩散模型(Conditional Diffusion):
        以RGB图像为条件，从噪声逐步去噪生成NIR图像。
        生成质量更高，多样性更好，但推理速度较慢。

    可通过配置参数选择方案。

参考:
    - Wang et al., "High-Resolution Image Synthesis and Semantic Manipulation 
      with Conditional GANs", CVPR 2018 (Pix2PixHD)
    - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


# ============================================================
# 方案A: Pix2PixHD 生成器
# ============================================================

class GlobalGenerator(nn.Module):
    """Pix2PixHD全局生成器

    处理整体图像结构，输出粗粒度的NIR图像。

    Args:
        input_channels: 输入通道数
        output_channels: 输出通道数
        base_features: 基础特征数 (默认64)
        n_downsample: 下采样层数 (默认5)
        n_res_blocks: 残差块数量 (默认9)
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        base_features: int = 64,
        n_downsample: int = 5,
        n_res_blocks: int = 9,
    ):
        super().__init__()
        f = base_features

        # 编码器: 逐步下采样
        encoder_layers = []
        # 首层
        encoder_layers.append(nn.Sequential(
            nn.Conv2d(input_channels, f, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(f),
            nn.ReLU(inplace=True),
        ))
        # 下采样
        for i in range(n_downsample):
            in_ch = f * (2 ** i)
            out_ch = f * (2 ** (i + 1))
            encoder_layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.encoder = nn.Sequential(*encoder_layers)

        # 残差块: 保持分辨率不变
        res_layers = []
        ch = f * (2 ** n_downsample)
        for _ in range(n_res_blocks):
            res_layers.append(ResBlock(ch))
        self.residual = nn.Sequential(*res_layers)

        # 解码器: 逐步上采样
        decoder_layers = []
        for i in range(n_downsample):
            in_ch = f * (2 ** (n_downsample - i))
            out_ch = f * (2 ** (n_downsample - i - 1))
            decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        # 最后一层
        decoder_layers.append(nn.Sequential(
            nn.Conv2d(f, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        ))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x


class ResBlock(nn.Module):
    """残差块 (InstanceNorm版本，Pix2PixHD风格)"""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LocalEnhancer(nn.Module):
    """Pix2PixHD局部增强网络

    在全局生成器的基础上补充高分辨率细节。
    输入高分辨率图像和全局生成器的输出，融合生成精细NIR图。

    Args:
        input_channels: 输入通道数
        output_channels: 输出通道数
        base_features: 基础特征数
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        base_features: int = 32,
    ):
        super().__init__()
        f = base_features

        # 下采样分支: 提取低分辨率特征
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

        # 处理低分辨率输入
        self.encoder_low = nn.Sequential(
            nn.Conv2d(input_channels, f * 2, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(f * 2),
            nn.ReLU(inplace=True),
        )

        # 处理高分辨率输入
        self.encoder_high = nn.Sequential(
            nn.Conv2d(input_channels, f, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(f),
            nn.ReLU(inplace=True),
        )

        # 融合并输出
        self.decoder = nn.Sequential(
            nn.Conv2d(f * 3, f * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(f * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 2, f, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(
        self,
        x_high_res: torch.Tensor,
        global_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_high_res: 高分辨率RGB输入
            global_output: 全局生成器的低分辨率NIR输出
        """
        # 低分辨率分支
        x_low = self.downsample(x_high_res)
        feat_low = self.encoder_low(x_low)

        # 高分辨率分支
        feat_high = self.encoder_high(x_high_res)

        # 上采样低分辨率特征并拼接
        feat_low_up = nn.functional.interpolate(
            feat_low, size=feat_high.shape[2:], mode='bilinear', align_corners=False
        )
        # 上采样全局输出特征
        global_up = nn.functional.interpolate(
            global_output, size=feat_high.shape[2:], mode='bilinear', align_corners=False
        )

        # 融合
        fused = torch.cat([feat_high, feat_low_up, global_up], dim=1)
        output = self.decoder(fused)

        return output


# ============================================================
# 方案B: 条件扩散模型 (骨架)
# ============================================================

class ConditionalDiffusionGenerator(nn.Module):
    """条件扩散模型生成器骨架

    以RGB图像为条件，从高斯噪声逐步去噪生成NIR图像。

    实现思路:
    1. 训练阶段: 对真实NIR图加噪，模型学习预测噪声
    2. 推理阶段: 从纯噪声开始，迭代去噪T步得到NIR图
    3. 条件注入: RGB图像通过编码器提取特征，注入到U-Net各层

    注意: 完整实现需要较复杂的噪声调度器和采样逻辑，
    此处提供架构骨架，具体训练/推理循环需补充。

    Args:
        input_channels: 条件输入通道数 (RGB=3)
        output_channels: 输出通道数 (NIR=1)
        base_features: U-Net基础特征数
        timesteps: 扩散步数 (默认1000)
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        base_features: int = 128,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.input_channels = input_channels
        self.output_channels = output_channels

        # 条件编码器: 提取RGB特征
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_features, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_features, base_features, 3, 1, 1),
            nn.SiLU(),
        )

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_features * 4),
            nn.SiLU(),
            nn.Linear(base_features * 4, base_features * 4),
        )

        # 去噪U-Net核心 (简化版，实际应使用带时间步注入的完整U-Net)
        f = base_features
        self.denoiser = nn.ModuleDict({
            'down1': nn.Sequential(
                nn.Conv2d(output_channels + f, f, 3, 2, 1), nn.SiLU(),
            ),
            'down2': nn.Sequential(
                nn.Conv2d(f, f * 2, 3, 2, 1), nn.SiLU(),
            ),
            'mid': nn.Sequential(
                ResBlock(f * 2),
                ResBlock(f * 2),
            ),
            'up2': nn.Sequential(
                nn.ConvTranspose2d(f * 4, f, 3, 2, 1, 1), nn.SiLU(),
            ),
            'up1': nn.Sequential(
                nn.ConvTranspose2d(f * 2 + f, f, 3, 2, 1, 1), nn.SiLU(),
            ),
            'out': nn.Sequential(
                nn.Conv2d(f + f, output_channels, 3, 1, 1),
            ),
        })

        # 噪声调度参数 (线性调度)
        self.register_buffer('beta_start', torch.tensor(1e-4))
        self.register_buffer('beta_end', torch.tensor(0.02))
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1.0 - self.betas, dim=0))

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向扩散: 对x_0在第t步加噪

        Args:
            x_0: 原始图像 [B, C, H, W]
            t: 时间步 [B] (0~T-1)
            noise: 可选噪声，默认随机生成

        Returns:
            加噪后的图像 x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 取对应时间步的alpha_cumprod
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """预测噪声 (训练时使用)

        Args:
            x_t: 加噪后的NIR图像 [B, 1, H, W]
            t: 时间步 [B]
            condition: RGB条件图像 [B, 3, H, W]

        Returns:
            预测的噪声 [B, 1, H, W]
        """
        # 条件特征
        cond_feat = self.condition_encoder(condition)

        # 时间步嵌入
        t_emb = self.time_mlp(t.float().unsqueeze(-1))  # [B, f*4]

        # 去噪U-Net
        d1 = self.denoiser['down1'](torch.cat([x_t, cond_feat], dim=1))
        d2 = self.denoiser['down2'](d1)
        mid = self.denoiser['mid'](d2)
        u2 = self.denoiser['up2'](torch.cat([mid, d2], dim=1))
        u1 = self.denoiser['up1'](torch.cat([u2, d1], dim=1))
        pred_noise = self.denoiser['out'](torch.cat([u1, cond_feat], dim=1))

        return pred_noise

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        img_size: int = 256,
    ) -> torch.Tensor:
        """推理: 从噪声生成NIR图像 (DDPM采样)

        Args:
            condition: RGB条件图像 [B, 3, H, W]
            img_size: 生成图像尺寸

        Returns:
            生成的NIR图像 [B, 1, img_size, img_size]
        """
        B = condition.shape[0]
        device = condition.device

        # 从纯噪声开始
        x = torch.randn(B, self.output_channels, img_size, img_size, device=device)

        # 迭代去噪
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            pred_noise = self.forward(x, t, condition)

            # DDPM去噪步骤
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1, 1)

            # 计算x_{t-1}
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            # 只在t>0时加噪声
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(beta_t) * noise
            else:
                x = x0_pred

        return x


# ============================================================
# 统一接口
# ============================================================

def create_advanced_generator(
    method: str = "pix2pixhd",
    input_channels: int = 3,
    output_channels: int = 1,
    base_features: int = 64,
) -> nn.Module:
    """创建高级生成器

    Args:
        method: 生成方法 ("pix2pixhd" / "diffusion")
        input_channels: 输入通道数
        output_channels: 输出通道数
        base_features: 基础特征数

    Returns:
        生成器模型
    """
    if method == "pix2pixhd":
        return GlobalGenerator(
            input_channels=input_channels,
            output_channels=output_channels,
            base_features=base_features,
        )
    elif method == "diffusion":
        return ConditionalDiffusionGenerator(
            input_channels=input_channels,
            output_channels=output_channels,
            base_features=base_features,
        )
    else:
        raise ValueError(f"不支持的生成方法: {method}，请选择 'pix2pixhd' 或 'diffusion'")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 测试Pix2PixHD全局生成器
    print("\n=== Pix2PixHD全局生成器 ===")
    gen = GlobalGenerator(input_channels=3, output_channels=1).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        y = gen(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in gen.parameters()):,}")

    # 测试条件扩散模型（小规模）
    print("\n=== 条件扩散模型 ===")
    diff = ConditionalDiffusionGenerator(
        input_channels=3, output_channels=1, base_features=64, timesteps=100
    ).to(device)
    x_t = torch.randn(2, 1, 64, 64).to(device)
    t = torch.randint(0, 100, (2,)).to(device)
    cond = torch.randn(2, 3, 64, 64).to(device)
    with torch.no_grad():
        pred = diff(x_t, t, cond)
    print(f"噪声预测输出: {pred.shape}")
