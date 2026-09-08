"""
Deep U-Net Generator: RGB (3ch) → NIR (1ch) — 7层下采样
=========================================================
来自队友训练的模型，PSNR 24dB，支持加载预训练权重。

架构: 7层编码器 + 瓶颈 + 7层解码器 (比4层版本更深)
输入: [B, 3, 256, 256] RGB, 值域 [-1, 1]
输出: [B, 1, 256, 256] NIR, 值域 [-1, 1]
"""

import torch
import torch.nn as nn


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.5))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x, skip_input):
        x = self.block(x)
        return torch.cat([x, skip_input], dim=1)


class UNetGenerator(nn.Module):
    """7层 U-Net 生成器 (来自队友项目，PSNR 24)
    
    编码器: 256→128→64→32→16→8→4→2
    解码器: 1→2→4→8→16→32→64→128→256
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, num_downs=7):
        super().__init__()
        f = base_channels
        
        # 编码器 (7层下采样)
        self.down1 = UNetDownBlock(in_channels, f, use_bn=False)       # 256→128, ch=64
        self.down2 = UNetDownBlock(f, f * 2)                           # 128→64,  ch=128
        self.down3 = UNetDownBlock(f * 2, f * 4)                       # 64→32,   ch=256
        self.down4 = UNetDownBlock(f * 4, f * 8)                       # 32→16,   ch=512
        self.down5 = UNetDownBlock(f * 8, f * 8)                       # 16→8,    ch=512
        self.down6 = UNetDownBlock(f * 8, f * 8)                       # 8→4,     ch=512
        self.down7 = UNetDownBlock(f * 8, f * 8)                       # 4→2,     ch=512

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 解码器 (7层上采样 + skip connection)
        self.up7 = UNetUpBlock(f * 8, f * 8, use_dropout=True)     # 1→2, out=512, +d7(512)=1024
        self.up6 = UNetUpBlock(f * 16, f * 8, use_dropout=True)    # 2→4, out=512, +d6(512)=1024
        self.up5 = UNetUpBlock(f * 16, f * 8, use_dropout=True)    # 4→8, out=512, +d5(512)=1024
        self.up4 = UNetUpBlock(f * 16, f * 4)                      # 8→16, out=256, +d4(512)=768
        self.up3 = UNetUpBlock(f * 12, f * 2)                      # 16→32, out=128, +d3(256)=384
        self.up2 = UNetUpBlock(f * 6, f)                          # 32→64, out=64, +d2(128)=192
        self.up1 = UNetUpBlock(f * 3, f)                          # 64→128, out=64, +d1(64)=128
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bottleneck = self.bottleneck(d7)

        u7 = self.up7(bottleneck, d7)
        u6 = self.up6(u7, d6)
        u5 = self.up5(u6, d5)
        u4 = self.up4(u5, d4)
        u3 = self.up3(u4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)
        out = self.up0(u1)

        return out
