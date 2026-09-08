"""
图像质量评估指标
=================
评估NIR图像生成质量的PSNR和SSIM指标。

PSNR (Peak Signal-to-Noise Ratio):
    衡量生成图像与真实图像之间的像素级误差
    PSNR = 10 * log10(MAX^2 / MSE)
    典型值: 20-40 dB, 越高越好

SSIM (Structural Similarity Index):
    衡量结构相似性，更符合人眼感知
    SSIM范围 [-1, 1], 通常 [0, 1], 越接近1越好
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """计算PSNR (Peak Signal-to-Noise Ratio)

    Args:
        pred: 预测图像 [B, C, H, W], 值域[0, max_val]
        target: 目标图像 [B, C, H, W], 值域[0, max_val]
        max_val: 像素最大值 (默认1.0)

    Returns:
        PSNR标量 (dB)
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(max_val ** 2 / mse)


def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """生成1D高斯核"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    """生成2D高斯核"""
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    kernel = k2d.expand(channels, 1, size, size).contiguous()
    return kernel


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    size_average: bool = True,
) -> torch.Tensor:
    """计算SSIM (Structural Similarity Index)

    基于高斯加权的局部统计量（均值、方差、协方差）计算结构相似性。

    Args:
        pred: 预测图像 [B, C, H, W]
        target: 目标图像 [B, C, H, W]
        window_size: 高斯窗口大小 (默认11)
        sigma: 高斯核标准差 (默认1.5)
        data_range: 数据范围 (默认1.0)
        size_average: 是否对所有像素取平均

    Returns:
        SSIM标量
    """
    C = pred.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, C).to(pred.device, pred.dtype)

    # 常数，防止除零
    L = data_range
    K1, K2 = 0.01, 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # 计算局部统计量
    mu1 = F.conv2d(pred, kernel, groups=C, padding=window_size // 2)
    mu2 = F.conv2d(target, kernel, groups=C, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, kernel, groups=C, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, kernel, groups=C, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, groups=C, padding=window_size // 2) - mu1_mu2

    # SSIM公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


class ImageMetrics:
    """图像质量评估指标集合

    方便批量计算PSNR和SSIM。

    Args:
        device: 计算设备
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        max_val: float = 1.0,
    ) -> Dict[str, float]:
        """计算所有图像指标

        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            max_val: 像素最大值

        Returns:
            dict: {'psnr': float, 'ssim': float}
        """
        pred = pred.to(self.device)
        target = target.to(self.device)

        psnr_val = psnr(pred, target, max_val).item()
        ssim_val = ssim(pred, target, data_range=max_val).item()

        return {
            "psnr": psnr_val,
            "ssim": ssim_val,
        }

    def compute_batch(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        max_val: float = 1.0,
    ) -> Dict[str, float]:
        """批量计算图像指标（逐样本计算后取平均）

        Args:
            preds: 批量预测 [B, C, H, W]
            targets: 批量目标 [B, C, H, W]

        Returns:
            dict: {'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std'}
        """
        B = preds.shape[0]
        psnr_list, ssim_list = [], []

        for i in range(B):
            metrics = self.compute(preds[i:i+1], targets[i:i+1], max_val)
            psnr_list.append(metrics["psnr"])
            ssim_list.append(metrics["ssim"])

        return {
            "psnr_mean": float(np.mean(psnr_list)),
            "psnr_std": float(np.std(psnr_list)),
            "ssim_mean": float(np.mean(ssim_list)),
            "ssim_std": float(np.std(ssim_list)),
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成测试数据
    target = torch.rand(4, 1, 64, 64).to(device)
    # 添加不同级别的噪声
    pred_good = target + torch.randn_like(target) * 0.05   # 高质量
    pred_poor = target + torch.randn_like(target) * 0.3     # 低质量

    metrics = ImageMetrics(device)

    result_good = metrics.compute(pred_good, target)
    result_poor = metrics.compute(pred_poor, target)

    print(f"高质量预测: PSNR={result_good['psnr']:.2f} dB, SSIM={result_good['ssim']:.4f}")
    print(f"低质量预测: PSNR={result_poor['psnr']:.2f} dB, SSIM={result_poor['ssim']:.4f}")

    # 批量测试
    batch_result = metrics.compute_batch(pred_good, target)
    print(f"批量结果: PSNR={batch_result['psnr_mean']:.2f}±{batch_result['psnr_std']:.2f}, "
          f"SSIM={batch_result['ssim_mean']:.4f}±{batch_result['ssim_std']:.4f}")
