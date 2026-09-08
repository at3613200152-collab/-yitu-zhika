"""
阶段一训练脚本: NIR图像生成器
===============================
训练Pix2Pix模型，从RGB图像生成NIR图像。

训练流程:
    1. 加载配置
    2. 创建数据加载器
    3. 初始化Pix2Pix模型(生成器+判别器)
    4. 交替训练生成器和判别器
    5. 保存checkpoint和日志

使用方法:
    python training/train_generator.py --config configs/default.yaml
    python training/train_generator.py --config configs/default.yaml --resume  # 续训
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.generator.pix2pix import Pix2PixModel
from models.generator.losses import GeneratorLoss
from data.hsifoodingr_loader import create_hsifoodingr_dataloader
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="阶段一: 训练NIR图像生成器")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", action="store_true",
                        help="从最新checkpoint续训")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 (覆盖配置文件)")
    return parser.parse_args()


def train_generator(config: Dict, resume: bool = False):
    """训练NIR生成器主函数

    Args:
        config: 配置字典
        resume: 是否续训
    """
    # 提取配置
    gen_cfg = config.get('generator', {})
    data_cfg = config.get('data', {})
    general_cfg = {
        'seed': config.get('seed', 42),
        'device': config.get('device', 'cuda'),
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints/generator'),
        'log_dir': config.get('log_dir', './logs/generator'),
        'output_dir': config.get('output_dir', './output'),
    }

    # 设备
    device_str = general_cfg['device']
    if not torch.cuda.is_available():
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"设备: {device}")

    # 随机种子
    seed = general_cfg['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 数据路径自动检测(Kaggle适配)
    data_path = data_cfg.get('hsifoodingr_path', './data/HSIFoodIngr-64')
    if os.path.exists('/kaggle/input'):
        kaggle_path = '/kaggle/input/hsifoodingr64/HSIFoodIngr-64'
        if os.path.exists(kaggle_path):
            data_path = kaggle_path
            print(f"Kaggle环境检测: 使用 {data_path}")

    # 创建数据加载器
    img_size = data_cfg.get('img_size', 256)
    batch_size = gen_cfg.get('batch_size', 16)
    num_workers = data_cfg.get('num_workers', 4)

    print("创建数据加载器...")
    train_loader = create_hsifoodingr_dataloader(
        root_dir=data_path, split="train", img_size=img_size,
        batch_size=batch_size, num_workers=num_workers, augmentation=True,
    )
    val_loader = create_hsifoodingr_dataloader(
        root_dir=data_path, split="val", img_size=img_size,
        batch_size=batch_size, num_workers=num_workers, augmentation=False,
    )
    print(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")

    # 创建模型
    print("初始化Pix2Pix模型...")
    model = Pix2PixModel(
        input_channels=gen_cfg.get('input_channels', 3),
        output_channels=gen_cfg.get('output_channels', 1),
        base_features_g=gen_cfg.get('base_features', 64),
        lambda_l1=gen_cfg.get('lambda_l1', 100.0),
        lambda_gan=gen_cfg.get('lambda_gan', 1.0),
    ).to(device)

    # 损失函数
    criterion = GeneratorLoss(
        lambda_gan=gen_cfg.get('lambda_gan', 1.0),
        lambda_l1=gen_cfg.get('lambda_l1', 100.0),
        lambda_perceptual=gen_cfg.get('lambda_perceptual', 10.0),
        use_perceptual=gen_cfg.get('use_perceptual', True),
    ).to(device)

    # 优化器
    lr = gen_cfg.get('lr', 0.0002)
    beta1 = gen_cfg.get('beta1', 0.5)
    beta2 = gen_cfg.get('beta2', 0.999)

    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # 学习率调度器
    epochs = gen_cfg.get('epochs', 200)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=epochs)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=epochs)

    # TensorBoard
    log_dir = os.path.join(general_cfg['log_dir'], 'generator')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 续训
    start_epoch = 0
    best_g_loss = float('inf')
    ckpt_dir = os.path.join(general_cfg['checkpoint_dir'], 'generator')
    if resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer_g.load_state_dict(ckpt['optimizer_g_state_dict'])
            optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_g_loss = ckpt.get('best_g_loss', float('inf'))
            print(f"从第 {start_epoch} 轮续训")

    # 训练循环
    save_interval = gen_cfg.get('save_interval', 10)
    print(f"\n开始训练，共 {epochs} 轮")

    for epoch in range(start_epoch, epochs):
        model.train()
        d_losses, g_losses = [], []

        for batch_idx, batch in enumerate(train_loader):
            rgb = batch['rgb'].to(device)
            nir_real = batch['nir'].to(device)

            # ======== 训练判别器 ========
            optimizer_d.zero_grad()
            d_loss, d_dict = model.compute_discriminator_loss(rgb, nir_real)
            d_loss.backward()
            optimizer_d.step()

            # ======== 训练生成器 ========
            optimizer_g.zero_grad()
            nir_fake = model.generator(rgb)
            disc_fake = model.discriminator(rgb, nir_fake)
            g_loss, g_dict = criterion(nir_fake, nir_real, disc_fake)
            g_loss.backward()
            optimizer_g.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            # 每100步打印
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{len(train_loader)}] "
                      f"D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f}")

        # Epoch统计
        avg_d = sum(d_losses) / len(d_losses)
        avg_g = sum(g_losses) / len(g_losses)

        # 更新学习率
        scheduler_g.step()
        scheduler_d.step()

        # 验证
        model.eval()
        val_psnr_list = []
        with torch.no_grad():
            for val_batch in val_loader:
                rgb_v = val_batch['rgb'].to(device)
                nir_v = val_batch['nir'].to(device)
                nir_pred = model.generator(rgb_v)
                # 计算PSNR
                mse = torch.mean((nir_pred - nir_v) ** 2)
                if mse > 0:
                    val_psnr_list.append(10 * torch.log10(1.0 / mse).item())

        avg_val_psnr = sum(val_psnr_list) / len(val_psnr_list) if val_psnr_list else 0

        print(f"Epoch [{epoch+1}/{epochs}] D_loss={avg_d:.4f} G_loss={avg_g:.4f} "
              f"Val_PSNR={avg_val_psnr:.2f}dB")

        # TensorBoard
        writer.add_scalar('train/d_loss', avg_d, epoch)
        writer.add_scalar('train/g_loss', avg_g, epoch)
        writer.add_scalar('val/psnr', avg_val_psnr, epoch)

        # 保存checkpoint
        is_best = avg_g < best_g_loss
        if is_best:
            best_g_loss = avg_g

        if (epoch + 1) % save_interval == 0 or is_best:
            save_checkpoint(
                state={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'best_g_loss': best_g_loss,
                    'config': gen_cfg,
                },
                checkpoint_dir=ckpt_dir,
                is_best=is_best,
                max_keep=5,
            )

    writer.close()
    print(f"\n训练完成! 最佳G_loss: {best_g_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_generator(config, resume=args.resume)
