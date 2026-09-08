"""
阶段二训练脚本: 多任务营养估计网络
====================================
训练改造的ResNet50多任务网络，同时进行食物分类和营养回归。

训练流程:
    1. 加载训练好的NIR生成器(阶段一)
    2. 创建数据加载器
    3. 初始化多任务ResNet
    4. 训练分类+回归多任务
    5. 保存checkpoint和日志

使用方法:
    python training/train_multitask.py --config configs/default.yaml
    python training/train_multitask.py --config configs/default.yaml --resume
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

from models.multitask.resnet_multitask import ResNetMultiTask
from models.multitask.losses import MultiTaskLoss
from data.nutrition5k_loader import create_nutrition5k_dataloader
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="阶段二: 训练多任务营养估计网络")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", action="store_true",
                        help="从最新checkpoint续训")
    parser.add_argument("--generator_ckpt", type=str, default=None,
                        help="阶段一生成器checkpoint路径(用于生成NIR)")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备")
    return parser.parse_args()


def train_multitask(config: Dict, resume: bool = False, generator_ckpt: str = None):
    """训练多任务网络主函数

    Args:
        config: 配置字典
        resume: 是否续训
        generator_ckpt: 阶段一生成器checkpoint路径
    """
    # 提取配置
    mt_cfg = config.get('multitask', {})
    data_cfg = config.get('data', {})
    general_cfg = {
        'seed': config.get('seed', 42),
        'device': config.get('device', 'cuda'),
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints/multitask'),
        'log_dir': config.get('log_dir', './logs/multitask'),
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

    # 数据路径(Kaggle适配)
    data_path = data_cfg.get('nutrition5k_path', './data/Nutrition5k')
    if os.path.exists('/kaggle/input'):
        kaggle_path = '/kaggle/input/nutrition5k/Nutrition5k'
        if os.path.exists(kaggle_path):
            data_path = kaggle_path
            print(f"Kaggle环境检测: 使用 {data_path}")

    # 创建数据加载器
    img_size = data_cfg.get('img_size', 256)
    batch_size = mt_cfg.get('batch_size', 32)
    num_workers = data_cfg.get('num_workers', 4)

    print("创建数据加载器...")
    train_loader = create_nutrition5k_dataloader(
        root_dir=data_path, split="train", img_size=img_size,
        batch_size=batch_size, num_workers=num_workers, augmentation=True,
    )
    val_loader = create_nutrition5k_dataloader(
        root_dir=data_path, split="val", img_size=img_size,
        batch_size=batch_size, num_workers=num_workers, augmentation=False,
    )
    print(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")

    # 创建多任务模型
    print("初始化多任务ResNet...")
    model = ResNetMultiTask(
        num_classes=mt_cfg.get('num_classes', 61),
        input_channels=mt_cfg.get('input_channels', 4),
        pretrained=mt_cfg.get('pretrained', True),
    ).to(device)

    # 加载阶段一生成器(用于从RGB生成NIR)
    generator = None
    if generator_ckpt and os.path.exists(generator_ckpt):
        from models.generator.unet import UNetGenerator
        generator = UNetGenerator(
            input_channels=3, output_channels=1
        ).to(device)
        ckpt = torch.load(generator_ckpt, map_location=device)
        gen_state = ckpt.get('model_state_dict', ckpt)
        # 只加载生成器权重
        gen_state = {k.replace('generator.', ''): v
                     for k, v in gen_state.items() if k.startswith('generator.')}
        generator.load_state_dict(gen_state, strict=False)
        generator.eval()
        print("NIR生成器已加载")
    else:
        print("未加载NIR生成器，将使用零通道NIR进行训练")

    # 损失函数
    criterion = MultiTaskLoss(
        num_classes=mt_cfg.get('num_classes', 61),
        lambda_cls=mt_cfg.get('lambda_cls', 1.0),
        lambda_cal=mt_cfg.get('lambda_cal', 1.0),
        lambda_weight=mt_cfg.get('lambda_weight', 0.5),
        lambda_mape=mt_cfg.get('lambda_mape', 0.1),
    )

    # 优化器
    lr = mt_cfg.get('lr', 0.001)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,
        betas=(mt_cfg.get('beta1', 0.9), mt_cfg.get('beta2', 0.999)),
    )

    # 学习率调度器
    epochs = mt_cfg.get('epochs', 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # TensorBoard
    log_dir = os.path.join(general_cfg['log_dir'], 'multitask')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 续训
    start_epoch = 0
    best_val_loss = float('inf')
    ckpt_dir = os.path.join(general_cfg['checkpoint_dir'], 'multitask')
    if resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            info = load_checkpoint(latest, model, optimizer, scheduler, device)
            start_epoch = info['epoch'] + 1
            best_val_loss = info['best_metric']
            print(f"从第 {start_epoch} 轮续训")

    # 训练循环
    save_interval = mt_cfg.get('save_interval', 5)
    patience = mt_cfg.get('early_stopping_patience', 20)
    patience_counter = 0

    print(f"\n开始训练，共 {epochs} 轮")

    for epoch in range(start_epoch, epochs):
        # ======== 训练 ========
        model.train()
        train_loss_sum = 0.0
        train_cls_correct = 0
        train_total = 0
        n_batches = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch.get('category_idx', batch.get('label', torch.zeros(images.shape[0], dtype=torch.long))).to(device)
            calories = batch['calories'].to(device).float()
            mass = batch['mass'].to(device).float()
            nutrition_gt = torch.stack([calories, mass], dim=1)

            # 生成NIR通道(如果生成器可用)
            if generator is not None:
                with torch.no_grad():
                    nir = generator(images)
                # 反归一化images以适配4通道输入
                # 注意: 如果images已归一化，需要适配处理
                input_4ch = torch.cat([images, nir], dim=1)
            else:
                # 无生成器时，用零通道
                nir_zero = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3]).to(device)
                input_4ch = torch.cat([images, nir_zero], dim=1)

            optimizer.zero_grad()
            outputs = model(input_4ch)

            loss, loss_dict = criterion(
                outputs['logits'], labels,
                outputs['nutrition'], nutrition_gt,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_cls_correct += (outputs['logits'].argmax(1) == labels).sum().item()
            train_total += labels.shape[0]
            n_batches += 1

        avg_train_loss = train_loss_sum / n_batches
        train_acc = train_cls_correct / train_total if train_total > 0 else 0

        # ======== 验证 ========
        model.eval()
        val_loss_sum = 0.0
        val_cls_correct = 0
        val_total = 0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch.get('category_idx', batch.get('label', torch.zeros(images.shape[0], dtype=torch.long))).to(device)
                calories = batch['calories'].to(device).float()
                mass = batch['mass'].to(device).float()
                nutrition_gt = torch.stack([calories, mass], dim=1)

                if generator is not None:
                    nir = generator(images)
                    input_4ch = torch.cat([images, nir], dim=1)
                else:
                    nir_zero = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3]).to(device)
                    input_4ch = torch.cat([images, nir_zero], dim=1)

                outputs = model(input_4ch)
                loss, _ = criterion(outputs['logits'], labels, outputs['nutrition'], nutrition_gt)

                val_loss_sum += loss.item()
                val_cls_correct += (outputs['logits'].argmax(1) == labels).sum().item()
                val_total += labels.shape[0]
                val_n += 1

        avg_val_loss = val_loss_sum / val_n if val_n > 0 else float('inf')
        val_acc = val_cls_correct / val_total if val_total > 0 else 0

        # 更新学习率
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train_loss={avg_train_loss:.4f} Train_acc={train_acc:.4f} "
              f"Val_loss={avg_val_loss:.4f} Val_acc={val_acc:.4f}")

        # TensorBoard
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)

        # 保存
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % save_interval == 0 or is_best:
            save_checkpoint(
                state={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_val_loss,
                    'config': mt_cfg,
                },
                checkpoint_dir=ckpt_dir,
                is_best=is_best,
                max_keep=5,
            )

        if patience_counter >= patience:
            print(f"早停: 连续 {patience} 轮未改善")
            break

    writer.close()
    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_multitask(config, resume=args.resume, generator_ckpt=args.generator_ckpt)
