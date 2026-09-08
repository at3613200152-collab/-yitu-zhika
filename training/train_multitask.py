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
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="checkpoint根目录(覆盖配置文件; 对照实验用它避免覆盖主模型)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="TensorBoard日志根目录(覆盖配置文件)")
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

    # ImageNet反归一化常量: loader输出的是ImageNet归一化图像,
    # 而阶段一生成器训练时输入为[-1,1]域的RGB, 喂之前必须反归一化再缩放
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def make_4ch(images_tensor: torch.Tensor) -> torch.Tensor:
        """RGB(ImageNet归一化) + 生成器预测NIR → 4通道输入; 无生成器时NIR置零。"""
        if generator is not None:
            with torch.no_grad():
                rgb01 = images_tensor * imagenet_std + imagenet_mean   # → [0,1]
                nir = generator(rgb01 * 2.0 - 1.0)                    # → [-1,1]
            return torch.cat([images_tensor, nir], dim=1)
        nir_zero = torch.zeros(
            images_tensor.shape[0], 1, images_tensor.shape[2], images_tensor.shape[3],
            device=images_tensor.device,
        )
        return torch.cat([images_tensor, nir_zero], dim=1)

    # 损失函数
    # log_transform=True: 回归目标为log1p空间(loader返回nutrition_log),
    # L1在log空间计算, MAPE内部expm1还原到物理空间
    criterion = MultiTaskLoss(
        num_classes=mt_cfg.get('num_classes', 61),
        lambda_cls=mt_cfg.get('lambda_cls', 1.0),
        lambda_cal=mt_cfg.get('lambda_cal', 1.0),
        lambda_weight=mt_cfg.get('lambda_weight', 0.5),
        lambda_mape=mt_cfg.get('lambda_mape', 0.1),
        log_transform=True,
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

    # TensorBoard (每次训练使用独立时间戳目录, 避免多次运行的曲线混在一起)
    from datetime import datetime
    run_name = datetime.now().strftime('multitask_%Y%m%d_%H%M%S')
    log_dir = os.path.join(general_cfg['log_dir'], run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志目录: {log_dir}")

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
            # 回归目标: log1p空间(calories与mass跨2个数量级, 直接回归会淹没分类损失)
            nutrition_gt = batch['nutrition_log'].to(device)

            # 生成NIR通道并拼接4通道输入(内部完成反归一化适配)
            input_4ch = make_4ch(images)

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
        val_cal_abs_err = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch.get('category_idx', batch.get('label', torch.zeros(images.shape[0], dtype=torch.long))).to(device)
                nutrition_gt = batch['nutrition_log'].to(device)

                input_4ch = make_4ch(images)
                outputs = model(input_4ch)
                loss, _ = criterion(outputs['logits'], labels, outputs['nutrition'], nutrition_gt)

                # 原物理尺度kcal误差: expm1还原log1p预测(训练监控核心指标)
                cal_pred_raw = torch.expm1(outputs['nutrition'][:, 0]).clamp(min=0.0)
                cal_gt_raw = batch['calories'].to(device).float()
                val_cal_abs_err += (cal_pred_raw - cal_gt_raw).abs().sum().item()

                val_loss_sum += loss.item()
                val_cls_correct += (outputs['logits'].argmax(1) == labels).sum().item()
                val_total += labels.shape[0]
                val_n += 1

        avg_val_loss = val_loss_sum / val_n if val_n > 0 else float('inf')
        val_acc = val_cls_correct / val_total if val_total > 0 else 0
        val_cal_mae = val_cal_abs_err / val_total if val_total > 0 else 0.0

        # 更新学习率
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train_loss={avg_train_loss:.4f} Train_acc={train_acc:.4f} "
              f"Val_loss={avg_val_loss:.4f} Val_acc={val_acc:.4f} Val_kcalMAE={val_cal_mae:.1f}")

        # TensorBoard
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('val/kcal_mae', val_cal_mae, epoch)

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

    # CLI覆盖(对照组实验: 用独立目录存放, 避免覆盖主模型)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.log_dir:
        config['log_dir'] = args.log_dir

    train_multitask(config, resume=args.resume, generator_ckpt=args.generator_ckpt)
