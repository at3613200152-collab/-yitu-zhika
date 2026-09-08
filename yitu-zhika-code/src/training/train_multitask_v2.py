"""
阶段二训练脚本 v2: 多任务营养估计网络 (5回归头)
==================================================
训练改造的ResNet50多任务网络，同时进行食物分类和5维营养回归。

v2改动:
    - 回归目标: 2维 → 5维 [卡路里, 重量, 蛋白质, 碳水, 脂肪]
    - 损失函数: MultiTaskLossV2 (5个L1 + 5个MAPE)
    - 数据加载: nutrition5k_loader已支持protein/carb/fat字段

使用方法:
    python training/train_multitask_v2.py --config configs/default.yaml
    python training/train_multitask_v2.py --config configs/default.yaml --resume
    python training/train_multitask_v2.py --generator_ckpt checkpoints/phase1/final_model.pth
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

# 添加src目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from models.resnet_multitask import ResNetMultiTask
from models.multitask_losses import MultiTaskLossV2
from data.nutrition5k_loader import create_nutrition5k_dataloader
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="阶段二v2: 训练5维多任务营养估计网络")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", action="store_true",
                        help="从最新checkpoint续训")
    parser.add_argument("--generator_ckpt", type=str, default=None,
                        help="阶段一生成器checkpoint路径(用于生成NIR)")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备")
    parser.add_argument('--epochs', type=int, default=None)
    return parser.parse_args()


def build_multitask_input(images, generator, input_channels, mean_t=None, std_t=None):
    if input_channels == 3:
        return images
    if input_channels != 4:
        raise ValueError('Only explicit RGB (3ch) or RGB+predicted NIR (4ch) is supported')
    if generator is None:
        raise ValueError('RGB+NIR training requires a loaded generator; zero-channel substitution is prohibited')
    with torch.no_grad():
        rgb_norm = (images * std_t + mean_t) * 2 - 1
        nir = generator(rgb_norm)
        if nir.shape != (images.shape[0], 1, images.shape[2], images.shape[3]) or not torch.isfinite(nir).all():
            raise ValueError('Invalid predicted NIR tensor')
        nir_imagenet = ((nir + 1) / 2 - 0.485) / 0.229
    return torch.cat([images, nir_imagenet], dim=1)


def train_multitask_v2(config: Dict, resume: bool = False, generator_ckpt: str = None):
    """训练多任务网络主函数 (v2)"""

    mt_cfg = config.get('multitask', {})
    data_cfg = config.get('data', {})
    general_cfg = {
        'seed': config.get('seed', 42),
        'device': config.get('device', 'cuda'),
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints/multitask_v2'),
        'log_dir': config.get('log_dir', './logs/multitask_v2'),
    }

    # 设备
    device_str = general_cfg['device']
    if not torch.cuda.is_available():
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # 随机种子
    seed = general_cfg['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 数据路径
    data_path = data_cfg.get('nutrition5k_path', './data/Nutrition5k')
    if not os.path.exists(os.path.join(data_path, 'dishes_verified.csv')):
        raise ValueError('Verified official nutrition metadata is required before new training')
    input_channels = mt_cfg.get('input_channels', 4)
    if input_channels not in (3, 4):
        raise ValueError('input_channels must be 3 or 4')
    if input_channels == 4 and (not generator_ckpt or not os.path.isfile(generator_ckpt)):
        raise ValueError('RGB+NIR training requires --generator_ckpt; refusing a zero NIR ablation disguised as the full model')

    # 创建数据加载器
    img_size = data_cfg.get('img_size', 256)
    batch_size = mt_cfg.get('batch_size', 16)
    num_workers = mt_cfg.get('num_workers', 2)

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

    # 打印类别信息
    if hasattr(train_loader.dataset, 'category_to_idx'):
        cat_map = train_loader.dataset.category_to_idx
        print(f"食物类别数: {len(cat_map)}")
        if len(cat_map) <= 20:
            for cat, idx in sorted(cat_map.items(), key=lambda x: x[1]):
                print(f"  {idx}: {cat}")
        else:
            print(f"  类别过多({len(cat_map)})，只显示前10个:")
            for cat, idx in sorted(cat_map.items(), key=lambda x: x[1])[:10]:
                print(f"  {idx}: {cat}")

    # 创建多任务模型 (v2: 5回归头)
    num_classes = mt_cfg.get('num_classes', len(cat_map) if 'cat_map' in dir() else 61)
    if num_classes != len(cat_map) or len(cat_map) <= 1:
        raise ValueError('Configured class count must match a nontrivial dataset category mapping')
    print(f"初始化多任务ResNet (v2, 5回归头, {num_classes}分类)...")
    model = ResNetMultiTask(
        num_classes=num_classes,
        input_channels=mt_cfg.get('input_channels', 4),
        pretrained=mt_cfg.get('pretrained', True),
        num_regression_targets=5,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 加载阶段一生成器
    generator = None
    mean_t = None
    std_t = None

    if input_channels == 4:
        from models.generator import UNetGenerator
        print(f"加载NIR生成器(4层): {generator_ckpt}")
        generator = UNetGenerator(
            in_channels=3, out_channels=1, base_filters=64
        ).to(device)
        ckpt = torch.load(generator_ckpt, map_location=device, weights_only=False)
        gen_state = ckpt.get('G_state_dict', ckpt.get('generator_state_dict',
                              ckpt.get('model_state_dict', ckpt)))
        result = generator.load_state_dict(gen_state, strict=True)
        if result.missing_keys:
            print(f"  ⚠️ Missing keys: {len(result.missing_keys)} layers")
        if result.unexpected_keys:
            print(f"  ⚠️ Unexpected keys: {len(result.unexpected_keys)} layers")
        if not result.missing_keys and not result.unexpected_keys:
            print("  ✅ 生成器权重完全匹配")
        generator.eval()
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    else:
        print('Explicit RGB-only mode: no synthetic NIR channel is added')

    # 损失函数 v2
    criterion = MultiTaskLossV2(
        num_classes=num_classes,
        lambda_cls=mt_cfg.get('lambda_cls', 1.0),
        lambda_cal=mt_cfg.get('lambda_cal', 1.0),
        lambda_weight=mt_cfg.get('lambda_weight', 0.5),
        lambda_protein=mt_cfg.get('lambda_protein', 0.3),
        lambda_carb=mt_cfg.get('lambda_carb', 0.3),
        lambda_fat=mt_cfg.get('lambda_fat', 0.3),
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
    log_dir = os.path.join(general_cfg['log_dir'])
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 续训
    start_epoch = 0
    best_val_loss = float('inf')
    ckpt_dir = os.path.join(general_cfg['checkpoint_dir'])
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

    print(f"\n{'='*60}")
    print(f"开始训练 v2 (5维回归)")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR: {lr}")
    print(f"  回归目标: [卡路里, 重量, 蛋白质, 碳水, 脂肪]")
    print(f"  分类类别: {num_classes}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        # ======== 训练 ========
        model.train()
        train_loss_sum = 0.0
        train_cls_correct = 0
        train_total = 0
        n_batches = 0
        # 累积各回归目标的MAPE
        train_mape_sums = [0.0] * 5  # cal, weight, protein, carb, fat

        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch.get('category_idx',
                      batch.get('label', torch.zeros(images.shape[0], dtype=torch.long))).to(device)

            # 构建5维回归真值
            calories = batch['calories'].to(device).float()
            mass = batch['mass'].to(device).float()
            protein = batch['protein'].to(device).float()
            carb = batch['carb'].to(device).float()
            fat = batch['fat'].to(device).float()
            nutrition_gt = torch.stack([calories, mass, protein, carb, fat], dim=1)

            # 生成NIR通道
            input_4ch = build_multitask_input(images, generator, input_channels, mean_t, std_t)

            optimizer.zero_grad()
            outputs = model(input_4ch)

            loss, loss_dict = criterion(
                outputs['logits'], labels,
                outputs['nutrition'], nutrition_gt,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite multitask loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

            train_loss_sum += loss.item()
            train_cls_correct += (outputs['logits'].argmax(1) == labels).sum().item()
            train_total += labels.shape[0]
            n_batches += 1

            # 累积MAPE
            for i in range(5):
                pred_i = outputs['nutrition'][:, i].detach()
                gt_i = nutrition_gt[:, i]
                mape_i = torch.abs(pred_i - gt_i) / (torch.abs(gt_i) + 1.0)
                train_mape_sums[i] += mape_i.mean().item()

        avg_train_loss = train_loss_sum / max(n_batches, 1)
        train_acc = train_cls_correct / max(train_total, 1)
        train_mape_avgs = [s / max(n_batches, 1) for s in train_mape_sums]

        # ======== 验证 ========
        model.eval()
        val_loss_sum = 0.0
        val_cls_correct = 0
        val_total = 0
        val_n = 0
        val_mape_sums = [0.0] * 5

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch.get('category_idx',
                          batch.get('label', torch.zeros(images.shape[0], dtype=torch.long))).to(device)

                calories = batch['calories'].to(device).float()
                mass = batch['mass'].to(device).float()
                protein = batch['protein'].to(device).float()
                carb = batch['carb'].to(device).float()
                fat = batch['fat'].to(device).float()
                nutrition_gt = torch.stack([calories, mass, protein, carb, fat], dim=1)

                input_4ch = build_multitask_input(images, generator, input_channels, mean_t, std_t)

                outputs = model(input_4ch)
                loss, _ = criterion(outputs['logits'], labels,
                                   outputs['nutrition'], nutrition_gt)

                val_loss_sum += loss.item()
                val_cls_correct += (outputs['logits'].argmax(1) == labels).sum().item()
                val_total += labels.shape[0]
                val_n += 1

                for i in range(5):
                    pred_i = outputs['nutrition'][:, i]
                    gt_i = nutrition_gt[:, i]
                    mape_i = torch.abs(pred_i - gt_i) / (torch.abs(gt_i) + 1.0)
                    val_mape_sums[i] += mape_i.mean().item()

        avg_val_loss = val_loss_sum / max(val_n, 1)
        val_acc = val_cls_correct / max(val_total, 1)
        val_mape_avgs = [s / max(val_n, 1) for s in val_mape_sums]

        # 更新学习率
        scheduler.step()

        # 打印
        names = ["cal", "wt", "prot", "carb", "fat"]
        train_mape_str = " ".join(f"{n}={v * 100:.2f}%" for n, v in zip(names, train_mape_avgs))
        val_mape_str = " ".join(f"{n}={v * 100:.2f}%" for n, v in zip(names, val_mape_avgs))

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train_loss={avg_train_loss:.4f} acc={train_acc:.4f} [{train_mape_str}] | "
              f"Val_loss={avg_val_loss:.4f} acc={val_acc:.4f} [{val_mape_str}]")

        # TensorBoard
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        for i, name in enumerate(names):
            writer.add_scalar(f'train/mape_{name}_percent', train_mape_avgs[i] * 100, epoch)
            writer.add_scalar(f'val/mape_{name}_percent', val_mape_avgs[i] * 100, epoch)

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
                    'version': 'v2',
                    'num_regression_targets': 5,
                    'target_names': ['calories', 'mass', 'protein', 'carb', 'fat'],
                    'category_to_idx': cat_map,
                    'classification_valid': True,
                    'label_source': 'official_nutrition_values_and_derived_ingredient_categories',
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
    print(f"Checkpoint目录: {ckpt_dir}")


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.device is not None:
        config['device'] = args.device
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError('--epochs must be positive')
        config.setdefault('multitask', {})['epochs'] = args.epochs

    train_multitask_v2(config, resume=args.resume, generator_ckpt=args.generator_ckpt)


if __name__ == '__main__':
    main()
