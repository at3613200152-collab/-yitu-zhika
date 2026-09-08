"""
通用训练器
===========
封装训练循环、验证、日志记录等通用逻辑。

功能:
    - 训练循环: 前向传播 → 计算损失 → 反向传播 → 参数更新
    - 验证循环: 评估模型性能
    - 学习率调度
    - TensorBoard日志
    - 自动checkpoint保存与续训
    - 早停(Early Stopping)支持
"""

import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass

from .checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 100
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    scheduler: str = "cosine"       # linear / cosine / step / none
    warmup_epochs: int = 5
    save_interval: int = 10
    early_stopping_patience: int = 20
    gradient_clip_val: float = 1.0   # 梯度裁剪值 (0=不裁剪)
    seed: int = 42
    device: str = "auto"


class Trainer:
    """通用训练器

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        config: 训练配置
        checkpoint_dir: checkpoint保存目录
        log_dir: TensorBoard日志目录
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: Callable,
        config: TrainConfig,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        # 设备
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        # 学习率调度器
        self.lr_scheduler = self._create_scheduler(config)

        # TensorBoard
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # 训练状态
        self.current_epoch = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0
        self.global_step = 0

    def _create_scheduler(self, config: TrainConfig):
        """创建学习率调度器"""
        if config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
            )
        elif config.scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=config.epochs
            )
        elif config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            return None

    def train(self, resume: bool = True) -> Dict[str, Any]:
        """执行训练

        Args:
            resume: 是否从最新checkpoint续训

        Returns:
            训练结果摘要
        """
        # 续训
        start_epoch = 0
        if resume:
            latest_ckpt = find_latest_checkpoint(self.checkpoint_dir)
            if latest_ckpt:
                info = load_checkpoint(
                    latest_ckpt, self.model, self.optimizer, self.lr_scheduler, self.device
                )
                start_epoch = info['epoch'] + 1
                self.best_val_metric = info['best_metric']
                print(f"从第 {start_epoch} 轮续训")

        # 训练循环
        print(f"开始训练，共 {self.config.epochs} 轮，设备: {self.device}")
        train_start_time = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # 训练一个epoch
            train_metrics = self._train_epoch()

            # 验证
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()

            # 更新学习率
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            # 打印日志
            log_str = f"Epoch [{epoch+1}/{self.config.epochs}] lr={current_lr:.6f} time={epoch_time:.1f}s"
            for k, v in train_metrics.items():
                log_str += f" train_{k}={v:.4f}"
            for k, v in val_metrics.items():
                log_str += f" val_{k}={v:.4f}"
            print(log_str)

            # TensorBoard记录
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            self.writer.add_scalar("lr", current_lr, epoch)

            # 保存checkpoint
            val_metric = val_metrics.get('loss', train_metrics.get('loss', float('inf')))
            is_best = val_metric < self.best_val_metric
            if is_best:
                self.best_val_metric = val_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                save_checkpoint(
                    state={
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler_state_dict': (
                            self.lr_scheduler.state_dict() if self.lr_scheduler else {}
                        ),
                        'best_metric': self.best_val_metric,
                    },
                    checkpoint_dir=self.checkpoint_dir,
                    is_best=is_best,
                    max_keep=5,
                )

            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"早停: 验证指标连续 {self.patience_counter} 轮未改善")
                break

        total_time = time.time() - train_start_time
        self.writer.close()

        return {
            "total_epochs": self.current_epoch + 1,
            "best_val_metric": self.best_val_metric,
            "total_time": total_time,
        }

    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 数据移到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 前向传播
            self.optimizer.zero_grad()
            loss, metrics = self.criterion(self.model, batch)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_val
                )

            self.optimizer.step()

            # 累积指标
            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            n_batches += 1
            self.global_step += 1

        # 平均
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
        avg_metrics['loss'] = total_loss / n_batches

        return avg_metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss, metrics = self.criterion(self.model, batch)

                total_loss += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                n_batches += 1

        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
        avg_metrics['loss'] = total_loss / n_batches

        return avg_metrics


def load_config(config_path: str) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    print("通用训练器模块")
    print("请使用 train_generator.py 或 train_multitask.py 启动训练")
