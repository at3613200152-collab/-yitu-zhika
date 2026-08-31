"""
Checkpoint保存/续训逻辑
========================
支持自动检测最新checkpoint、恢复model/optimizer/epoch/lr_scheduler。

Checkpoint内容:
    {
        'epoch': int,
        'model_state_dict': dict,
        'optimizer_state_dict': dict,
        'lr_scheduler_state_dict': dict (可选),
        'best_metric': float,
        'config': dict,
    }
"""

import os
import glob
import torch
from typing import Optional, Dict, Any
from collections import OrderedDict


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint_epoch_{epoch:04d}.pt",
    is_best: bool = False,
    max_keep: int = 5,
) -> str:
    """保存checkpoint

    Args:
        state: 要保存的状态字典
        checkpoint_dir: 保存目录
        filename: 文件名模板，{epoch}会被替换
        is_best: 是否为最佳模型
        max_keep: 最多保留的checkpoint数量 (0=不限制)

    Returns:
        保存的文件路径
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 替换文件名中的占位符
    fname = filename.format(epoch=state.get('epoch', 0))
    filepath = os.path.join(checkpoint_dir, fname)

    # 保存checkpoint
    torch.save(state, filepath)
    print(f"Checkpoint已保存: {filepath}")

    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_path)
        print(f"最佳模型已保存: {best_path}")

    # 清理旧checkpoint
    if max_keep > 0:
        cleanup_old_checkpoints(checkpoint_dir, max_keep, pattern="checkpoint_epoch_*.pt")

    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """加载checkpoint，恢复训练状态

    Args:
        checkpoint_path: checkpoint文件路径
        model: 模型
        optimizer: 优化器 (可选)
        lr_scheduler: 学习率调度器 (可选)
        device: 设备

    Returns:
        checkpoint中保存的额外信息 (epoch, best_metric等)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")

    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 恢复模型权重
    model_state = checkpoint.get('model_state_dict', checkpoint)
    # 处理DataParallel的module.前缀
    model_state = _strip_module_prefix(model_state)
    model.load_state_dict(model_state)

    # 恢复优化器
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 恢复学习率调度器
    if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    # 返回额外信息
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', float('inf')),
    }

    print(f"  恢复到第 {info['epoch']} 轮, 最佳指标: {info['best_metric']:.4f}")
    return info


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """查找目录中最新的checkpoint

    按修改时间排序，返回最新的checkpoint文件路径。

    Args:
        checkpoint_dir: checkpoint目录

    Returns:
        最新checkpoint路径，不存在则返回None
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    # 查找所有checkpoint文件
    patterns = ["checkpoint_epoch_*.pt", "checkpoint_*.pt", "*.pt"]
    checkpoints = []

    for pattern in patterns:
        checkpoints.extend(glob.glob(os.path.join(checkpoint_dir, pattern)))

    # 排除best_model.pt
    checkpoints = [p for p in checkpoints if "best_model" not in os.path.basename(p)]

    if not checkpoints:
        return None

    # 按修改时间排序
    latest = max(checkpoints, key=os.path.getmtime)
    return latest


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    max_keep: int,
    pattern: str = "checkpoint_epoch_*.pt",
):
    """清理旧checkpoint，只保留最新的N个

    Args:
        checkpoint_dir: checkpoint目录
        max_keep: 最多保留数量
        pattern: 文件名模式
    """
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, pattern)),
        key=os.path.getmtime,
    )

    # 删除旧文件
    while len(checkpoints) > max_keep:
        old_file = checkpoints.pop(0)
        os.remove(old_file)
        print(f"删除旧checkpoint: {old_file}")


def _strip_module_prefix(state_dict: Dict) -> Dict:
    """去除DataParallel的'module.'前缀"""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[name] = value
    return new_state_dict


def save_model_only(
    model: torch.nn.Module,
    save_path: str,
):
    """只保存模型权重（不含optimizer等）

    Args:
        model: 模型
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = model.state_dict()
    # 去除module前缀
    state_dict = _strip_module_prefix(state_dict)
    torch.save({'model_state_dict': state_dict}, save_path)
    print(f"模型权重已保存: {save_path}")


if __name__ == "__main__":
    # 测试checkpoint逻辑
    import tempfile

    # 创建临时目录
    tmpdir = tempfile.mkdtemp()
    print(f"临时目录: {tmpdir}")

    # 模拟保存和加载
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        # 模拟训练
        loss = torch.randn(1).item()

        # 保存
        save_checkpoint(
            state={
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': loss,
            },
            checkpoint_dir=tmpdir,
            is_best=(epoch == 2),
            max_keep=3,
        )

    # 查找最新checkpoint
    latest = find_latest_checkpoint(tmpdir)
    print(f"\n最新checkpoint: {latest}")

    # 加载
    new_model = torch.nn.Linear(10, 5)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    info = load_checkpoint(latest, new_model, new_optimizer)
    print(f"恢复信息: {info}")
