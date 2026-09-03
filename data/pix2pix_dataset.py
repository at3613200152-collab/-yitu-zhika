"""
Pix2PixHD Dataset: RGB+NIR 配对数据集 (Phase1 - 多数据源)
=========================================================
支持 deepNIR 项目的 Pix2PixHD 格式 (train_A/train_B 文件夹)，
可同时加载多个数据目录并合并。

数据格式:
    data_root/
    ├── train_A/   (RGB PNG images)
    ├── train_B/   (NIR PNG images, 3-channel same value)
    ├── val_A/
    ├── val_B/
    └── test_A/, test_B/

NIR 图像的 3 通道值相同，取第 1 通道转为 1 通道。
所有图像 resize 到 img_size (默认 256)。

使用方法:
    from src.data.pix2pix_dataset import build_dataloaders
    train_loader, val_loader = build_dataloaders(
        data_roots=["data/nirscene1_x10", "data/capsicum"],
        batch_size=8, img_size=256)
    for rgb, nir in train_loader:
        # rgb: [B, 3, H, W]  nir: [B, 1, H, W]  值域 [-1, 1]
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Pix2PixHDDataset(Dataset):
    """Pix2PixHD 格式的 RGB+NIR 配对数据集。

    Args:
        data_root: 数据根目录，包含 train_A/train_B/val_A/val_B 子目录
        split: 'train' 或 'val'
        img_size: 输出图像尺寸 (默认 256)
    """

    def __init__(self, data_root: str, split: str = 'train',
                 img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.split = split

        # 确定 A/B 文件夹
        folder_a = f"{split}_A"
        folder_b = f"{split}_B"

        dir_a = os.path.join(data_root, folder_a)
        dir_b = os.path.join(data_root, folder_b)

        if not os.path.isdir(dir_a) or not os.path.isdir(dir_b):
            # 尝试其他命名变体
            for alt_a, alt_b in [(f"{split}_A", f"{split}_B"),
                                 (f"{split}_rgb", f"{split}_nir"),
                                 ("A", "B")]:
                alt_dir_a = os.path.join(data_root, alt_a)
                alt_dir_b = os.path.join(data_root, alt_b)
                if os.path.isdir(alt_dir_a) and os.path.isdir(alt_dir_b):
                    dir_a, dir_b = alt_dir_a, alt_dir_b
                    break

        # val 不存在时回退到 test
        if not os.path.isdir(dir_a) or not os.path.isdir(dir_b):
            if split == 'val':
                test_a = os.path.join(data_root, "test_A")
                test_b = os.path.join(data_root, "test_B")
                if os.path.isdir(test_a) and os.path.isdir(test_b):
                    dir_a, dir_b = test_a, test_b
                    print(f"[Pix2PixHD] {data_root}: val not found, using test as val")

        # 匹配 A/B 配对（按文件名排序后一一对应）
        files_a = sorted(glob.glob(os.path.join(dir_a, "*.png")) +
                         glob.glob(os.path.join(dir_a, "*.jpg")) +
                         glob.glob(os.path.join(dir_a, "*.jpeg")))
        files_b = sorted(glob.glob(os.path.join(dir_b, "*.png")) +
                         glob.glob(os.path.join(dir_b, "*.jpg")) +
                         glob.glob(os.path.join(dir_b, "*.jpeg")))

        # 按文件名 stem 匹配，确保 A/B 一一对应
        self.pairs = []
        b_map = {os.path.splitext(os.path.basename(f))[0]: f for f in files_b}
        for fa in files_a:
            stem = os.path.splitext(os.path.basename(fa))[0]
            # 尝试精确匹配和去后缀匹配
            if stem in b_map:
                self.pairs.append((fa, b_map[stem]))
            else:
                # 尝试去掉 _A 后缀
                stem_clean = stem.rstrip('A').rstrip('_') if stem.endswith('_A') else stem
                if stem_clean in b_map:
                    self.pairs.append((fa, b_map[stem_clean]))

        # 如果精确匹配失败，退回到按索引配对
        if not self.pairs and files_a and files_b:
            n = min(len(files_a), len(files_b))
            self.pairs = list(zip(files_a[:n], files_b[:n]))

        print(f"[Pix2PixHD] {data_root} {split}: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, nir_path = self.pairs[idx]

        # 读取 RGB (3通道)
        rgb_img = Image.open(rgb_path).convert('RGB')
        if rgb_img.size != (self.img_size, self.img_size):
            rgb_img = rgb_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0  # [H, W, 3]

        # 读取 NIR (可能3通道值相同，取第1通道)
        nir_img = Image.open(nir_path).convert('RGB')
        if nir_img.size != (self.img_size, self.img_size):
            nir_img = nir_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        nir_arr = np.array(nir_img, dtype=np.float32) / 255.0  # [H, W, 3], [0, 1]
        # 取第1通道 (deepNIR 格式: R=G=B)
        nir_arr = nir_arr[:, :, 0]  # [H, W]

        # 转 [C, H, W] 并归一化到 [-1, 1]
        rgb = np.transpose(rgb_arr, (2, 0, 1))  # [3, H, W]
        nir = nir_arr[np.newaxis, :, :]  # [1, H, W]

        rgb_tensor = torch.from_numpy(rgb * 2.0 - 1.0)
        nir_tensor = torch.from_numpy(nir * 2.0 - 1.0)

        return rgb_tensor, nir_tensor


class MultiSourceDataset(Dataset):
    """合并多个数据源的 Dataset。

    支持同时加载 Pix2PixHD 格式和 HSI 格式的数据。
    """

    def __init__(self, datasets):
        """Args:
            datasets: list of (Dataset, weight) 或 list of Dataset
        """
        self.datasets = []
        self.indices = []  # (dataset_idx, sample_idx)
        for i, ds in enumerate(datasets):
            if isinstance(ds, tuple):
                ds, weight = ds
            else:
                weight = 1.0
            n = len(ds)
            repeat = max(1, int(weight * n))
            for j in range(repeat):
                self.indices.append((i, j % n))
            self.datasets.append(ds)

        total = len(self.indices)
        print(f"[MultiSource] Combined: {total} samples from {len(self.datasets)} sources")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.indices[idx]
        return self.datasets[ds_idx][sample_idx]


def build_dataloaders(data_roots, batch_size=8, num_workers=4, img_size=256,
                      hsi_root=None, hsi_weight=1.0):
    """构建多数据源的 dataloader。

    Args:
        data_roots: list of str, Pix2PixHD 格式的数据根目录
        batch_size: batch size
        num_workers: 数据加载线程数
        img_size: 图像尺寸
        hsi_root: 可选, HSI 格式数据根目录 (HSIFoodIngr-64)
        hsi_weight: HSI 数据的采样权重 (用于平衡数据量)

    Returns:
        (train_loader, val_loader)
    """
    train_datasets = []
    val_datasets = []

    # Pix2PixHD 格式数据
    for root in data_roots:
        root_abs = root if os.path.isabs(root) else os.path.join(os.getcwd(), root)
        if os.path.isdir(root_abs):
            train_datasets.append(Pix2PixHDDataset(root_abs, 'train', img_size))
            val_datasets.append(Pix2PixHDDataset(root_abs, 'val', img_size))

    # HSI 格式数据 (可选)
    if hsi_root and os.path.isdir(hsi_root):
        try:
            from data.hsi_dataset import HSIFoodIngrDataset
            hsi_train = HSIFoodIngrDataset(hsi_root, split='train', img_size=img_size)
            hsi_val = HSIFoodIngrDataset(hsi_root, split='val', img_size=img_size)
            # HSI 数据量小，用权重放大
            if hsi_weight != 1.0:
                train_datasets.append((hsi_train, hsi_weight))
                val_datasets.append((hsi_val, hsi_weight))
            else:
                train_datasets.append(hsi_train)
                val_datasets.append(hsi_val)
        except ImportError:
            print("[Warning] Cannot import HSIFoodIngrDataset, skipping HSI data")

    # 合并
    if len(train_datasets) == 1:
        train_ds = train_datasets[0] if isinstance(train_datasets[0], Dataset) else train_datasets[0][0]
    else:
        train_ds = MultiSourceDataset(train_datasets)
    if len(val_datasets) == 1:
        val_ds = val_datasets[0] if isinstance(val_datasets[0], Dataset) else val_datasets[0][0]
    else:
        val_ds = MultiSourceDataset(val_datasets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"  Total train: {len(train_ds)} samples, val: {len(val_ds)} samples")
    return train_loader, val_loader


if __name__ == '__main__':
    import sys
    roots = sys.argv[1:] if len(sys.argv) > 1 else ["data/nirscene1_x10"]
    print(f"=== Pix2PixHD Dataset Test ===")
    print(f"Data roots: {roots}")
    train_loader, val_loader = build_dataloaders(roots, batch_size=4, img_size=256)
    if len(train_loader.dataset) > 0:
        rgb, nir = train_loader.dataset[0]
        print(f"  rgb: {rgb.shape}, range [{rgb.min():.2f}, {rgb.max():.2f}]")
        print(f"  nir: {nir.shape}, range [{nir.min():.2f}, {nir.max():.2f}]")
    else:
        print("  No samples found!")
