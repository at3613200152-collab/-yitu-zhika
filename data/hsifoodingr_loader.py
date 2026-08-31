"""
HSIFoodIngr-64 数据集加载器
============================
加载HDF5格式的HSIFoodIngr数据集，包含：
- hsi: 高光谱图像 (64波段)
- rgb: 对应RGB图像
- mask: 食物区域掩码
- meta: 元数据（类别标签等）

数据归一化策略：
- RGB: [0, 255] → [0, 1] (除以255)
- HSI: 按波段归一化到[0, 1]（使用全局最大值或按样本最大值）
- NIR通道: 从HSI中提取特定波段（默认第40波段，约850nm）
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List


class HSIFoodIngrDataset(Dataset):
    """HSIFoodIngr-64 数据集

    Args:
        root_dir: HDF5文件所在目录
        split: 数据集划分 (train/val/test)
        img_size: 输出图像尺寸
        nir_band: 从HSI中提取NIR的波段索引 (0-63)
        augmentation: 是否启用数据增强
        normalize_hsi: HSI归一化方式 ('global' / 'per_sample')
    """

    # HSIFoodIngr-61 类别名称
    CLASS_NAMES = [
        "apple", "banana", "beef", "bread", "broccoli", "burger", "cabbage",
        "cake", "candy", "carrot", "cheese", "chicken", "chocolate", "coffee",
        "cookie", "corn", "cracker", "cucumber", "doughnut", "dumpling", "egg",
        "fish", "french_fries", "fried_rice", "grape", "hotdog", "ice_cream",
        "juice", "kiwi", "lemon", "mango", "milk", "mushroom", "noodle",
        "onion", "orange", "pancake", "pasta", "peach", "pear", "pie",
        "pizza", "plum", "potato", "pudding", "rice", "salad", "sandwich",
        "sausage", "shrimp", "soup", "steak", "strawberry", "sushi", "tea",
        "tomato", "waffle", "watermelon", "yogurt", "zucchini", "pretzel"
    ]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 256,
        nir_band: int = 40,
        augmentation: bool = True,
        normalize_hsi: str = "per_sample",
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.nir_band = nir_band
        self.normalize_hsi = normalize_hsi
        self.num_classes = len(self.CLASS_NAMES)

        # 构建样本索引
        self.samples = self._build_sample_index()

        # 数据增强
        if augmentation and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
            ])

        # HSI只做resize和归一化，不做颜色增强
        self.hsi_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
        ])

    def _build_sample_index(self) -> List[Dict[str, Any]]:
        """构建样本索引列表

        扫描root_dir下的HDF5文件，按split划分构建索引。
        文件组织方式:
            root_dir/
            ├── train/
            │   ├── sample_001.h5
            │   └── ...
            ├── val/
            └── test/
        """
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.isdir(split_dir):
            # 兼容单文件模式：所有数据在一个HDF5文件中，通过meta中的split字段划分
            h5_files = [f for f in os.listdir(self.root_dir) if f.endswith('.h5')]
            if h5_files:
                return [{"h5_path": os.path.join(self.root_dir, f), "index": i}
                        for f in h5_files for i in range(self._count_h5_samples(
                            os.path.join(self.root_dir, f)))]
            raise FileNotFoundError(f"找不到数据目录或H5文件: {self.root_dir}")

        h5_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.h5')])
        samples = []
        for h5_file in h5_files:
            h5_path = os.path.join(split_dir, h5_file)
            n_samples = self._count_h5_samples(h5_path)
            for i in range(n_samples):
                samples.append({"h5_path": h5_path, "index": i})

        return samples

    @staticmethod
    def _count_h5_samples(h5_path: str) -> int:
        """统计HDF5文件中的样本数"""
        with h5py.File(h5_path, 'r') as f:
            # 尝试常见key
            for key in ['rgb', 'hsi', 'images', 'data']:
                if key in f:
                    return len(f[key])
            # 取第一个dataset的长度
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    return len(f[key])
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本

        Returns:
            dict: {
                'rgb':        Tensor [3, H, W]  归一化到[0,1]的RGB图像
                'nir':        Tensor [1, H, W]  从HSI提取的NIR通道
                'hsi':        Tensor [64, H, W] 归一化后的高光谱图像
                'mask':       Tensor [1, H, W]  食物区域二值掩码
                'label':      int               食物类别索引
                'label_name': str               食物类别名称
            }
        """
        sample_info = self.samples[idx]
        h5_path = sample_info["h5_path"]
        index = sample_info["index"]

        with h5py.File(h5_path, 'r') as f:
            # 读取RGB图像
            rgb = f['rgb'][index] if 'rgb' in f else f['images'][index]
            # 读取高光谱图像
            hsi = f['hsi'][index] if 'hsi' in f else f['spectral'][index]
            # 读取掩码（如果没有则生成全1掩码）
            if 'mask' in f:
                mask = f['mask'][index]
            else:
                mask = np.ones(hsi.shape[:2], dtype=np.float32)
            # 读取类别标签
            if 'label' in f:
                label = int(f['label'][index])
            elif 'meta' in f and 'category' in f['meta'].attrs:
                label = int(f['meta'].attrs['category'])
            else:
                label = 0

        # 处理RGB
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        elif rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0

        # 转为PIL用于transform
        rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
        rgb_tensor = self.transform(rgb_pil)
        rgb_tensor = transforms.ToTensor()(rgb_tensor)  # [3, H, W], [0, 1]

        # 处理HSI
        hsi = hsi.astype(np.float32)
        if self.normalize_hsi == "per_sample":
            hsi_max = hsi.max()
            if hsi_max > 0:
                hsi = hsi / hsi_max
        elif self.normalize_hsi == "global":
            hsi = hsi / 65535.0  # 16位HSI最大值

        # HSI: [H, W, 64] → [64, H, W]
        if hsi.ndim == 3 and hsi.shape[-1] == 64:
            hsi = np.transpose(hsi, (2, 0, 1))
        elif hsi.ndim == 3 and hsi.shape[0] == 64:
            pass  # 已经是 [64, H, W]
        else:
            # 如果波段数不对，截取或补零到64
            if hsi.ndim == 3:
                hsi = np.transpose(hsi, (2, 0, 1))
            c = hsi.shape[0] if hsi.ndim == 3 else 1
            if c < 64:
                pad = np.zeros((64 - c, hsi.shape[1], hsi.shape[2]), dtype=np.float32)
                hsi = np.concatenate([hsi, pad], axis=0) if hsi.ndim == 3 else np.concatenate([hsi[np.newaxis], pad], axis=0)

        hsi_tensor = torch.from_numpy(hsi).float()
        # Resize HSI到目标尺寸
        hsi_tensor = torch.nn.functional.interpolate(
            hsi_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        # 提取NIR通道（指定波段）
        nir_band = min(self.nir_band, hsi_tensor.shape[0] - 1)
        nir_tensor = hsi_tensor[nir_band:nir_band + 1, :, :]  # [1, H, W]

        # 处理掩码
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='nearest'
        ).squeeze(0)

        label_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else "unknown"

        return {
            "rgb": rgb_tensor,          # [3, H, W]
            "nir": nir_tensor,          # [1, H, W]
            "hsi": hsi_tensor,          # [64, H, W]
            "mask": mask_tensor,        # [1, H, W]
            "label": label,             # int
            "label_name": label_name,   # str
        }


def create_hsifoodingr_dataloader(
    root_dir: str,
    split: str = "train",
    img_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    nir_band: int = 40,
    augmentation: bool = True,
) -> DataLoader:
    """创建HSIFoodIngr数据加载器的便捷函数

    Args:
        root_dir: 数据集根目录
        split: 数据划分 (train/val/test)
        img_size: 图像尺寸
        batch_size: 批大小
        num_workers: DataLoader工作进程数
        nir_band: NIR波段索引
        augmentation: 是否数据增强

    Returns:
        DataLoader实例
    """
    dataset = HSIFoodIngrDataset(
        root_dir=root_dir,
        split=split,
        img_size=img_size,
        nir_band=nir_band,
        augmentation=augmentation,
    )

    shuffle = (split == "train")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )

    return dataloader


if __name__ == "__main__":
    # 测试代码
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/HSIFoodIngr-64"
    print(f"测试HSIFoodIngr数据加载器，路径: {data_path}")

    dataset = HSIFoodIngrDataset(
        root_dir=data_path,
        split="train",
        img_size=256,
    )
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数: {dataset.num_classes}")

    if len(dataset) > 0:
        sample = dataset[0]
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: {val}")
