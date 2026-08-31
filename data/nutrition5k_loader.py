"""
Nutrition5k 数据集加载器
========================
加载Nutrition5k数据集，包含：
- RGB图像
- 营养元数据（卡路里、重量、食材列表等）

Nutrition5k数据集结构:
    Nutrition5k/
    ├── imagery/
    │   ├── dish_1/
    │   │   ├── rgb.png
    │   │   ├── depth.png
    │   │   └── ...
    │   └── ...
    ├── dish_metadata/
    │   └── dish_metadata_cafe1.csv
    └── ingredient_metadata/
        └── ingredient_metadata.csv

CSV字段:
    dish_id, total_calories, total_mass, total_macronutrients, ...
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional, Dict, List, Any


class Nutrition5kDataset(Dataset):
    """Nutrition5k 数据集

    Args:
        root_dir: Nutrition5k数据集根目录
        split: 数据划分 (train/val/test)
        img_size: 输出图像尺寸
        augmentation: 是否启用数据增强
        metadata_csv: 元数据CSV文件路径（为None则自动搜索）
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 256,
        augmentation: bool = True,
        metadata_csv: Optional[str] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        # 加载元数据
        self.metadata = self._load_metadata(metadata_csv)

        # 划分数据集
        all_dish_ids = list(self.metadata.keys())
        n_total = len(all_dish_ids)

        # 固定随机种子确保划分一致
        rng = np.random.RandomState(42)
        rng.shuffle(all_dish_ids)

        train_end = int(n_total * 0.8)
        val_end = int(n_total * 0.9)

        if split == "train":
            dish_ids = all_dish_ids[:train_end]
        elif split == "val":
            dish_ids = all_dish_ids[train_end:val_end]
        else:
            dish_ids = all_dish_ids[val_end:]

        # 过滤掉图像文件不存在的dish
        self.samples = []
        for dish_id in dish_ids:
            img_path = self._find_rgb_image(dish_id)
            if img_path is not None:
                self.samples.append({
                    "dish_id": dish_id,
                    "img_path": img_path,
                    **self.metadata[dish_id],
                })

        # 数据增强
        if augmentation and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def _load_metadata(self, metadata_csv: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """加载元数据CSV

        Returns:
            dict: {dish_id: {calories, mass, category, ingredients, ...}}
        """
        if metadata_csv is None:
            # 自动搜索CSV文件
            metadata_dir = os.path.join(self.root_dir, "dish_metadata")
            if os.path.isdir(metadata_dir):
                csv_files = [f for f in os.listdir(metadata_dir) if f.endswith('.csv')]
                if csv_files:
                    metadata_csv = os.path.join(metadata_dir, csv_files[0])

        metadata = {}

        if metadata_csv and os.path.exists(metadata_csv):
            with open(metadata_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dish_id = row.get('dish_id', '')
                    if not dish_id:
                        continue
                    try:
                        metadata[dish_id] = {
                            "calories": float(row.get('total_calories', 0)),
                            "mass": float(row.get('total_mass', 0)),
                            "protein": float(row.get('total_protein_g', 0)),
                            "carb": float(row.get('total_carb_g', 0)),
                            "fat": float(row.get('total_fat_g', 0)),
                            "category": row.get('category', 'unknown'),
                            "ingredients": row.get('ingredients', ''),
                        }
                    except (ValueError, TypeError):
                        metadata[dish_id] = {
                            "calories": 0.0, "mass": 0.0,
                            "protein": 0.0, "carb": 0.0, "fat": 0.0,
                            "category": "unknown", "ingredients": "",
                        }

        if not metadata:
            # 如果没有CSV，从目录结构推断
            imagery_dir = os.path.join(self.root_dir, "imagery")
            if os.path.isdir(imagery_dir):
                for dish_dir in os.listdir(imagery_dir):
                    dish_path = os.path.join(imagery_dir, dish_dir)
                    if os.path.isdir(dish_path):
                        metadata[dish_dir] = {
                            "calories": 0.0, "mass": 0.0,
                            "protein": 0.0, "carb": 0.0, "fat": 0.0,
                            "category": "unknown", "ingredients": "",
                        }

        return metadata

    def _find_rgb_image(self, dish_id: str) -> Optional[str]:
        """查找dish对应的RGB图像路径"""
        # 尝试多种路径模式
        candidates = [
            os.path.join(self.root_dir, "imagery", dish_id, "rgb.png"),
            os.path.join(self.root_dir, "imagery", dish_id, "rgb.jpg"),
            os.path.join(self.root_dir, "imagery", dish_id, "rgb.png"),
            os.path.join(self.root_dir, "images", f"{dish_id}.png"),
            os.path.join(self.root_dir, "images", f"{dish_id}.jpg"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本

        Returns:
            dict: {
                'image':       Tensor [3, H, W]    归一化RGB图像
                'calories':    float                卡路里 (kcal)
                'mass':        float                重量 (g)
                'protein':     float                蛋白质 (g)
                'carb':        float                碳水 (g)
                'fat':         float                脂肪 (g)
                'category':    str                  类别
                'ingredients': str                  食材列表
                'dish_id':     str                  菜品ID
            }
        """
        sample = self.samples[idx]

        # 加载图像
        image = Image.open(sample["img_path"]).convert("RGB")
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "calories": float(sample["calories"]),
            "mass": float(sample["mass"]),
            "protein": float(sample["protein"]),
            "carb": float(sample["carb"]),
            "fat": float(sample["fat"]),
            "category": sample["category"],
            "ingredients": sample["ingredients"],
            "dish_id": sample["dish_id"],
        }


def create_nutrition5k_dataloader(
    root_dir: str,
    split: str = "train",
    img_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation: bool = True,
) -> DataLoader:
    """创建Nutrition5k数据加载器的便捷函数"""
    dataset = Nutrition5kDataset(
        root_dir=root_dir,
        split=split,
        img_size=img_size,
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
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/Nutrition5k"
    print(f"测试Nutrition5k数据加载器，路径: {data_path}")

    dataset = Nutrition5kDataset(
        root_dir=data_path,
        split="train",
        img_size=256,
    )
    print(f"数据集大小: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: {val}")
