# 数据加载器模块

多源数据加载，支持三种格式的 RGB-NIR 配对数据。

## 模块清单

| 文件 | 描述 | 数据格式 |
|------|------|----------|
| [pix2pix_dataset.py](pix2pix_dataset.py) | Pix2PixHD 格式数据加载器 | train_A/train_B PNG, 3通道, 256×256 |
| [hsi_dataset.py](hsi_dataset.py) | HSI 高光谱数据加载器 | ENVI .hdr/.dat, 204波段, 选640/550/460/860nm |
| [nutrition5k_loader.py](nutrition5k_loader.py) | Nutrition5k 营养数据加载器 | CSV + 扁平图片目录 |

## 数据格式说明

### Pix2PixHD 格式（pix2pix_dataset.py）
- 读取 train_A(RGB)/train_B(NIR) 文件夹的 PNG 图片
- NIR 图像的 R=G=B，取第1通道转为单通道
- **归一化流程**：RGB和NIR都先 `÷255` → `[0,1]` → `×2-1` → `[-1,1]`
- 自动 resize 到 256×256
- 支持多数据源合并（MultiSourceDataset）+ 权重采样
- 数据源：nirscene1 x10（2597对）、capsicum（592对食物）

### HSI 格式（hsi_dataset.py）
- 读取 ENVI .hdr/.dat 文件，从204个波段中选取640/550/460/860nm
- RGB 从可见光波段合成，NIR 从近红外波段提取
- 归一化：percentile_normalize → [0,1] → [-1,1]
- 数据源：HSIFoodIngr-64（123样本，8类食材，训练权重×5）

### Nutrition5k 格式（nutrition5k_loader.py）
- 从 dishes.csv 查找卡路里/重量标签
- 从 dish_ingredients.csv 获取食材组成
- 图片从 images/ 扁平目录按 dish_id 检索
- 输出：`(image[4,H,W], {calorie, weight, mass, category})`

## 数据统计

| 数据源 | 训练集 | 验证集 | 备注 |
|--------|--------|--------|------|
| nirscene1 x10 | 2597 | 320 | deepNIR数据集 |
| HSIFoodIngr-64 | 123 (×5权重) | 21 | 高光谱数据 |
| capsicum | 592 | 84 | 深度学习NIR数据 |
| **合计** | **3212** | **425** | — |
