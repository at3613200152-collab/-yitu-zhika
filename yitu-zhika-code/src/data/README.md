# 数据加载器模块

多源数据加载，当前主线是 **HSIFoodIngr-64**（Phase1 生成器）与 **Nutrition5k**（Phase2 多任务）两套数据集。

## 当前状态（2026-09-10）

> 本文件中的数据集构成、划分与统计数字最后一次审计于 **2026-09-10**。
> 早前文档中的 `nirscene1`、`capsicum`、"HSIFoodIngr-64 123 样本 × 5 权重" 等描述均已过时；
> 当前实验不再使用 nirscene1 / capsicum 数据源，权重采样也不是现行协议的一部分。

## 模块清单

| 文件 | 描述 | 数据格式 |
|------|------|----------|
| [hsi_dataset.py](hsi_dataset.py) | HSI 高光谱数据加载器（当前主线） | ENVI .hdr/.dat / H5，选 640/550/460/860nm 波段 |
| [hsi_manifest.py](hsi_manifest.py) | Phase1 全量数据集清单（`data/hsi_full_v3/`） | .npy memmap |
| [nutrition5k_loader.py](nutrition5k_loader.py) | Nutrition5k 营养数据加载器（当前主线） | CSV + 扁平分片图片目录 |
| [meal_manifest.py](meal_manifest.py) | Phase2 餐盘清单（RGB / RGB+NIR 对照） | CSV + 图片路径 |
| [meal_expanded_manifest.py](meal_expanded_manifest.py) | 扩展集餐盘清单 | CSV + 图片路径 |
| [meal_macros_manifest.py](meal_macros_manifest.py) | 宏量营养素（脂肪/碳水/蛋白）清单 | CSV |
| [meal_macros_corrected_manifest.py](meal_macros_corrected_manifest.py) | 修正标签体系后的宏量营养素清单 | CSV |
| [build_categories.py](build_categories.py) | 类别标签构建与支持度检查 | CSV |
| [pix2pix_dataset.py](pix2pix_dataset.py) | Pix2PixHD 格式加载器（**历史兼容**，非当前协议） | train_A/train_B PNG, 3通道, 256×256 |

## 数据格式说明

### HSI 格式（hsi_dataset.py）— 当前主线
- 读取高光谱数据，RGB 由可见光波段（640/550/460nm）合成，NIR 取 **860nm** 单波段
- 归一化：**训练集统计量固定缩放** low=0.17514 / high=1.67553 → [0,1] → [-1,1]（验证/测试复用训练集统计量，不做逐图归一化）
- 数据源：**HSIFoodIngr-64**
  - 144 对 RGB/NIR 扫描（860nm），小数据协议划分 train 93 / val 18 / test 33
  - 全量数据版本由 `scripts/build_hsi_full_dataset.py` 构建为 `data/hsi_full_v3/{rgb,nir}_{train,val,test}.npy` memmap
    （3389 对：train 2772 / val 290 / **test 327**）
  - 小数据模型与全量模型都在**同一批 327 张 test** 上评估，这是两者可比的依据

#### NIR↔RGB 对齐（重要）
HSIFoodIngr-64 自带的 NIR 相对 RGB **旋转了 90°**，直接配对会让生成器学到错误目标。构建脚本按下面方式对齐：

```python
nir01 = np.clip(SLOPE * np.rot90(nir_u16.astype(np.float32), 3) + INTERCEPT, 0, 1)
# SLOPE     = 4.0411221561953425e-05
# INTERCEPT = -0.11354496330022812
```

- 对齐由 144 张重叠扫描标定得到
- 对齐后指标：平均绝对差 **0.0029**，相关系数 **0.99965**，标定 **R² = 0.9990**

### Nutrition5k 格式（nutrition5k_loader.py）— 当前主线
- 从 dishes.csv 查找卡路里 / 重量标签，从 dish_ingredients.csv 获取食材组成
- 图片从扁平目录按 dish_id 检索
- 输出：`(image[4,H,W], {calorie, weight, mass, category})`
- **回归目标统计量只由训练集计算**，验证/测试集复用训练集统计量
- 分类只统计冻结测试集中**有样本**的类别（见下方说明）

### Pix2PixHD 格式（pix2pix_dataset.py）— 历史兼容
- 读取 train_A(RGB)/train_B(NIR) 文件夹的 PNG 图片，NIR 的 R=G=B，取第 1 通道转单通道
- 归一化：RGB 和 NIR 都先 `÷255` → `[0,1]` → `×2-1` → `[-1,1]`，自动 resize 到 256×256
- 该加载器保留用于复现早期实验；早期的 `nirscene1`（2597 对）与 `capsicum`（592 对）数据源**已不在当前实验中使用**

## 数据统计（2026-09-10 审计）

| 数据源 | 训练集 | 验证集 | 测试集 | 备注 |
|--------|--------|--------|--------|------|
| HSIFoodIngr-64（小数据协议） | 93 对扫描 | 18 对扫描 | 33 对扫描 | 860nm；小数据模型自测集 |
| HSIFoodIngr-64（全量，`data/hsi_full_v3/`） | 2772 对 | 290 对 | **327 对** | 小数据模型与全量模型共用此 test 划分 |
| Nutrition5k（多任务） | 2188 道菜 | 567 道菜 | 507 道菜（**冻结测试集**） | 目标统计量仅来自训练集；扩展清单 3390 |

## 类别支持度说明

冻结测试划分中以下类别**无测试样本**，相关指标不计入：

| 类别 | 情况 |
|------|------|
| `soup_stew` | 测试集 0 样本 |
| `dessert` | 验证集 0 样本 |
| `sauce_condiment` | 测试集 0 样本 |

因此分类 **Macro-F1 仅在 9 个受支持类别上报告**，不应表述为全类别指标。
