# 训练数据清洗、提取与 NIR 预测模型训练方法（2026-09-07）

适用范围：`yitu-zhika`（RGB → 预测 NIR → RGB＋NIR 多任务营养估计）。

## 1. 数据源概览

| 数据源 | 用途 | 文件类型 | 数量 | 协议 manifest |
|---|---|---|---:|---|
| HSIFoodIngr-64 (本地子集) | RGB + 真 NIR（859.42 nm）配对 | ENVI `.hdr`/`.dat` + 同步 RGB `.png` | 144 扫描 | `hsi_rgb859_capture_day_aligned_v2` |
| Nutrition5k (官方 ID 子集) | RGB（俯视） + 营养标签（卡路里、重量、宏量） | `dish_XXX_rgb.jpg` + `dishes_verified.csv` | 3,262 餐盘 | `nutrition5k_overhead_official_ids_v1` |
| Nutrition5k (扩容侧视) | 同餐盘多视角 RGB | mp4 视频 | 128 餐盘 / 384 帧 | `video_expansion_v1`（与训练集独立） |

> 两套 manifest 已通过 SHA256 绑定 + 跨集精确像素去重；下游消费方不得绕过 manifest 直接读原文件。

## 2. 清洗与提取

### 2.1 HSIFoodIngr-64 → `data/hsi_prepared_v2/`

入口：[`src/data/hsi_manifest.py`](file:///C:/Users/user/Desktop/yitu-zhika/yitu-zhika-code/src/data/hsi_manifest.py)

| 步骤 | 关键检查 |
|---|---|
| 1. 解析 `.hdr` | ENVI；samples/lines/bands/int dtype/byte order/interleave/header offset/wavelength 全部存在；波长严格单调递增 |
| 2. 文件大小 | `len(.dat) == lines × samples × bands × 4 + header offset`（否则拒收） |
| 3. NIR 提取 | 找 859.42 nm 附近最近波段（要求 |Δλ| ≤ 2 nm）；该立方体非全有限、非常数 |
| 4. 几何校验 | RGB PNG 尺寸 = `samples × lines`；透明通道若存在须不透明（否则拒收） |
| 5. 像素去重 | 全图字节 SHA256 跨文件去重；重复即拒收 |
| 6. 旋转对齐 | 整立方体顺时针旋转 90° 与数据集 PNG 对齐（不旋转直接训练会出现空间错位） |
| 7. 归一化 | 训练集所有像素第 0.5/99.5 分位数固定缩放 + clip [0,1]；验证/测试复用同一系数 |
| 8. resize | 256×256 双线性 → 拼成 `[RGB(3), NIR(1)]` float32，写入 `*.npy` |
| 9. 划分 | 按拍摄日（acquisition date）哈希分桶：第 1 天 = test、第 2 天 = val、其余 = train |
| 10. 落盘 | 每个 npy 同步写 `prepared_sha256`；manifest 记录 `nir_band_index`、`nir_wavelength_nm`、几何与原始 SHA |

清洗结果（已落盘）：

```
data/hsi_prepared_v2/
├── manifest.json       # 144 行；train=93 / val=18 / test=33
├── REFLECTANCE_218.npy # [4, 256, 256] float32 ∈ [0,1]
├── …
└── aligned_audit.json  # 旋转/对齐审计通过
```

### 2.2 Nutrition5k → `results/meal_official_v1/manifest.json`

入口：[`src/data/meal_manifest.py`](file:///C:/Users/user/Desktop/yitu-zhika/yitu-zhika-code/src/data/meal_manifest.py)

| 步骤 | 关键检查 |
|---|---|
| 1. 标签审计 | `dishes_verified.csv`（已纠错 total_mass ↔ total_calories 列对调）SHA256 与审计 `nutrition5k_audit.json` 一致 |
| 2. 官方划分 | 从官方 GCS 拉取 `rgb_train_ids.txt` / `rgb_test_ids.txt`；本地覆盖 2,755 训练 + 507 测试 |
| 3. 验证划分 | 训练集内按 UTC 拍摄日 SHA256 阈值 0.15 划 val；不读测试集做选择 |
| 4. 像素去重 | 完整图像字节 SHA256 跨集合去重；命中即拒收 |
| 5. 标签 | targets = [calories (kcal), mass (g)]；要求 mass>0、finite；零热量保留在 MAE/RMSE，仅 MAPE 分母排除 |
| 6. 类别 | 由配料关键词生成的 11 个粗类别（vegetable / meat / mixed / …），非官方 61 类 |
| 7. 训练统计 | mean/std 只在 train 上计算，下游做归一化 |

清洗结果：train=2,188 / val=567 / test=507，共 3,262 餐盘。

### 2.3 统一训练池 → `data/training_pool/`

入口：`scripts/extract_training_pool.py`（一次性脚本）

将两份 manifest 物理复制到统一目录，并生成 `manifest.json`（包含 SHA256 链路）+ `AUDIT.json`（指标摘要）。

```
data/training_pool/
├── HSI/{train,val,test}/   # *.npy  [4,256,256] float32
├── Meal/{train,val,test}/  # *_rgb.jpg + *_depth.jpg
├── manifest.json
└── AUDIT.json
```

| 池 | 训练 | 验证 | 测试 | 体积 |
|---|---:|---:|---:|---:|
| HSI (paired RGB+NIR) | 93 | 18 | 33 | 144 MB |
| Meal (RGB+depth+labels) | 2,188 | 567 | 507 | 2.18 GB |
| Meal 物理文件数（rgb+depth） | 4,376 | 1,134 | 1,014 | — |

## 3. NIR 预测模型训练方法

### 3.1 模型

`src/models/generator.py::UNetGenerator` — 4 层 U-Net（base=64）：

```
输入 [3,256,256]
  enc1 3→64        (s=2, no BN, LeakyReLU 0.2)
  enc2 64→128
  enc3 128→256
  enc4 256→512
  bottleneck 512→512
  dec4 512→512 (+skip 512 → 1024)  Dropout 0.5
  dec3 1024→256 (+skip 256 → 512)  Dropout 0.5
  dec2 512→128  (+skip 128 → 256)
  dec1 256→64   (+skip 64  → 128)
  head 64→1, Tanh → [1,256,256] ∈ [-1,1]
```

初始化：Normal(0, 0.02)。设计参考 Isola et al., Pix2Pix (CVPR 2017)。

### 3.2 训练脚本

[`src/training/train_hsi_official.py`](file:///C:/Users/user/Desktop/yitu-zhika/yitu-zhika-code/src/training/train_hsi_official.py) — 当前线上 v2 协议。

```powershell
& 'C:\Users\user\miniconda3\envs\yitu\python.exe' -X utf8 `
   src/training/train_hsi_official.py --epochs 100
```

### 3.3 关键超参与训练流程

| 项 | 取值 |
|---|---|
| 输入 | 数据池 `*.npy` 的前 3 通道（RGB，[0,1]） |
| 目标 | 同 `*.npy` 的第 4 通道（NIR 859.42 nm，[0,1]） |
| 归一化 | 模型前后统一 ×2 − 1 到 [-1,1] |
| 优化器 | Adam, lr=2e-4, betas=(.5, .999) |
| 学习率 | ReduceLROnPlateau factor=0.5 patience=5（监控 val L1） |
| 损失 | L1（[-1,1] 域）；验证按同 L1 选 best |
| 精度 | BF16 autocast；loss/grad 始终 float32；grad-norm clip=1，non-finite 报错 |
| 批大小 | 8 |
| 早停 | 验证 20 轮未改善即停 |
| 训练预算 | 8h 硬上限（防止单次跑超长） |
| 种子 | 42（DataLoader 用 `torch.Generator().manual_seed(42+epoch)` 控制随机） |
| 增强 | 仅训练集：按 `id`+`epoch` 哈希决定水平翻转（确定性可复现） |
| 冒烟 | 真实数据先跑 2 批 train+val，丢弃更新，重新初始化正式模型 |
| 续训 | last.pt 读 `epoch/optimizer/scheduler/history`；config 不匹配则拒绝 |
| 选 best | 验证集 L1（[-1,1] 域），test 在训练结束后用 best 跑一次 |

### 3.4 评测

`pass_epoch` 收集每个样本的：

- `l1_01` — [0,1] 域 L1
- `psnr_db` — `10·log10(1/MSE)`，MSE 下限 1e-12 防止除零
- `ssim` — `skimage.metrics.structural_similarity(data_range=1.0)`

汇报时使用 `data_range=1.0`（[0,1] 域），与代码 `src/evaluation/eval_generator.py` 一致。

### 3.5 当前 v2 结果（已审计）

来源：[docs/hsi_source_audit.md](file:///C:/Users/user/Desktop/yitu-zhika/yitu-zhika-code/docs/hsi_source_audit.md) 与 `results/hsi_unet_v2/test_metrics.json`

| 指标 | 数值 |
|---|---:|
| 最佳轮 | 96 / 100 |
| 测试 PSNR | 23.333 dB |
| 测试 SSIM | 0.81198 |
| 测试 L1（[0,1]） | — |
| 训练样本 | 93 |
| 验证样本 | 18 |
| 测试样本 | 33 |

### 3.6 配套审计与门禁

| 文件 | 作用 |
|---|---|
| `data/hsi_prepared_v2/manifest.json` | 144 配对，含 hash、分桶、缩放参数 |
| `data/hsi_prepared_v2/aligned_audit.json` | 旋转/对齐审计 |
| `results/hsi_unet_v2/protocol.json` | 训练配置（架构、超参、code hash、缩放参数） |
| `results/hsi_unet_v2/completion_audit.json` | 训练后生成（best/last 可回读 + 预测哈希一致） |
| `results/hsi_unet_v2/test_metrics.json` | 最终测试指标 |
| `results/hsi_unet_v2/test_predictions.json` | 逐图预测 + 像素 L1/PSNR/SSIM |

门禁在 [`scripts/audit_hsi_completion.py`](file:///C:/Users/user/Desktop/yitu-zhika/yitu-zhika-code/scripts/audit_hsi_completion.py)：
- `nir_scale.low/high` 必须由训练集分位数固定
- `raw_shape/interleave/nir_band_index` 与原始 `.hdr` 重新解析一致
- `prepared_sha256` 重新算必须与 manifest 一致
- `best.pt` / `last.pt` 能以 `weights_only=True` 回读，所有权重有限
- `epochs.json` 的最优轮与 `best.pt['best_epoch']` 与 `test_metrics['best_epoch']` 一致

## 4. 复现命令

```powershell
# 0. 进入项目
Set-Location 'C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code'

# 1. 重建/校验 HSI 训练数据
& 'C:\Users\user\miniconda3\envs\yitu\python.exe' -X utf8 src/data/hsi_manifest.py

# 2. 重建/校验 Nutrition5k 训练数据
& 'C:\Users\user\miniconda3\envs\yitu\python.exe' -X utf8 src/data/meal_manifest.py

# 3. 导出统一训练池（含 manifest + AUDIT）
python scripts/extract_training_pool.py

# 4. 训练 NIR 生成器（CPU 不会触发，仅在 GPU 上执行）
& 'C:\Users\user\miniconda3\envs\yitu\python.exe' -X utf8 src/training/train_hsi_official.py --epochs 100

# 5. 训练后审计
& 'C:\Users\user\miniconda3\envs\yitu\python.exe' -X utf8 scripts/audit_hsi_completion.py
```

## 5. 边界与已知限制

- 本机 HSI 仅 144 扫描，HSIFoodIngr-64 完整数据集为 3,389 配对，跨域（菜品域、拍摄设备）不能直接外推。
- NIR 缩放系数仅由训练集分位数决定，复用同一缩放不接触验证/测试。
- v2 协议按「拍摄日」分桶并不等价于真实独立餐盘分组；记录在 manifest `limitations` 中。
- Nutrition5k 没有真实 NIR；其 RGB 仅用于训练营养估计网络；可由 v2 训练好的生成器产出预测 NIR。
- 单种子、单 GPU；多种子稳定性与跨设备可复现性需在后续轮次补充。
