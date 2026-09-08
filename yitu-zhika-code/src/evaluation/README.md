# 评估工具模块

Phase1 生成器质量评估 + Phase2 多任务模型评估。

## 模块清单

| 文件 | 描述 | 指标 |
|------|------|------|
| [eval_generator.py](eval_generator.py) | Phase1 生成器评估 | PSNR, SSIM, L1 |
| [eval_multitask.py](eval_multitask.py) | Phase2 多任务评估 | MAE, MAPE, RMSE, R² |

## Phase1 评估指标

- **PSNR**（峰值信噪比）：值域 [-1,1]，`10*log10(4.0/mse)`，per-image 平均
- **SSIM**（结构相似性）：衡量生成 NIR 与真实 NIR 的结构一致性
- **L1 Loss**：像素级绝对误差

| 版本 | PSNR | SSIM | L1 | 备注 |
|------|------|------|-----|------|
| 旧模型(4层U-Net) | 19.29±2.10 dB | 0.799±0.040 | 0.1718 | 从头训练200轮 |
| 新模型(7层U-Net) | 🔄 重训中 | 🔄 重训中 | 🔄 重训中 | 预训练PSNR25.8 + fine-tune |
| 论文参考 | 30.61 dB | 0.865 | — | — |

## Phase2 评估指标

| 指标 | 卡路里 | 重量 |
|------|--------|------|
| MAE | 25.38 kcal | 38.54 g |
| MAPE | 16.97% | 29.92% |
| RMSE | 37.55 | 63.87 |
| R² | 0.9403 | 0.9036 |

- 评估产物输出到 `results/phase2_eval/`：
  - `metrics.txt` — 数值指标
  - `scatter_plot.png` — 预测 vs 真值散点图
  - `training_curves.png` — 训练曲线
- 论文参考：卡路里 MAPE=12.13%, 识别率=98.24%
