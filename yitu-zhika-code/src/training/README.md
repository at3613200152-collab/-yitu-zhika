# 训练管线模块

Phase1 生成器训练 + Phase2 多任务训练 + 检查点管理。

## 模块清单

| 文件 | 描述 | 阶段 |
|------|------|------|
| [train_generator.py](train_generator.py) | Pix2Pix 训练（7层U-Net + PatchGAN） | Phase1 |
| [train_multitask.py](train_multitask.py) | 多任务 ResNet50 训练 | Phase2 |
| [checkpoint.py](checkpoint.py) | 检查点保存/加载/恢复 | 通用 |

## Phase1 训练流程

1. 加载 RGB-NIR 配对数据（MultiSource: nirscene1 + HSI + capsicum）
2. 7层 Deep U-Net 生成器：RGB[3,256,256] → NIR[1,256,256]
3. 支持加载预训练权重（`--pretrained` 参数，队友PSNR 25.8模型）
4. PatchGAN 判别器：RGB+NIR 拼接 → 真伪概率
5. 交替训练：先更新判别器，再更新生成器
6. 损失：L1（λ=100）+ 对抗损失（GAN loss）
7. 每10轮保存检查点到 `checkpoints/phase1/`，最终保存 `final_model.pth`

## Phase2 训练流程

1. 加载 Nutrition5k 数据 + Phase1 生成器
2. RGB 经生成器生成 NIR，拼成 4 通道输入 ResNet50
3. 多任务输出：卡路里（回归）+ 重量（回归）+ 分类
4. 损失：L1 + CrossEntropy 加权组合
5. 验证集监控 val_loss，保存最优模型到 `checkpoints/multitask/best_model.pt`

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 200 (Phase1) / 100 (Phase2) | 训练轮数 |
| batch_size | 8 | 批次大小 |
| lr | 2e-4 (Phase1) / 1e-3 (Phase2) | 学习率 |
| num_workers | 2 | DataLoader 工作进程数 |
| save_freq | 10 | 检查点保存频率 |
| pretrained | None | 预训练权重路径（Phase1） |

## 启动方式

```bash
# 一键启动（推荐）：自动下载→解压→加载预训练→训练
pythonw run_phase1_bg.py

# 手动启动（数据已就绪）
python src/training/train_generator.py --pretrained <path_to_weights.pth>
```
