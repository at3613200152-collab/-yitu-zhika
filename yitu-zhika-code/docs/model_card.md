# 上线模型卡片 (Model Card)

**生成日期:** 2026-09-08
**审核原则:** 权重、数据清单、指标对应一致；不仅按测试集成绩挑模型

## 一、上线模型选定

**选定模型:** `meal_rgb_official_v1`
**权重路径:** `checkpoints/meal_rgb_official_v1/best.pt` (281.7 MB)
**模型 SHA256:** `3f189a439fca3b79ecd820ae68605bcbfe61679def14f8dfa548c34261c89cf0`

## 二、数据清单

| 项 | 值 |
|----|----|
| 数据集 | Nutrition5k |
| Manifest SHA256 | `f1e293b826c98e517c3d1179871ec0a10d8ee471652b5d2e6f89f9ed1c755daa` |
| 训练样本 | 2188 |
| 验证样本 | 567 |
| 测试样本 | 507 |
| 总样本 | 3262 |

### 类别清单（11 类粗分类）

0=dairy, 1=dessert, 2=egg, 3=grain, 4=meat, 5=mixed, 6=other,
7=sauce_condiment, 8=seafood, 9=soup_stew, 10=vegetable

## 三、指标对应审计

- ✅ manifest SHA256 在 protocol/test_metrics/completion_audit 三处一致
- ✅ checkpoint SHA256 在 test_metrics 与 completion_audit 一致
- ✅ 重新计算的 MAE/RMSE/R² 与原始 test_metrics 完全一致
- ✅ limitations 字段已诚实声明 7 条限制

## 四、训练配置

- 协议: meal_rgb_official_v1
- 种子: 42
- Epochs: 30（早停 epoch 28）
- Batch size: 8
- 输入尺寸: 256×256
- 优化器: AdamW (backbone=1e-5, heads=1e-4)
- 损失: mean(L1/target_std) + 0.2 × category CE
- 设备: NVIDIA RTX 5060 Laptop GPU
- PyTorch: 2.7.1+cu128

## 五、已知限制（必须告知用户）

1. 本地俯拍子集，不是官方完整多视角 benchmark
2. capture-day 分组是代理，严格餐盘 ID 独立性未完全验证
3. 零卡路里标签保留
4. 历史随机划分权重，不能作为干净的官方测试基线
5. 内部 RGB 对照，不是外部方法复现
6. 11 类粗分类，不是官方食物识别类别
7. 单种子试点，未做多种子稳定性验证
8. 预测未 clamp，可能出现负值（测试集 4 条负预测）

## 六、上线决策

- 上线模型: meal_rgb_official_v1/best.pt
- 接口返回 model_version 字段
- 启动时校验权重 SHA256
- 不用 s0_seed42（无 completion_audit）
- 不用 meal_nir_official_v1（依赖 NIR 生成器）
- 不调 clamp（保留预测负值可能）

## 七、复核签字

- 数据清单: ✅ 一致
- 指标对应: ✅ 一致
- 代码溯源: ✅ 一致
- limitations: ✅ 诚实声明 7 条
- 测试集成绩: 仅作参考，非唯一依据
