# 课程设计报告数据包（2026-09-08）

> 用途：撰写课程设计报告（含算法复现与对比实验分析）时引用。
> 所有数字均可回溯到 `yitu-zhika-code/results/<run>/test_metrics.json`
> 与 `completion_audit.json`/`paired_completion_audit.json`（含 checkpoint/manifest/predictions SHA256 校验链）。
> 模型卡见 `docs/model_card.md`（上线模型 meal_rgb_official_v1 的完整审计说明）。

## 一、数据与任务设定（报告实验章节必备）

| 项 | 值 |
|----|----|
| 数据集 | Nutrition5k（本地俯拍子集，非官方全视角 benchmark） |
| 划分 | 训练 2188 / 验证 567 / 测试 507（冻结，按拍摄日分组作验证代理） |
| Manifest SHA256 | `f1e293b826c98e517c3d1179871ec0a10d8ee471652b5d2e6f89f9ed1c755daa` |
| 任务 | 回归（热量 kcal + 重量 g）+ 11 类粗分类（CE 辅助） |
| 异常过滤 | kcal ∈ [0,2000]、grams ∈ [0,2000]、kcal/g ∈ [0.1,9.0]（零热量样本保留在 MAE/RMSE） |
| 种子 | 单种子 42（报告需注明不确定性：无多种子区间） |

## 二、三模型对比（同一 507 测试集，测试集仅用于选择 best epoch 之后的评估）

| 模型 | 方案 | 热量 MAE | 热量 RMSE | R² | 非零 MAPE | 重量 MAE | 粗类别准确率 | 权重 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `meal_rgb_official_v1` | RGB（内部对照） | 56.55 | 85.13 | 0.8388 | 48.07% | 36.70 | 72.78% | `3f189a43…c89cf0` |
| `meal_nir_official_v1` | RGB＋预测 NIR | 56.21 | 84.90 | 0.8397 | 51.02% | 35.64 | 73.18% | `3a13788a…ffae950` |
| `calorieclip_official_v1` | CalorieCLIP 类外部基线 | 58.74 | 91.28 | 0.8146 | 35.36% | 不支持 | 不支持 | `512227cb…422427f6` |

来源：`results/{meal_rgb_official_v1,meal_nir_official_v1,calorieclip_official_v1}/test_metrics.json`
与 `results/experiment_comparison_v1.md`。

### 对比结论的诚实表述（直接可用于报告）
- RGB 与 RGB＋预测 NIR 同骨干、同初始化共享、同训练预算；RGB＋NIR 热量 MAE 低 0.34 kcal，
  按 64 个拍摄日分组 10000 次 bootstrap 区间 [-3.091, 2.503] kcal 跨过 0；MAPE 反而变差 →
  **尚无稳定增益证据**，应表述为"内部消融未观察到显著提升"，不建议声称多模态优于 RGB。
- CalorieCLIP 为公开结构/config 的本地位实现，非作者精确复现、非官方另一测试集成绩；
  它只输出热量，不提供重量/类别。
- 全部为单种子试点；预测未 clamp，测试集存在 4 条负热量预测（保留原值，不裁剪美化）。

## 三、NIR 生成器（阶段 1，Pix2Pix 风格 U-Net）
- HSI 测试：33 扫描，PSNR 23.333 dB / SSIM 0.81198（来源：`results/experiment_comparison_v1.md`；
  早期 phase1 报告中的 21.5 dB 为旧版本指标，写报告请以本文件与各 run audit 为准）。
- 用途说明：生成的是**相对强度预测图，不是实测光谱**；域迁移到餐盘照片仍有限制
  （见 `meal_nir_official_v1/test_metrics.json` limitations）。

## 四、⚠️ 数字勘误（重要）
- 进度报告表格中主模型 "MAE kcal=49.9，R²=0.867" 在本仓库任何 results/audit 文件中
  **找不到出处**，与 audited 测试集指标（56.55 / 0.8388）不一致。报告/对外材料一律使用本文件
  第二节 audited 数字；如要引用 49.9 需先给出可复现来源，否则按项目约束（移除未经证实的
  精度声明）删除。

## 五、图表材料（结果目录，供报告贴图）
- 训练曲线/散点：`results/phase2_eval_corrected/{training_curves.png,scatter_plot.png}`（legacy 模型）
- 现有汇总：`results/experiment_comparison_v1.{json,md}`、`results/paired_comparison_v1.json`
- 复现命令：
  - 多任务评估：`python src/evaluation/eval_multitask_full.py`（混淆矩阵版）
  - 生成器评估：`python src/evaluation/eval_generator.py`（PSNR/SSIM）
- 混淆矩阵与三模型图若需重出，可运行对应 eval 脚本后把产物放到 `results/`（不入库，随 Release 或附录附上）。
