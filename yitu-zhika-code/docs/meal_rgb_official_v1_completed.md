# RGB 餐食对照：完成与独立核验

完成时间：2026-09-04 23:21:56（Asia/Shanghai）。30 轮全部完成，验证集选定第 28 轮为最佳模型；没有因早停提前终止。status.json 中最终 epoch=28 是测试所用模型轮次，完整训练轮次以 epochs.json/last.pt 的 30 为准。

## 测试结果

使用官方 RGB test IDs 的本地 overhead 子集，共 507 个餐盘 ID；并非官方完整多视角测试集。训练/验证分别为 2188/567，且验证集仅从官方训练 IDs 内按拍摄日代理分组产生。

| 指标 | 热量（kcal） | 质量（g） |
| --- | ---: | ---: |
| MAE | 56.5491 | 36.6963 |
| RMSE | 85.1264 | 53.5051 |
| R² | 0.8388 | 0.8753 |
| 非零目标 MAPE | 48.0671% | 41.3753% |
| MAPE 分母样本数 | 506 | 507 |
| MAE / 目标均值 | 22.1356% | 18.4860% |
| 负值预测个数 | 4 | 3 |

测试集中 1 个零热量目标仍参与 MAE/RMSE/R²，仅从需要除以目标的 MAPE 中排除。两种相对误差不是同一指标，不能混报。预测未截断为零来改善指标；少量负预测说明模型仍需后续物理约束与独立验证，当前未接入 Demo。

派生粗类别准确率 72.7811%。这些是按食材质量和关键词产生的 11 组粗标签，不是官方食物类别，且类别分布很不均衡；不能声称完成了 61 类官方食物识别。

## 核验结果

`scripts/audit_meal_run.py` 已执行并通过，输出 `results/meal_rgb_official_v1/completion_audit.json`。核查内容：

- best/last 能以 CPU、weights_only=True 回读，权重均为有限数；last.training_complete=True，历史完整包含 30 轮。
- 最佳轮次与全部历史验证 normalized regression L1 的最小值一致；没有依赖测试误差重新挑选模型。
- 检查点、训练 protocol、冻结 manifest 的目标次序与单位一致；训练标签统计只使用训练集。
- 预测 CSV 恰好包含测试清单的 507 个不重复 ID；真实标签与冻结清单经训练用 float32 转换后的值完全一致。
- 逐餐盘 CSV 独立重算 MAE/RMSE/R²/MAPE/均值相对误差/负预测计数与结果 JSON 一致。
- D 盘试点清单哈希一致；4 个新增餐盘/12 帧未混入当前冻结实验。
- 最终 error.log 为空，训练进程已退出，C 盘剩余约 62.73 GiB，高于 35 GiB 保留阈值。

| 文件 | SHA256 |
| --- | --- |
| checkpoints/meal_rgb_official_v1/best.pt | `3f189a439fca3b79ecd820ae68605bcbfe61679def14f8dfa548c34261c89cf0` |
| results/meal_official_v1/manifest.json | `f1e293b826c98e517c3d1179871ec0a10d8ee471652b5d2e6f89f9ed1c755daa` |
| results/meal_rgb_official_v1/test_predictions.csv | `75c7d29b2fffeaf240136c086ff63de8ee91bef6be2657532bae5d61f2fdb7bf` |

这是保存产物的一致性检查，不是对原始餐盘独立性或统计显著性的证明。没有再次跑模型推理、重新训练或根据测试结果调参。

## 结论和下一阶段

已得到有固定划分、正确物理单位、可回读权重和可追溯预测的内部 RGB 对照。不能称为 CalorieCLIP/Oatsty 外部方法复现；也不能与旧随机验证划分的约 38.55 kcal MAE 直接比较优劣，因为数据划分与实验协议不同。

后续仍需审计/重训来源明确的 RGB→NIR 生成器，再做同划分同预算的 RGB+预测 NIR 对照；另外补齐外部 RGB 方法复现。扩容视频应在独立新版本清单中统一视角、餐盘分组及评估协议，不能把多个视频帧当作多个独立餐盘。

本轮训练及扩容试点已完成，因此结束仅用于这轮 RGB 训练的自动监控。项目整体仍未全部完成；本次未替换 Demo，也未启动下一轮训练。
