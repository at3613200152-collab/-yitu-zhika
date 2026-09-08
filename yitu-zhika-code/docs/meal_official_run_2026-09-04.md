# 新餐食对照训练与 D 盘扩容试点

记录时间：2026-09-04 23:05（Asia/Shanghai），静态记录，不代表实时进度。

## 已落地

- `src/data/meal_manifest.py` 冻结官方 RGB train/test ID 的本地 overhead 子集；官方 train 内按 UTC 拍摄日的确定性哈希另划验证集。训练 2188、验证 567、测试 507。按拍摄日分组是保守代理，不等于已确认所有物理餐盘独立。
- 清单 `results/meal_official_v1/manifest.json` SHA256：`f1e293b826c98e517c3d1179871ec0a10d8ee471652b5d2e6f89f9ed1c755daa`。逐图文件哈希和跨集合精确像素去重通过，标签由已审计的官方 metadata 提供。
- `src/training/train_meal_official.py` 使用 ImageNet V2 初始化 ResNet50，独立的热量/质量回归与 11 个派生粗类别辅助分类，不加载历史餐食回归权重。RGB 对照不是 CalorieCLIP/Oatsty 的外部基线复现。
- 标签次序固定 `[calories, mass]`，单位 kcal/g；训练集统计标准化，前向输出恢复物理单位；损失为标准差缩放 L1 加 0.2 CE。类别并非官方食物类别，存在显著不均衡与未覆盖类别。
- BF16、有限输入/损失/梯度检查、梯度裁剪；2 批真实数据预检后重新初始化模型和优化器，避免冒烟更新污染正式训练。
- batch 8，最多 30 轮，验证集连续 8 轮未改善早停；8 小时运行预算在轮次边界检查。每轮保存可恢复 last 和验证最优 best；测试仅在训练终止后执行，不用测试挑选 epoch。单随机种子，不能声称统计显著改善。
- 新文件独立存放于 `checkpoints/meal_rgb_official_v1` 与 `results/meal_rgb_official_v1`，未替换 Demo。检查点封装与旧 Demo 不兼容是刻意设计，接入前需显式适配和验证。
- 2026-09-04 23:03:38 隐藏窗口启动，初始 PID 27268；23:04 已进入第二轮并保存第一轮 best/last（各约 295 MB）。实时状态以 status.json 与真实进程命令行为准。
- `python -m unittest discover -s tests -v`：29 项测试通过。并非所有训练恢复分支都已有故障注入测试。

## D 盘试点

用户已允许 D 盘下载。`scripts/nutrition_video_pilot.py` 已完成 4 个原先没有本地 overhead 图像的官方训练餐盘侧视视频下载。固定云对象 generation、大小与 MD5，续传验证后用 OpenCV 解码每段的第 0/30/60 帧，总共 12 帧。文件保留在 `D:\yitu-data\Nutrition5k\side_angle_pilot_v1`。

清单 SHA256：`048bbe453771c69de3884acb486795337121975eabd17df9602138833b49be82`。项目内指针为 `results/video_expansion_pilot.json`。

这是 4 个新增餐盘 ID，不是 12 个独立餐盘；近似重复/增量扫描关系尚需扩大前审计。试点未加入当前冻结对照实验，不改变本轮训练/测试分母。下一版扩容需把同一餐盘所有视角与帧放在同一集合，并统一训练/评估视角协议。

## 后续边界

1. 等当前 RGB 对照完成，核验模型、测试清单、逐餐盘预测与指标。
2. RGB+预测 NIR 需要同划分同预算；旧生成器的训练数据来源未被哈希绑定，先审计或从明确数据重新训练，不能直接称无泄漏严格对照。
3. 外部 RGB 方法的许可、固定版本、公开权重的训练污染审计见 `rgb_external_baseline_audit.md`。research 技能促使本轮单独记录这些复现缺口，而没有把内部消融冒充外部复现。
4. 用户不需要手动运行命令。查看当前进度可读取 `results/meal_rgb_official_v1/console.log`；建立该结果目录下的 `STOP` 文件会在下一批次安全退出，保留上一个完整轮次的检查点。
