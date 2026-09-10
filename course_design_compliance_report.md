# 课程设计实现情况汇报

> 课题：以食物卡路里估计为核心，复现论文《Multi-Spectral Food Classification and Caloric Estimation Using Predicted Images》
> 汇报时间：2026-09-09
> 对照源：`project_progress_2026-09-08.md` + `yitu-zhika-code/docs/course_report_data_2026-09-08.md`

---

## 一、课题要求与实现对照

| # | 课题要求 | 实现状态 | 实现位置 / 证据 |
|---|---|---|---|
| 1 | 阶段一：HSIFoodIngr-64 上训练 RGB→NIR 生成网络（U-Net/Pix2Pix） | ✅ 已完成 | `src/models/generator.py`（U-Net + PatchGAN）、`src/training/train_generator.py`、权重 `checkpoints/hsi_unet_v2/best.pt` |
| 2 | 计算 PSNR / SSIM 评估生成质量 | ✅ 已完成 | `src/evaluation/eval_generator.py`；HSI 测试集 PSNR **23.33 dB** / SSIM **0.812** |
| 3 | 阶段二：Nutrition5k 上 RGB+预测NIR 拼接多通道 | ✅ 已完成 | `src/models/resnet_multitask.py`（4 通道输入版）、权重 `checkpoints/meal_nir_official_v1/best.pt` |
| 4 | ResNet 多任务：分类（CE）+ 卡路里/重量（L1/MAPE） | ✅ 已完成 | `src/models/resnet_multitask.py`（11 类 + 5 维回归头）、`src/training/train_multitask_v2.py` |
| 5 | 引入纯 RGB 基线对比（CalorieCLIP 等） | ✅ 已完成 | `src/models/baseline/calorieclip_wrapper.py`、权重 `checkpoints/calorieclip_official_v1/best.pt` |
| 6 | 消融实验（MAPE/RMSE） | ✅ 已完成 | 三模型同 507 测试集对比（见下表），含 bootstrap 区间分析 |
| 7 | 演示应用：上传图像→展示预测 NIR + 类别 + 卡路里 + 基线对比 | ✅ 已完成 | 微信小程序（4 tab）+ Flask 后端（`app/inference_service.py`） |
| 8 | 提交源代码仓库 | ✅ 已完成 | GitHub: `at3613200152-collab/-yitu-zhika`（含队友 pure-pix2pix 分支合并） |
| 9 | 提交训练好的模型权重 | ⚠️ 本地就绪，未上传 | 6 个权重共 ~1.6 GB（见 `results/weights_manifest.json`），待用 GitHub Releases 上传 |
| 10 | 课程设计报告（含算法复现 + 对比实验分析） | ⏳ 未开始 | 数据已备齐（见 `docs/course_report_data_2026-09-08.md`），可直接撰写 |

**结论**：10 项要求中 8 项已完成，1 项待上传权重，1 项（报告）待撰写。核心算法、对比实验、演示应用全部就绪。

---

## 二、三模型对比实验数据（同一 507 测试集）

| 模型 | 方案 | 热量 MAE | 热量 RMSE | R² | 非零 MAPE | 重量 MAE | 类别准确率 |
|---|---|---:|---:|---:|---:|---:|---:|
| `meal_rgb_official_v1` | 纯 RGB（内部对照） | **56.55** | 85.13 | 0.8388 | 48.07% | 36.70 | 72.78% |
| `meal_nir_official_v1` | RGB＋预测 NIR | **56.21** | 84.90 | 0.8397 | 51.02% | 35.64 | 73.18% |
| `calorieclip_official_v1` | CalorieCLIP 外部基线 | 58.74 | 91.28 | 0.8146 | 35.36% | 不支持 | 不支持 |

### 关键发现（可直接写入报告）

1. **多模态 vs 纯 RGB**：RGB+NIR 热量 MAE 比 RGB 低 0.34 kcal，但 64 拍摄日分组 10000 次 bootstrap 区间 **[-3.091, +2.503] kcal 跨过 0**，MAPE 反而变差。**结论：内部消融未观察到显著提升**，需如实表述。

2. **vs CalorieCLIP 基线**：我们的多任务模型热量 MAE 56.55 vs CalorieCLIP 58.74，略优 2.19 kcal；且我们额外支持重量估计和 11 类分类，CalorieCLIP 只输出热量。R² 也更高（0.839 vs 0.815）。

3. **分类头偏向"蔬菜"**：混淆矩阵显示高置信度时聚集到"蔬菜"类。不影响热量估计，但影响类别展示。

4. **NIR 生成器**：PSNR 23.33 dB / SSIM 0.812，生成的是**相对强度预测图**而非实测光谱。域迁移到餐盘照片仍有限制。

---

## 三、演示应用功能清单

| 模块 | 功能 | 实现位置 |
|---|---|---|
| 拍照/选图 | EXIF strip + dish_uuid 生成 | `miniprogram/pages/camera/` |
| 食物识别 | ImageNet 前置过滤 + 主模型推理 | `app/inference_service.py` `/predict` |
| 非食物拦截 | ImageNet ResNet50 food_score < 0.3 拒绝 | `app/food_filter.py` |
| 结果展示 | 类别 + 卡路里 + 重量 + 三档反馈 + 手动改 | `miniprogram/pages/result/` |
| 营养师 | TDEE 估算 + 7 天食谱（中文菜名 + 溯源） | `recipe/template_planner.py` `/weekly-plan` |
| 历史记录 | 本地存储 + 缩略图落盘 | `miniprogram/pages/history/` |
| 贡献统计 | 上传餐数 + 确认率 + 跳过率 + 手动修正 | `miniprogram/pages/contribution/` |
| 引导问卷 | 饮食目标 + 过敏原 + 偏好 | `miniprogram/pages/onboarding/` |
| 反馈系统 | SQLite + JSONL 双写，4 级质量分类 | `app/inference_service.py` `/feedback` |
| 商家菜单 | 商家上传菜品营养 + 公开菜单 + 配餐 | `app/merchant.py` |
| 内测采集 | 训练授权 + 标注分层 + 溯源 | `app/collection.py` |

---

## 四、待完成事项

| # | 事项 | 优先级 | 说明 |
|---|---|---|---|
| 1 | 模型权重上传 GitHub Releases | 🔴 高 | 6 个权重 ~1.6 GB，manifest 已就绪 |
| 2 | 课程设计报告撰写 | 🔴 高 | 数据全部备齐，可直接写 |
| 3 | 服务器部署 + 域名备案 | 🟡 中 | 上线用，非课设评审必需 |
| 4 | 小程序发布审核 | 🟡 中 | 需 HTTPS 域名 |

---

## 五、报告撰写建议

1. **算法复现章节**：引用 `course_report_data_2026-09-08.md` 的 audited 数字，不要用早期 49.9/0.867（无出处）。
2. **对比实验章节**：三模型表格 + 混淆矩阵图（`results/course_report_plots/`）+ bootstrap 区间分析。
3. **消融结论**：如实写"多模态未显著优于 RGB"，这是科学诚实的体现，比强行声称提升更有价值。
4. **演示应用章节**：贴小程序截图 + API 测试结果（`results/api_smoke_recheck_result.json`）。
5. **图表素材**：`results/course_report_plots/` 下有散点图、混淆矩阵、delta 直方图，可直接贴报告。
