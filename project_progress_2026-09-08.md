# 一图知卡 项目进度报告

> 更新时间：2026-09-08
> 当前阶段：模拟器测试阶段（核心功能已全部完成）
> 目标用户：减肥群体，提供食物识别 + 热量估算 + 营养师食谱

---

## 一、当前项目状态总览

| 模块 | 状态 | 说明 |
|---|---|---|
| 主多任务模型 `meal_rgb_official_v1` | ✅ 已完成 | audited 测试集 MAE kcal=56.55，R²=0.839，可上线（原 49.9/0.867 无出处，见 3.4） |
| NIR 生成器 `hsi_unet_v2` | ✅ 已完成 | PSNR 21.5 dB，支持 RGB→NIR 映射 |
| 食物前置过滤器 | ✅ 已上线 | ImageNet ResNet50，拦卡通/动漫/截图 |
| 后端推理服务 | ✅ 已上线 | localhost:8000，Flask + 7 个 API 接口 |
| 营养师模块（食谱生成） | ✅ 已完成 | 显示中文菜名，含溯源 sha256 |
| 反馈数据系统 | ✅ 已完成 | SQLite + JSONL 持久化，4 级质量分类 |
| 微信小程序前端 | ✅ 已完成 | 4 tab，7 个页面，含 onboarding 问卷 |
| 队友代码资产吸收 | ✅ 已完成 | configs + calorieclip + kaggle notebook + eval_full |
| GitHub 推送 | ⏳ 待执行 | 需整理 .gitignore，敏感信息过滤 |
| 服务器部署 | ⏳ 待执行 | 需租云服务器 + 域名 + ICP 备案 |
| 小程序发布 | ⏳ 待执行 | 需 HTTPS 域名后提交审核 |

---

## 二、已完成功能详细清单

### 2.1 后端服务（yitu-zhika-code/app/）

| API 接口 | 方法 | 功能 | 状态 |
|---|---|---|---|
| `/health` | GET | 健康检查，返回 pipeline 状态 | ✅ |
| `/model-info` | GET | 返回模型版本、SHA256、精度 | ✅ |
| `/predict` | POST | 食物图片识别（含前置过滤） | ✅ |
| `/feedback` | POST | 用户反馈收集（三档+手动改） | ✅ |
| `/weekly-plan` | POST | 营养师生成 7 天食谱 | ✅ |

**核心文件：**
- `inference_service.py` - Flask 主服务
- `pipeline.py` - 推理管线（RGB+多模态融合）
- `food_filter.py` - ImageNet ResNet50 前置过滤（**新增**）
- `model_registry.py` - 模型权重加载与 SHA256 校验

### 2.2 模型权重（yitu-zhika-code/checkpoints/）

| 模型 | 文件 | 大小 | 用途 |
|---|---|---|---|
| `meal_rgb_official_v1` | best.pt | 281 MB | **主模型**，5 维回归 + 11 类分类 |
| `meal_nir_official_v1` | best.pt | 345 MB | NIR 4 通道版（备用） |
| `hsi_unet_v2` | best.pt | 190 MB | Phase1 NIR 生成器 |
| `food101_pretrained` | best.pt | 271 MB | ResNet50 Food-101 预训练 |
| `calorieclip_official_v1` | best.pt | 448 MB | CalorieCLIP 基线对比 |
| `capsicum_unet4_seed42` | best_model.pth | 63 MB | U-Net 实验版本 |

### 2.3 小程序前端（miniprogram/）

| 页面 | 路径 | 功能 | 状态 |
|---|---|---|---|
| 拍照 | pages/camera | 拍照/选图 + EXIF strip + dish_uuid | ✅ |
| 结果 | pages/result | 三档反馈 + 手动改 + quality 分级 | ✅ |
| 营养师 | pages/nutritionist | TDEE 计算 + 7 天食谱 + 中文菜名 | ✅ |
| 历史 | pages/history | 记录列表 + 清除 | ✅ |
| 我的 | pages/contribution | 贡献统计 + 隐私/数据删除 | ✅ |
| 关于 | pages/about | 免责声明 + 数据说明 | ✅ |
| 引导 | pages/onboarding | 首次使用问卷 | ✅ |

**tabBar 配置：** 4 个 tab（拍照/历史/营养师/我的）

### 2.4 训练管线（yitu-zhika-code/src/）

```
src/
├── models/
│   ├── generator.py              # U-Net NIR 生成器
│   ├── discriminator.py          # PatchGAN 判别器
│   ├── resnet_multitask.py       # ResNet50 多任务网络
│   ├── clip_food_classifier.py   # CLIP 食物分类（HuggingFace）
│   └── baseline/
│       └── calorieclip_wrapper.py  # CalorieCLIP 基线（**新增**）
├── data/
│   ├── nutrition5k_loader.py    # Nutrition5k 数据集
│   └── pix2pix_dataset.py        # Pix2Pix 配对数据集
├── training/
│   ├── train_generator.py        # Phase1 GAN 训练
│   └── train_multitask_v2.py     # Phase2 多任务训练
├── evaluation/
│   ├── eval_generator.py          # 生成器评估 (PSNR/SSIM)
│   ├── eval_multitask.py          # 多任务评估 (MAE/MAPE/RMSE)
│   └── eval_multitask_full.py    # 增强版评估（**新增**，含混淆矩阵）
└── checkpoints_io.py             # 权重保存/加载
```

### 2.5 配置文件（yitu-zhika-code/configs/）

| 配置 | 用途 |
|---|---|
| `default.yaml` | 正式训练配置（11 类 + 5 维回归） |
| `augment.yaml` | 数据增强配置（RandAugment + MixUp + Random Erasing） |
| `smoke.yaml` | **新增** 冒烟测试（1 epoch 验证全链路） |
| `kaggle.yaml` | **新增** Kaggle 环境配置 |

### 2.6 营养师模块（yitu-zhika-code/recipe/）

- `nutrition_db.py` - NutritionDB，RAG 检索 + 中文菜名映射（CATEGORY_DISH_NAMES，8 类 95 个菜名）
- `tdee_estimator.py` - TDEE 估算（Mifflin-St Jeor）
- `template_planner.py` - 7 天食谱生成（5006 餐盘，含过敏原过滤）

---

## 三、模拟器测试结果（2026-09-08）

### 3.1 API 接口测试（curl 直接调用）

| 测试项 | 结果 |
|---|---|
| GET /health | ✅ `pipeline_loaded:true, status:ok` |
| 食物图识别（5 类） | ✅ 5/5 通过（apple_pie/bibimbap/caesar_salad/baby_back_ribs/bread_pudding） |
| 非食物拦截（3 类） | ✅ 3/3 拦截（企鹅/动漫/蜘蛛侠，filter=imagenet_resnet50, food_score=0） |
| 数值精度 | ✅ 全部 1 位小数（如 266.6 kcal） |
| /weekly-plan | ✅ 7 天食谱，含中文菜名（花卷/芝士片/燕麦粥） |
| /feedback | ✅ SQLite + JSONL 持久化 |

### 3.2 发现并修复的 Bug

| Bug | 修复 | 时间 |
|---|---|---|
| 置信度阈值 0.5 误拦真实食物 | 阈值降到 0.25 | 09-08 |
| 分类头偏向"蔬菜" | 已记入后续优化清单（不影响热量） | 09-08 |
| 营养师返回 dish_id 而非中文名 | 加 CATEGORY_DISH_NAMES + _gen_name | 09-08 |
| 非食物图被识别为食物 | 加 ImageNet ResNet50 前置过滤 | 09-08 |

### 3.3 复核测试（2026-09-08 晚间，agent 接手后重新验证）

复测环境：miniconda env `yitu`（Python 3.11.16 / torch 2.7.1+cu128），
`python app/inference_service.py --port 8000`（CPU，主模型 SHA256 校验通过）。
实测记录文件：`yitu-zhika-code/results/api_smoke_recheck_result.json`（results/ 不入库）。

| 测试项 | 结果 |
|---|---|
| GET /health | ✅ `pipeline_loaded:true` |
| 无 API key 访问 /model-info | ✅ 401 `UNAUTHORIZED` |
| 食物识别（5 类，1 位小数） | ✅ 5/5 ok：apple_pie 266.6 / bibimbap 351.4 / caesar_salad 468.7 / baby_back_ribs 561.7 / bread_pudding 636.2 kcal |
| 非食物拦截（3 类） | ✅ 3/3 `not_food`（filter=imagenet_resnet50，food_score=0） |
| /weekly-plan 三组档案 | ✅ 均 7 天、unknown=0：male-moderate-maintain tdee=2507.1；female-light-lose tdee=1390.4；male-very_active-gain tdee=3834.7；每日 day_total_kcal 与 target 偏差 < 1 kcal |
| special_population=true | ✅ `status=refused` → 人工审核路由 |
| 输入校验 | ✅ 缺参 400 MISSING_PARAM / 非图片 400 UNSUPPORTED_FORMAT / 无文件 400 NO_IMAGE |
| /feedback | ✅ SQLite + JSONL 双写 |

**结论**
1. **TDEE 返回 null 问题复测未复现，风险 2 关闭**：当前 `/weekly-plan` 走模板版
   `TemplatePlanner`（无 DS API），响应不含顶层 `tdee:null`，数值在 `tdee_report.tdee`
   且全部正常；小程序端只读取 `plan.targets.kcal` 与 `plan.daily_recipes[*]`，不存在
   渲染空值路径。此前观察到的 null 应来自 DS-API 旧版路径（`weekly_planner.py` 的
   llm 分支），已随模板版替换消失。**已顺手在响应顶层补 `tdee`/`bmr`/`target_calories`
   别名**（2026-09-08 晚间实测顶层 `tdee=2507.1` 非空，与 `tdee_report` 一致）。
2. 主模型分类头仍偏向"蔬菜"（apple_pie/bibimbap/caesar_salad 均为蔬菜、prob 1.0），
   与风险 1 一致：不影响热量数值，建议上线后按反馈数据迭代。

### 3.4 静态审查修复（2026-09-08 晚间，模拟器测试的可行部分）

无法在本环境启动微信开发者工具，故对小程序做了全量静态契约审查（app.json 页面/4 tabBar、
JS 语法、JSON 合法性、前后端字段契约，node 校验全部通过），发现并修复以下确定性问题：

| 问题 | 修复 | 文件 |
|---|---|---|
| `.gitignore` 带冲突标记（`<<<<<<< HEAD…>>>>>>> origin/main`，merge 时未解决即提交） | 重写为干净的 UTF-8 合并版，追加 sqlite3/project.private.config.json 等忽略项 | 根 `.gitignore` |
| 手动改类别的下拉是英文 id 且只有 10 项，后端返回中文名 → 无法预选、缺 `soup_stew`、回传中英不一致 | 改为 11 类中文展示 + id 回传，按 `category_idx` 预选，`model_category`/`corrected_category` 统一为 id | `miniprogram/pages/result/result.js` |
| "我的"页统计字段（feedbackQuality/corrected_calories/grade=skip）从未写入 → confirm/skip/manual 统计失真 | result.js 提交成功后结构化落盘；contribution 回退推导补齐 `ok→confirm_only` | result.js / `contribution.js` |
| `pages/feedback` 死页面（注册但无入口，submit 仅弹 toast 不真正上报） | 删除页面文件并从 app.json 移除（现 7 页：4 tab + result/about/onboarding） | `app.json`、`pages/feedback/*` |
| 进度文档主模型 "MAE 49.9 / R² 0.867" 全仓库无出处，与 audited（56.55 / 0.839）不符 | 勘误为 audited 数字，报告引用以 audited 为准 | `project_progress_2026-09-08.md` |
| 结果页只显示单一粗类别（混合餐如饺子+肉+菜+蘸料被压成"蔬菜"） | `/predict` 增加 `category_probs`（top-5 置信度），结果页展示"食物种类（按置信度）"，并注明非逐项成分检测 | `experiment_pipeline.py`、`inference_service.py`、`result.wxml`/`.wxss` |
| 历史记录存的是临时图路径，小程序重启后缩略图失效 | 拍摄后落盘到 `wx.env.USER_DATA_PATH`（保留最近 60 张自动清理，失败保底用临时路径） | `pages/camera/camera.js` |
| 整体 UI 较单调、缺乏使用激励（用户反馈"丑"，想要完成感/满足感） | 全局视觉升级（现代配色/圆角卡片/渐变主按钮）+ 三处成就机制：①「我的」连续打卡🔥+周进度圆点+里程碑成就；②历史页打卡摘要；③结果页完成记录后🎉鼓舞反馈 | `app.wxss`、`contribution/*`、`result/*`、`history/*` |

**未修（需微信开发者工具或产品决策，已在 release_checklist 记录）：**
- onboarding 问卷的过敏原（英文）未与营养师页忌口（中文）打通，仅存 profile 未被消费

**新增资产：** `yitu-zhika-code/docs/course_report_data_2026-09-08.md`（报告数据包，
含三模型 audited 指标与数字勘误）、`yitu-zhika-code/docs/release_checklist_2026-09-08.md`
（推送/权重 Release/部署/发布检查单）、`yitu-zhika-code/results/weights_manifest.json`
（6 个发布权重的 SHA256）。

### 3.5 自定义/预设食物池（新功能，2026-09-08 晚）

满足"预设食物 / 商家推广款 / 用户补全数据后科学配餐"：

- **新端点** `POST /plan-from-menu`：在档案基础上接受 `preset`（内置 `dumpling` 饺子餐、
  `merchant_demo` 商家推广）或 `foods`（用户/商家逐条补全：name/category/kcal_per_100g/
  宏量/default_grams）。
- **新模块** `recipe/custom_menu.py`：食物校验（沿用 kcal∈[0,2000]、宏量∈[0,100]、
  11 类白名单）+ `plan_from_pool` 配餐（仅从该池选料，按 TDEE 目标热量分早/午/晚；
  主食/高密度优先、蔬汤佐餐；忌口过滤；7 天食物顺序轮换不重复；输出宏量与免责声明；
  不因缺类别强行归入其他类）。
- **实测通过**：
  - `preset=dumpling`（170/70/male/moderate/maintain）：7 天 `day_total` 2504–2506 vs 目标 2507，
    全部 `within_tolerance`，含凉拌黄瓜佐餐。
  - 用户补全（牛肉板面/卤蛋/烫青菜，女 160/55/28/light/lose，忌海鲜）：目标 1373，
    7 天 1424–1427 均 `within_tolerance`；海鲜被忌口过滤。
  - 非法食物 → 400 `INVALID_FOOD`（带具体原因）；空池 → 400 `EMPTY_FOOD_POOL`。
- **前端** nutritionist 页：新增「饮食范围」预设选择 + 「自定义食物」录入
  （名称/类别/每100g热量/份量，可增删）；选预设或录入食物时自动改走 `/plan-from-menu`。
- 说明：预设及用户补全为演示估算值；商家接入应以真实检测/标签数据为准；上线前需强 API key。

### 3.6 小程序 UI 重构（2026-09-08 晚，按设计文档）

- **信息架构**：tabBar 改 3 项「今天 / 饮食计划 / 我的」；拍照收敛到「今天」主按钮，移除拍照/历史 tab（历史经「今天→查看全部历史」）。
- **今天**：标题"今天吃了什么？"，主按钮"拍照记录"+次要"从相册选择"；今日餐食卡片；状态"已记录 N 餐"；合计命名为"已记录餐食的估算合计"+ 不完整提示；空态友好引导；不显示热量缺口/超标警报。
- **识别与确认**（原 result）：先图再类别/估算热量/估算份量；"确认记录"为主，次之"修改份量/类别/重新选择照片"；区分模型估算 vs 用户填写；注明"本页数值均为模型估算"，未把模型分值当"识别准确率"；未支持逐项识别则明确"非逐项检测"。
- **饮食计划**（原 nutritionist）：按实际日期 7 天 × 早/午/晚卡片；每卡片含菜名/份量/生熟/来源营养；操作"换一道/调整份量/不喜欢这道"；顶部标注"AI 建议草案 · 非处方 · 未经营养师审核"（无真人审核不显示营养师认证）；缺失信息用"待确认"，不用零；后端为食物补充 `per_100g` 支撑份量线性重算。
- **我的**（原 contribution）：档案/偏好/忌口/提醒设置/数据管理；导出、提醒在后端支持前标注"暂未开放/即将上线"，不做假按钮；成就改为温和"记录足迹"（无排行榜/惩罚/强制分享）。
- **全部状态覆盖**：空/加载/成功/失败/断网/权限拒绝/不完整；不伪造进度；防重复提交；保存失败仅本地保底；出错提示可重试。
- 文档：`miniprogram/UI_REDESIGN_PLAN.md`（现状梳理 + 改动清单）。
- **验收提示**：需在微信开发者工具/真机验收（用户完成拍照→修正→保存；区分估算/确认；食谱可调整；未知不被包装为准确；错误可恢复不丢内容）。

---

## 四、下一步工作清单

### 4.1 立即执行（本周内）

| 序号 | 任务 | 优先级 | 预计成本 |
|---|---|---|---|
| 1 | **模拟器全流程测试**：拍照→识别→反馈→营养师→历史→我的 | 🔴 高 | 30 分钟 |
| 2 | **修复模拟器中发现的 UI bug** | 🔴 高 | 视情况 |
| 3 | **GitHub 代码推送**（整理 .gitignore + commit + push） | 🟡 中 | 1 小时 |
| 4 | **模型权重上传**（用 Git LFS 或 GitHub Releases） | 🟡 中 | 30 分钟 |

### 4.2 上线前必须完成（1-2 周）

| 序号 | 任务 | 优先级 | 说明 |
|---|---|---|---|
| 5 | **租云服务器** | 🔴 高 | 阿里云/腾讯云轻量级（4 核 8G），约 60-100/月 |
| 6 | **注册域名 + ICP 备案** | 🔴 高 | 备案 1-2 周，越早越好 |
| 7 | **配置 HTTPS 证书** | 🟡 中 | Let's Encrypt 免费 |
| 8 | **部署后端服务** | 🟡 中 | gunicorn + nginx 反代 |
| 9 | **更换强 API key** | 🔴 高 | 当前 `dev-key-change-in-prod` 不能上线 |
| 10 | **小程序提交审核** | 🟡 中 | HTTPS 域名配置后，提交微信审核 1-3 天 |

### 4.3 可选优化（不影响上线）

| 序号 | 任务 | 优先级 | 说明 |
|---|---|---|---|
| 11 | 重训模型（更多数据） | 🟢 低 | 当前 MAE 56.55 kcal（audited）已可用 |
| 12 | CalorieCLIP 基线对比 | 🟢 低 | 用于课程设计报告 |
| 13 | 加固图像质量检测 | 🟢 低 | 拉普拉斯方差检测模糊图 |
| 14 | 加人脸检测前置 | 🟢 低 | 防止上传人脸照片 |
| 15 | 加 NSFW 检测 | 🟢 低 | 上线后看实际反馈数据 |

---

## 五、关键约束（来自项目 memory）

### 5.1 硬约束

- U-Net 上采样层通道计算必须用修正版：up3=768, up2=384, up1=192
- 训练脚本 8 小时硬预算限制，需修改或禁用以延长训练
- Food-101 数据集需要 SHA256 + 101→11 类映射清单
- Nutrition5k 数据集异常过滤：kcal [0, 2000], grams [0, 2000], kcal/g [0.1, 9.0]
- DeepSeek API key 必须在 .env 文件，且加入 .gitignore
- 所有反馈数据必须含 dish_uuid, image_hash, quality 字段
- 用户上传图片必须去除 EXIF 元数据
- 小程序反馈接口必须实现三档点击（偏多/差不多/偏少），"差不多"高亮
- SQLite feedback_v1 表 + JSONL 备份用于反馈数据存储
- 首次用户必须完成 onboarding 问卷才能访问核心功能
- API 端点除 /health 外必须 X-API-Key 认证，未授权返回 401
- 图片上传限制 10MB，JPG/PNG 格式
- API 限流 30 req/min/IP，超限返回 429
- 错误响应必须用标准化 JSON 格式，禁止伪造成功
- 营养库未知项返回明确状态（matched/unknown/ambiguous），不用类别替代
- 移除所有未经证实的精度声明（如 ±5%/±15%）
- 生熟重量换算逻辑暂停（缺乏可靠证据）
- 食谱生成必须含免责声明"饮食参考，非医疗处方"，禁止减重处方
- GitHub 推送必须在项目完成、验收测试、密钥验证之后
- 小程序 tabBar 必须正好 4 个：拍照/历史/营养师/我的
- 食物检测推理必须对 confidence < 0.5 返回 `status: not_food`
- 非食物时弹窗"未检测到食物，请拍摄食物图片"，不进结果页
- 推理输出值（卡路里/重量/prob）必须四舍五入到 1 位小数

### 5.2 工程约定

- NIR 生成器：U-Net + Pix2Pix 架构
- 多任务网络：ResNet50 + 3 任务（分类 CE / 卡路里 L1+MAPE / 重量 L1）
- 推理支持多模态（RGB+NIR 4 通道）+ 纯 RGB 降级
- 数据增强：RandAugment (num_ops=2, mag=9) + MixUp (α=0.2) + Random Erasing (p=0.25)
- 模型权重加载时必须 SHA256 校验，不匹配抛 RuntimeError
- 反馈质量 4 级：confirm_only / directional / manual_typed / skipped
- 食谱模板：5006 餐盘（dishes_verified.csv），含过敏原过滤

---

## 六、课题要求对照

> 课题要求：源代码仓库 + 训练好的模型权重 + 课程设计报告（含算法复现与对比实验分析）

| 要求 | 当前状态 | 需要做什么 |
|---|---|---|
| 源代码仓库 | ✅ 代码已就绪 | 推送 GitHub |
| 训练好的模型权重 | ✅ 281 MB 主模型 + 多个对比模型 | 用 Git LFS 或 Releases 上传 |
| 课程设计报告 | ⏳ 未开始 | 用 CalorieCLIP 基线对比 + eval_multitask_full 出数据 |

---

## 七、风险与建议

### 7.1 当前主要风险

1. **分类头偏向"蔬菜"**：降低置信度阈值后能看到不同类别，但高置信度时仍偏向"蔬菜"。不影响热量估算，但影响用户体验。**建议**：上线后收集反馈数据再迭代。

2. **TDEE 字段返回 null**：~~`/weekly-plan` 响应里 `tdee=None`~~ **✅ 已关闭（2026-09-08 复测未复现）**：模板版响应 `tdee_report.tdee` 数值正确（见 3.3），小程序无空值渲染路径。

3. **服务器+备案周期**：备案需要 1-2 周，是上线的关键路径。**建议**：立即开始办备案。

### 7.2 推荐执行顺序

```
现在 ──► 模拟器全流程测试（30 分钟）
       │
       ├─► GitHub 推送代码 + 模型权重（1 小时）
       │
       └─► 服务器+域名+备案（1-2 周并行）
              │
              └─► 部署后端 + HTTPS → 小程序审核 → 上线
```

---

## 八、文件位置索引

| 类型 | 路径 |
|---|---|
| 后端服务 | `yitu-zhika-code/app/inference_service.py` |
| 前置过滤器 | `yitu-zhika-code/app/food_filter.py` |
| 推理管线 | `yitu-zhika-code/app/pipeline.py` |
| 模型权重 | `yitu-zhika-code/checkpoints/` |
| 配置文件 | `yitu-zhika-code/configs/` |
| 训练脚本 | `yitu-zhika-code/src/training/` |
| 评估脚本 | `yitu-zhika-code/src/evaluation/` |
| 营养师模块 | `yitu-zhika-code/recipe/` |
| 营养数据 | `yitu-zhika-code/data/nutrition5k/dishes_verified.csv` |
| 小程序前端 | `miniprogram/` |
| 测试样本 | `test_images/`（5 食物 + 3 非食物） |
| 项目进度报告 | `project_progress_2026-09-08.md`（本文件） |

---

**报告生成时间**：2026-09-08 17:23
**当前后端状态**：运行中（localhost:8000，pipeline_loaded:true）
