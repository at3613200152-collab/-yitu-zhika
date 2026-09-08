# 一图知卡 小程序 UI 重构计划（现状梳理 + 改动清单）

> 依据需求文档：产品是微信小程序（不是 Gradio Demo / 网页仪表盘）。
> 原则：温和、可信、有掌控感；不评判；明确估算/手动/核验边界；不做假按钮。

## 一、现状梳理

### 1.1 页面（app.json 注册 7 页，tabBar 4 项）
| 现有页 | 现 tab | 角色 |
|---|---|---|
| pages/camera | ✅拍照 | 拍照/相册→上传→跳结果 |
| pages/result | - | 展示估算 + 三档反馈 + 手动改 |
| pages/history | ✅历史 | 本地记录列表 + 清空 |
| pages/nutritionist | ✅营养师 | TDEE + 生成 7 天食谱（模板/预设/自定义池） |
| pages/contribution | ✅我的 | 贡献统计 + 成就 + 清除数据 |
| pages/onboarding | - | 首启问卷 |
| pages/about | - | 免责/隐私 |

### 1.2 接口（后端，均可用）
`GET /health` · `GET /model-info` · `POST /predict`（图→类别+热量+重量+category_probs，含前置过滤）·
`POST /feedback`（三档+手动+quality 分级）· `POST /weekly-plan`（7 天餐）· `POST /plan-from-menu`（食物池）。

### 1.3 关键数据字段（predict）
`status, category_name(中文), category_prob, category_probs[{id,name,prob,pct}], calories, weight,
model_version, model_sha256_prefix`；（非食物 `status:not_food + message`）。
weekly-plan：`daily_recipes[{day,date,meals[{meal_name,foods[{name,grams,cooked_or_raw,nutrition{calories,protein,carb,fat},source,status,substitutions}],nutrition{calories}}],day_total_kcal,target_kcal,deviation_kcal,within_tolerance]`, `targets, tdee_report, tdee, bmr, target_calories, disclaimer, generator`。

### 1.4 现状问题（相对需求）
- tabBar 4 项与「今天/饮食计划/我的」3 个主入口不符；拍照是独立 tab，无「今天」聚合页。
- 结果页把分值当"识别准确率"展示？——当前展示为"模型分数"，需改为明确标注"估算"。
- 无"确认记录"主按钮与"修改份量/类别/重选"分层；反馈后即返回，缺"已确认保存"心智。
- 饮食计划无"按实际日期"组织、无"换一道/调整份量/不喜欢"，未区分 AI 草案 vs 真人审核（本产品无真人审核，需明确标注"AI 建议草案"）。
- 「我的」无档案/偏好/忌口/提醒/数据管理；成就项其实鼓励打卡，需改温和（无惩罚/排行榜）。
- 多处状态未完整：加载/失败/断网/权限拒绝/不完整。

## 二、改动清单

### 信息架构
- tabBar 改 3 项：**今天**（pages/today）｜**饮食计划**（pages/nutritionist，改标题）｜**我的**（pages/contribution，改标题）。
- 移除拍照/历史 两个 tab；拍照动作收敛到「今天」主按钮；历史经「今天」入口 /「我的·数据管理」可达；camera、history、about、onboarding 保留为非 tab 页。

### 新增/复用
- 新增 `utils/record.js`：拍照/相册 → EXIF 剥离+落盘 → 上传 → 存历史 → 跳「识别与确认」；today 与 result 的"重新选择照片"共用。
- 新增 `pages/today`：标题"今天吃了什么？"，主按钮"拍照记录"，次要"从相册选择"，今日餐食卡片，状态"已记录 N 餐"，估算合计命名为"已记录餐食的估算合计"，不完整提示"记录可能不完整"，空态友好引导。
- 改 `pages/result` → 识别与确认：先图，再类别/估算热量/估算份量；"确认记录"为主，"修改份量/修改类别/重新选择照片"为次；区分 模型估算/用户填/核验数据；不称分值=准确率；确认前不记为真实摄入。
- 改 `pages/nutritionist` → 饮食计划：实际日期 7 天 × 早/午/晚卡片；菜名+份量+生熟+来源营养；操作"换一道/调整份量/不喜欢这道"；信息缺失用"未知/待确认"，不用零；无真人审核则不显示营养师认证；注明"AI 建议草案，非处方"。
- 改 `pages/contribution` → 我的：档案/偏好/忌口/提醒设置/数据管理；导出与提醒在后端支持前明确"暂未开放"，不做假按钮；成就改为温和记录激励。
- 后端（兼容）：为 weekly-plan 的食物条目补充 `per_100g` 营养（kcal/protein/carb/fat），支撑"调整份量"线性重算；`/predict` 不变。

### 文案/边界
- 使用推荐话术（"已记录这一餐""你可以调整份量"）；避免"超标/失败/必须运动"等话术。
- 状态覆盖：空/加载/成功/失败/超时/断网/权限拒绝/不完整；不伪造进度；防重复提交；出错保留已填内容。

### 范围说明（本轮交付）
- 本轮：app.json 3 tab + pages/today + result/nutritionist/contribution 三页重构 + utils/record + 后端 per100g。
- 未实现（明确标注）：提醒推送、账号导出、真人营养师审核、按餐逐项识别（当前仅餐盘总量+粗类别）。
- 测试：JS/JSON/绑定 node 校验；后端接口兼容 curl 复测；真机状态需用户在微信开发者工具/真机验收。
