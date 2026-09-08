# 发布与推送检查单（2026-09-08 更新）

> 本文件汇总"上线前必须完成"的可执行步骤与已就绪资产。所有需要凭据/账号的步骤
> 均标注 [人工]，其余可在本机直接执行。

## 0. 当前状态速览
- ✅ 本地 main = 64447c0（已 merge origin/main，ahead 2，push 为快进，无需 force）
- ✅ 后端 API 复测 13 项全通过（`results/api_smoke_recheck_result.json`）
- ✅ 权重 SHA256 manifest：`results/weights_manifest.json`（6 个发布文件）
- ✅ .gitignore 已修复（合并残留的冲突标记已清除，UTF-8）
- ⚠️ 仓库含重复目录（旧扁平 teammate 布局 + `food_calorie_estimation/`），推送前需决策（见 §2）

## 1. 前提（本环境不可做，需在你本机）[人工]
1. 安装 GitHub CLI：`winget install GitHub.cli`，然后 `gh auth login`（浏览器授权）。
2. 推送前至少完成一次模拟器全流程冒烟（微信开发者工具指向 `http://localhost:8000`，
   DevTools 需勾选"不校验合法域名"）。
3. 决定仓库布局（见 §2）。

## 2. 仓库布局决策（二选一）[人工]
- 方案 A（推荐，干净）：删除根级重复/旧资产
  `app/ configs/ data/ evaluation/ models/ notebooks/ optimizations/ recipe/ training/
   requirements.txt food_calorie_estimation/` 以及零散文件，只保留
  `yitu-zhika-code/ miniprogram/ test_images/ docs/` + 根文档。
  注意：`yitu-zhika-code/src|scripts` 部分 import 依赖旧扁平 `models/` 包名的代码只在
  从仓库根运行时才会踩到；工程内部一律以 `yitu-zhika-code` 为根运行，删根目录不影响。
- 方案 B（保历史）：原样推送全部内容（含 teammate 布局），仓库较大、有重复。
决定后执行：
```
git rm -r app configs data evaluation models notebooks optimizations recipe training requirements.txt   # 方案 A 才执行
git rm -r food_calorie_estimation                                                                        # 方案 A 才执行（如需保留先在本地移走）
git add -A && git commit -m "chore: 清理重复/旧资产，修复 .gitignore 冲突残留"
git push origin main
```

## 3. 模型权重上传（推荐 GitHub Releases，不用 LFS）[人工]
权重合计约 1.6 GB，超出 GitHub LFS 免费配额（1 GB），Releases 更合适（单文件上限 2 GB，
不计 LFS 配额）。
```
git tag v1.0.0 && git push origin v1.0.0
gh release create v1.0.0 \
  checkpoints/meal_rgb_official_v1/best.pt \
  checkpoints/meal_nir_official_v1/best.pt \
  checkpoints/hsi_unet_v2/best.pt \
  checkpoints/food101_pretrained/best.pt \
  checkpoints/calorieclip_official_v1/best.pt \
  checkpoints/capsicum_unet4_seed42/best_model.pth \
  results/weights_manifest.json \
  --title "一图知卡 v1.0.0 模型权重" --notes "见 weights_manifest.json（SHA256 校验）"
```
（在 `yitu-zhika-code/` 目录下执行；`results/` 已 gitignore，`--notes` 里贴 manifest 摘要亦可）

## 4. 服务器部署 [人工]（约 1-2 周，备案是关键路径）
1. 租云服务器（4 核 8G 轻量，60-100 元/月）；域名 + ICP 备案立即启动。
2. HTTPS：Let's Encrypt（certbot）。
3. 部署：仓库拉到服务器 → 建 venv（依赖见 `requirements.txt` + torch CPU）→
   `gunicorn -w 2 -b 127.0.0.1:8000 "app.inference_service:app"` → nginx 反代 + HTTPS。
4. 密钥：新建 `.env`，设置 `INFERENCE_API_KEY`（长随机串，替换 `dev-key-change-in-prod`），
   `.env` 已 gitignore；小程序端同步替换 `app.globalData.apiKey`。
5. 自检：`/health`、`/model-info`、一次 `/predict`、一次 `/weekly-plan`、`/feedback`。

## 5. 小程序发布 [人工]
1. 微信公众平台配置 request/uploadFile 合法域名（你的 HTTPS 域名）。
2. `project.config.json` 填正式 appid；真机预览全流程。
3. 提交审核（1-3 天）；审核材料需含免责声明页（已有 pages/about）。

## 6. 验收红线（推送/发布前对照）
- API 端点除 /health 外均需 X-API-Key；限流 30/min；错误 JSON 标准格式
- 非食物：`status:not_food` + 弹窗"未检测到食物，请拍摄食物图片"，不进结果页
- 输出 1 位小数；反馈含 dish_uuid/image_hash/quality；上传去 EXIF；≤10MB jpg/png
- 食谱含免责声明；tabBar 恰好 4 项；onboarding 首启门禁
- 未授权 401 / 限流 429 / 校验 400 复测通过（见 `results/api_smoke_recheck_result.json`）
