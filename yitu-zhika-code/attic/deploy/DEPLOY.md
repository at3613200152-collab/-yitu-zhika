# 一图知卡 上线部署指南（租服务器 → 公网 → 开放给减肥群体）

> 目标：让微信小程序真正在手机上可用。关键路径是 **ICP 备案（1–2 周）**，越早开始越好。
> 副本：`deploy/gunicorn.conf.py`、`deploy/systemd/*`、`deploy/nginx/*`、`deploy/deploy.sh`、`deploy/env.prod.example`、`deploy/package_runtime.ps1`。

## ① 租服务器
- 云厂商：阿里云 / 腾讯云 / 华为云。**选国内地域**（若要做微信小程序正式版，域名需 ICP 备案）。
- 规格（早期内测，CPU 够用）：**4 核 8G，Ubuntu 22.04**（约 100–200 元/月）。8G 内存可带 2 个 gunicorn worker（各 ~1.3GB 模型）。想更快/并发更高再上 GPU 或加机。
- 系统盘 ≥80G；带宽建议 **5–10M 按量或固定**（上传的是图片）。
- 安全组：只开 **22(SSH)、80、443**；SSH 用密钥登录、关闭密码/root 远程登录；装 `ufw` 放行 22/80/443。

## ② 域名 + ICP 备案（关键路径）
1. 在同平台注册域名（.com/.cn），**实名**。
2. 提交 **ICP 备案**（云平台 App 内，用服务器 IP + 域名 + 身份证 + 手机号，人脸核验）。约 1–2 周。**备案是微信小程序"合法域名"的前提**。
3. 域名解析：A 记录 → 服务器公网 IP。可先加 `www` 与 裸域两条。

## ③ 把代码与运行时文件放到服务器
模型权重、审计文件、营养库**不在 git 里**（被 gitignore），需单独上传：
- 本机运行 `deploy\package_runtime.ps1`（Windows）→ 生成 `yitu-zhika-code-runtime.zip`（含 checkpoints/results/data）。
- 上传并解压到 `/opt/yitu-zhika`：
  ```
  scp yitu-zhika-code-runtime.zip user@服务器IP:/home/user/
  ssh user@服务器IP
  sudo mkdir -p /opt/yitu-zhika && sudo unzip /home/user/yitu-zhika-code-runtime.zip -d /opt/yitu-zhika
  ```
- 若走 `git clone`（只到代码），再把 `checkpoints/`、`results/`、`data/nutrition5k/` 用 `scp -r` 补上。

## ④ 安装依赖并自检
```
cd /opt/yitu-zhika
chmod +x deploy/deploy.sh && sudo bash deploy/deploy.sh
```
脚本会建 venv、装 torch CPU + `requirements.txt`、校验权重/营养库是否存在、并做一次模型加载冒烟。
> 若这是 git clone 的代码，先 `cp deploy/env.prod.example .env` 并生成强 key；`data/`、`results/`、`checkpoints/` 若缺失，参考上一步补传。

## ⑤ 配置密钥（生产必须换掉 dev 值）
`/opt/yitu-zhika/.env`：
```
INFERENCE_API_KEY=$(openssl rand -hex 32)
MERCHANT_API_KEY=$(openssl rand -hex 32)
ADMIN_API_KEY=$(openssl rand -hex 32)
```
小程序端 `miniprogram/app.js` 的 `apiKey` 改为与 `INFERENCE_API_KEY` 相同；`apiBase` 改为 `https://你的域名`。

## ⑥ gunicorn + systemd 守护
```
sudo cp deploy/systemd/yitu-zhika.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now yitu-zhika
sudo systemctl status yitu-zhika
# 日志：sudo journalctl -u yitu-zhika -f
```

## ⑦ HTTPS（Let's Encrypt 免费）
```
sudo apt update && sudo apt install -y nginx certbot python3-certbot-nginx
sudo cp deploy/nginx/nginx-yitu-zhika.conf /etc/nginx/sites-available/yitu-zhika
# 把里面两处 "你的域名" 换成真实域名
sudo ln -s /etc/nginx/sites-available/yitu-zhika /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d 你的域名 --redirect
```

## ⑧ 微信公众平台配置（开放给减肥群体前必做）
- 「开发管理 → 开发设置 → 服务器域名」：把 `https://你的域名` 加入 **request 合法域名** 和 **uploadFile 合法域名**（微信要求：已备案 + HTTPS + 一级域名，不能是 IP/localhost）。
- 「隐私保护指引」：说明收集照片、记录、健康信息的用途/保存/第三方；训练授权默认关（项目已实现）。
- 「类目与资质」：涉及食品/健康类目可能需要相应资质，按微信提示准备。
- 发布：用 `https://你的域名` 替换小程序 `apiBase` 后，`工具 → 上传 → 提交审核`（1–3 天）。

## ⑨ 安全与合规（给减肥群体用，务必做到）
- 强 API key；`.env` 不放 git；用户照片走 `data/annotations/`（私有，已 gitignore），访问控制 + HTTPS 传输。
- 上传 10MB、限流 30/min、标准化错误 JSON（已内置于服务）。
- 明确"估算值非真实摄入/非医疗诊断"，食谱为"AI 建议草案"，不做减重保证、不显示营养师认证（无真人审核）。
- 遵守《个人信息保护法》：单独同意、可撤回、保留期限说明、删除申请入口。

## ⑩ 上线前后自检
```
curl -k https://你的域名/health
curl -k -X POST https://你的域名/predict -H "X-API-Key: $INFERENCE_API_KEY" -F "image=@a.jpg"
# 预期返回 category_name / calories / weight / category_probs
```
- 真机预览通过；`/feedback`、`/weekly-plan`、`/plan-from-menu`、`/public/menus` 可用。
- 上线后先小范围内测，观察用户授权率、标签可靠性与失败率，再考虑扩量/上 GPU。
