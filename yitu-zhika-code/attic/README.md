# attic —— 已移出交付范围的内容（2026-09-10）

按项目负责人要求，以下内容**不再属于本课程设计项目的交付范围**，移到这里保留以便回溯。
需要恢复时用 `git mv` 移回原位即可（历史提交里仍有完整记录）。

## side_angle/ —— 侧视 / 拍摄角度相关
| 文件 | 原位置 | 移出原因 |
|---|---|---|
| `eval_side_views.py` | `scripts/` | 侧视帧评估（角度泛化检验） |
| `check_side_frame_duplication.py` | `scripts/` | 侧视三帧重复性检查 |
| `build_side_angle_index.py` | `scripts/` | 侧视索引表生成 |
| `侧视图位置与检验说明_2026-09-10.md` | `docs/` | 侧视数据位置与检验口径 |
| `侧视检验结果_2026-09-10.md` | `docs/` | 侧视/角度误差结论 |

> 说明：Nutrition5k 冻结 3262 之外那 128 条补充样本**仍在**训练清单里（`meal_macros_expanded_v1`，3390 行），
> 在线回归模型 `v1_expanded` 即基于它训练。本次只移除"角度分析/侧视检验"这条线，未改动数据与权重。
> 若要求连这 128 条也剔除，需要重训一版 3262 基线后再替换在线模型（约 30 分钟，见 `v4_noside_seed42` 的先例）。

## deploy/ —— 公网发布 / 线上部署相关
| 文件 | 说明 |
|---|---|
| `DEPLOY.md` | gunicorn + nginx + systemd 上线文档 |
| `gunicorn.conf.py`、`deploy.sh`、`env.prod.example` | 生产部署配置 |
| `nginx/nginx-yitu-zhika.conf` | 反向代理 + HTTPS/域名配置 |
| `systemd/yitu-zhika.service` | 服务托管 |
| `start_local.ps1`、`package_runtime.ps1` | 本地启动/打包脚本（随目录一并移出） |

> 说明：项目当前形态为**本地开发与局域网演示**，不做公网发布；如需上线再取回本目录。
