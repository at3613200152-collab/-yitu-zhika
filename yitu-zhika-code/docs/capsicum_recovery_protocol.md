# Capsicum 恢复与独立训练协议（2026-09-04）

## 目标与边界

本轮恢复可靠下载，下载完成后自动做校验、数据质量检查、冒烟训练及独立 RGB→NIR 实验。Capsicum 是甜椒农场域辅助实验，不是 Nutrition5k 餐盘热量模型的替代品；其 PSNR/SSIM 提升不能直接证明热量 MAPE/RMSE 改善。允许未来修改 Demo/权重，但必须先有同一餐盘测试集上的可复现实验支持。

官方来源：[Zenodo 6340415](https://zenodo.org/records/6340415)，[作者格式说明](https://inkyusa.github.io/deepNIR_dataset/download/synth/)。许可 CC BY 4.0，需要署名。

- 文件：capsicums_pix2pixHD_8_1_1.tar.gz
- 官方 API 核验大小：3,256,378,152 bytes
- MD5：1d8ad73bcfe917f23806fd3417dc5d12
- 预期：1,615 对 RGB/NIR，显式 train/val/test。

## 下载故障与修复证据

原命令在连接重置后以 curl 56 退出。保留原下载文件，没有从零重下。回归测试使用本地 HTTP 服务，在传输部分字节后触发 TCP reset；原 curl 参数失败，新程序能按正确 Range 恢复，最终内容逐字节一致。

新程序使用独立本地进程，不依靠 Codex 唤醒来完成阶段切换。连接/读取超时分别为 15/20 秒；长连接每 120 秒重新续传；异常有退避；每 10 秒更新状态。大小或 Range 异常立即阻止追加；最终通过大小和 MD5 后才可解压。最多 12 小时下载、12 次连续零进度；完整/部分文件均保留。curl 默认重试不是所有错误都重试，参见 [curl 官方说明](https://curl.se/docs/manpage.html#--retry-all-errors)。

## 数据质量门槛

安全解压拒绝绝对路径、路径穿越、链接和异常膨胀（8 GiB 上限）。配对严格按文件名，不按排序强行 zip；检查所有图像可解码、配对尺寸相同、NIR 三通道同值，自动判定 A/B 方向。保留官方划分，禁止用 test 代替 val；检查原始帧 ID 和跨 split 图像像素 SHA-256 重复。未证明同一视频近邻帧完全独立，因此官方 split 上的成绩必须注明这一局限。

## 训练协议

- 4 层 U-Net（已有 models/generator.py），从头训练，seed=42；不沿用来源不明的预训练权重。
- 256×256、RGB 3 通道→NIR 1 通道，输入/目标均 [-1,1]。
- L1 监督，Adam lr=0.0002，paired horizontal flip，CUDA AMP。
- 默认 batch=8，最多 100 轮；验证 L1 选择最优权重；15 轮不改善早停；正式训练每次运行最多 8 小时。
- 先对真实数据执行最多 2 个训练 batch + 2 个验证 batch；丢弃冒烟训练参数并重置随机种子后启动正式实验。
- PSNR 按图像在 [0,1] 上计算后取平均（MSE 下限 1e-12）；SSIM 使用 skimage、data_range=1；L1 报告 [-1,1] 尺度。
- test 仅在训练结束选定 best_model 后评估。记录 checkpoint 和 manifest 哈希、随机状态及实际库版本，支持从 last_model 续训。
- 保留至少 35 GiB 空间。NaN、OOM、文件校验/配对失败时停止，不生成伪成绩。

## 运行和查看

项目目录：C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code。

```powershell
& 'C:\Users\user\miniconda3\envs\yitu\python.exe' -X utf8 -u scripts\capsicum_job.py --epochs 100 --batch-size 8
```

程序持有 OS 文件锁；锁文件留存不等于进程存活。重复实例会被阻止。不要另外同时运行旧 curl 或其他解压/训练命令。

- `data/deepNIR_capsicum/job_status.json`：阶段、PID、字节、重试次数、更新时间。
- `data/deepNIR_capsicum/job_console.log` / `job_stderr.log`：当前后台启动的日志。
- `data/deepNIR_capsicum/manifest.json`：校验通过后生成的配对清单。
- `data/deepNIR_capsicum/training_console.log`：正式数据冒烟及训练日志。
- `results/capsicum_unet4_seed42/`：协议、逐轮指标、训练状态、最终测试指标。
- `checkpoints/capsicum_unet4_seed42/`：best_model.pth、last_model.pth。

需要暂停时，在 `data/deepNIR_capsicum/` 放置 `STOP` 文件。程序在检查点停止，保留文件。续跑前移除这个暂停标记。此程序没有创建开机启动项、改电源设置、上传数据或推送 GitHub。

## 后续餐盘实验（尚未执行）

先核对 Nutrition5k 官方 dish 划分与当前 CSV/图片覆盖率；按 dish 固定划分，所有视角保持同 split。在相同训练数据、回归目标、训练预算和指标定义下，训练 RGB-only 与 RGB+预测 NIR；优先报告独立 test 上的 kcal/weight MAE、标准 MAPE、RMSE，以及零真值处理规则。按 dish 做配对误差比较；资源允许时运行多 seed 并给置信区间。外部开源 RGB 基线须另行复现，不能把自建 RGB-only 消融称作已经完成开源基线要求。完成这些验证后再决定是否更换 Demo 权重。
