# -*- coding: utf-8 -*-
"""
Phase1 后台一键启动脚本（鲁棒版）
- 删除不完整的tar
- 流式下载678MB tar（带进度）
- 解压
- 验证数据
- DETACHED_PROCESS启动训练
全程日志写入 phase1_setup.log
"""
import os
import sys
import subprocess
import time
import urllib.request

# === 配置 ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TAR_NAME = "nirscene1_x10_complete.tar"
TAR_URL = "https://www.coze.cn/s/ADxHiLMq-PI/"
TAR_PATH = os.path.join(DATA_DIR, TAR_NAME)
EXTRACT_DIR = os.path.join(DATA_DIR, "nirscene1_x10", "nirscene_img_aug_10_oversample")
PRETRAINED = r"C:\Users\user\Desktop\新建文件夹\food_calorie_estimation\checkpoints\nir_generator\best.pth"
PYTHON = r"C:\Users\user\miniconda3\envs\yitu\pythonw.exe"
LOG_FILE = os.path.join(PROJECT_ROOT, "phase1_setup.log")

os.chdir(PROJECT_ROOT)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# 清空日志
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(f"=== Phase1 Setup Started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

# === Step 1: 检查是否已解压 ===
log("Step 1: 检查数据状态")
if os.path.isdir(os.path.join(EXTRACT_DIR, "train_A")) and len(os.listdir(os.path.join(EXTRACT_DIR, "train_A"))) > 100:
    log("  数据已解压，跳过下载和解压")
else:
    # === Step 2: 删除不完整的tar ===
    if os.path.exists(TAR_PATH):
        sz = os.path.getsize(TAR_PATH)
        if sz < 600_000_000:  # 小于600MB视为不完整
            log(f"  发现不完整的tar ({sz/1e6:.0f}MB)，删除重新下载")
            os.remove(TAR_PATH)
        else:
            log(f"  tar已存在且大小正常 ({sz/1e6:.0f}MB)")
    
    # === Step 3: 下载tar ===
    if not os.path.exists(TAR_PATH):
        log(f"  开始下载: {TAR_URL}")
        log(f"  目标: {TAR_PATH}")
        try:
            # 流式下载，带进度
            req = urllib.request.Request(TAR_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=300) as response:
                total = int(response.headers.get('Content-Length', 0))
                log(f"  文件大小: {total/1e6:.0f}MB")
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB
                last_report = 0
                with open(TAR_PATH, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= 50_000_000:  # 每50MB报告一次
                            pct = downloaded * 100 / total if total else 0
                            log(f"  下载进度: {downloaded/1e6:.0f}MB / {total/1e6:.0f}MB ({pct:.0f}%)")
                            last_report = downloaded
                log(f"  下载完成: {downloaded/1e6:.0f}MB")
        except Exception as e:
            log(f"  下载失败: {e}")
            log("  请手动从聊天消息下载tar到 data/ 目录")
            sys.exit(1)
    
    # === Step 4: 解压 ===
    log("Step 2: 解压数据")
    os.chdir(DATA_DIR)
    result = subprocess.run(["tar", "xf", TAR_PATH], capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log(f"  解压失败: {result.stderr}")
        sys.exit(1)
    os.chdir(PROJECT_ROOT)
    log("  解压完成")

# === Step 5: 验证数据 ===
log("Step 3: 验证数据")
for split in ["train_A", "train_B", "test_A", "test_B"]:
    d = os.path.join(EXTRACT_DIR, split)
    n = len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.isdir(d) else 0
    log(f"  {split}: {n} files")

# === Step 6: 检查预训练模型 ===
log("Step 4: 检查预训练模型")
has_pretrained = os.path.exists(PRETRAINED)
if has_pretrained:
    sz = os.path.getsize(PRETRAINED) / (1024*1024)
    log(f"  预训练模型: {PRETRAINED} ({sz:.0f}MB)")
else:
    log(f"  预训练模型不存在，将从头训练")

# === Step 7: 启动训练 (DETACHED_PROCESS) ===
log("Step 5: 启动 Phase1 训练")
cmd = [PYTHON, "-u", "src/training/train_generator.py"]
if has_pretrained:
    cmd += ["--pretrained", PRETRAINED]

log(f"  命令: {' '.join(cmd)}")
log(f"  模型: 7层 Deep U-Net")
log(f"  预训练: {'YES (PSNR24)' if has_pretrained else 'NO'}")
log(f"  轮数: 200, batch: 8")

creationflags = 0
if sys.platform == 'win32':
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS

train_log = os.path.join(PROJECT_ROOT, "phase1_train.log")
log_file = open(train_log, 'w')
proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                       creationflags=creationflags, cwd=PROJECT_ROOT)
log(f"  训练已启动! PID={proc.pid}")
log(f"  训练日志: phase1_train.log")
log(f"  预计2-3小时，请勿关闭电脑")
log(f"  完成后检查 checkpoints/phase1/final_model.pth")
log("=== Setup Complete ===")
