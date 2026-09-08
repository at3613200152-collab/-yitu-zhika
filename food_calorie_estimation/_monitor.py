"""实时监控训练, 写到独立文件, 避免被 sandbox 截断."""
import time
import os
import sys

log = r"C:\Users\user\AppData\Local\Temp\trae-agent-toolhost\jobs\job-2a261ed0579d40d0a8926d8b52cdc866\output.log"
ckpt = r"c:\Users\user\Desktop\新建文件夹\food_calorie_estimation\checkpoints\multitask"
out = r"C:\Users\user\Desktop\新建文件夹\food_calorie_estimation\_monitor_log.txt"

# 写到一个独立文件, 避免 sandbox 截断
with open(out, "w", encoding="utf-8") as f:
    f.write(f"=== Monitoring {log} ===\n")
    f.write(f"start: {time.strftime('%H:%M:%S')}\n")

last_size = 0
deadline = time.time() + 60 * 25  # 25 分钟
while time.time() < deadline:
    if os.path.exists(log):
        size = os.path.getsize(log)
        if size > last_size:
            try:
                with open(log, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(last_size)
                    new = f.read()
                # 只抓关键事件
                events = []
                for line in new.split("\n"):
                    s = line.strip()
                    if "Saved best" in s or "Accuracy:" in s or "Weight:" in s or "Epoch " in s and "[Val]" in s and "100%" in s:
                        events.append(s[:200])
                with open(out, "a", encoding="utf-8") as f:
                    for e in events:
                        f.write(f"[{time.strftime('%H:%M:%S')}] {e}\n")
                last_size = size
            except Exception as e:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(f"  [err] {e}\n")
    if os.path.exists(ckpt):
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"  [{time.strftime('%H:%M:%S')}] ckpt: {os.listdir(ckpt)}\n")
    time.sleep(15)

with open(out, "a", encoding="utf-8") as f:
    f.write(f"\n=== done at {time.strftime('%H:%M:%S')} ===\n")
