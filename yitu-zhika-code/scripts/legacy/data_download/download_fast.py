"""
HSIFoodIngr-64 数据集 - aria2c 多线程下载脚本
================================================
用 aria2c 16线程并行下载, 比单线程快5-10倍
支持断点续传

前置: 安装 aria2c
  方式1: conda install -c conda-forge aria2
  方式2: scoop install aria2
  方式3: choco install aria2
  方式4: 从 https://github.com/aria2/aria2/releases 下载exe放到PATH

用法:
  conda activate yitu
  python C:/Users/user/Desktop/download_fast.py
"""
import os
import sys
import time
import subprocess
import http.client
import ssl

TOKEN = "9d1e93b2-01da-499f-b54c-97400c8ef534"
HOST = "dataverse.harvard.edu"

FILES = {
    6570515:  "HSIFoodIngr-64_data_1.zip.tar.gz",
    6571250:  "HSIFoodIngr-64_data_2.zip.tar.gz",
    6571421:  "HSIFoodIngr-64_data_3.zip.tar.gz",
    6571420:  "HSIFoodIngr-64_data_4.zip.tar.gz",
    6571516:  "HSIFoodIngr-64_data_5.zip.tar.gz",
    6572125:  "HSIFoodIngr-64_data_10.zip.tar.gz",
    6574732:  "HSIFoodIngr-64_data_100.zip.tar.gz",
    6574734:  "HSIFoodIngr-64_data_101.zip.tar.gz",
}

DATA_DIR = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data\HSIFoodIngr-64"


def get_s3_url(file_id):
    """获取 Dataverse 303 重定向的 S3 签名 URL"""
    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(HOST, context=ctx, timeout=30)
    try:
        conn.request("GET", f"/api/access/datafile/{file_id}",
                     headers={"X-Dataverse-key": TOKEN})
        resp = conn.getresponse()
        if resp.status in (301, 302, 303, 307):
            return resp.getheader("Location")
        resp.read()
        return None
    finally:
        conn.close()


def find_aria2c():
    """查找 aria2c 可执行文件"""
    for cmd in ["aria2c", "aria2c.exe"]:
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=5)
            return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def download_with_aria2c(s3_url, dest_path, aria2c_cmd="aria2c"):
    """用 aria2c 多线程下载"""
    cmd = [
        aria2c_cmd,
        "-x", "16",          # 16 连接并行
        "-s", "16",          # 16 线程
        "-k", "1M",          # 1MB 分片
        "-c",                # 断点续传
        "--max-tries=5",
        "--retry-wait=10",
        "--timeout=60",
        "--connect-timeout=30",
        "--check-certificate=false",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "-d", os.path.dirname(dest_path),
        "-o", os.path.basename(dest_path),
        s3_url
    ]
    print(f"  aria2c 16线程下载中...")
    result = subprocess.run(cmd, cwd=os.path.dirname(dest_path))
    return result.returncode == 0


def download_with_urllib_resume(s3_url, dest_path):
    """urllib 单线程下载, 支持断点续传"""
    import urllib.request

    # 检查已有文件大小(用于续传)
    resume_from = 0
    if os.path.exists(dest_path):
        resume_from = os.path.getsize(dest_path)
        print(f"  续传: 从 {resume_from // 1048576}MB 继续")

    headers = {"User-Agent": "Python/HSIFoodIngr64-Downloader"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    req = urllib.request.Request(s3_url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=60)

    total = int(resp.getheader("Content-Length", 0))
    if resume_from > 0 and resp.status == 206:
        total += resume_from  # Range 返回的是剩余部分的大小

    mode = "ab" if resume_from > 0 else "wb"
    downloaded = resume_from
    last_time = time.time()
    last_bytes = resume_from

    with open(dest_path, mode) as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_time >= 2:  # 每2秒更新
                speed = (downloaded - last_bytes) / (now - last_time) / 1048576
                if total > 0:
                    pct = downloaded * 100 / total
                    eta_s = (total - downloaded) / (speed * 1048576) if speed > 0 else 0
                    eta_m = int(eta_s // 60)
                    print(f"\r  {pct:.1f}%  {downloaded//1048576}/{total//1048576}MB  "
                          f"{speed:.1f}MB/s  ETA {eta_m}min   ", end="", flush=True)
                last_time = now
                last_bytes = downloaded

    print()
    return downloaded


def download_file(file_id, filename, aria2c_cmd=None):
    """下载单个文件"""
    dest = os.path.join(DATA_DIR, filename)

    # 完整性检查: tar.gz 文件至少应 > 1.5GB
    MIN_SIZE = 1500 * 1024 * 1024  # 1.5GB
    if os.path.exists(dest) and os.path.getsize(dest) > MIN_SIZE:
        print(f"  [跳过] {filename} 已完成 ({os.path.getsize(dest)//1048576}MB)")
        return True

    # 删除不完整的旧文件(小于100MB的视为损坏, 大的保留用于续传)
    if os.path.exists(dest) and os.path.getsize(dest) < 100 * 1024 * 1024:
        print(f"  删除损坏文件 ({os.path.getsize(dest)//1048576}MB)")
        os.remove(dest)

    s3_url = get_s3_url(file_id)
    if not s3_url:
        print(f"  [错误] 无法获取 S3 下载链接")
        return False

    print(f"  S3 URL: {s3_url[:80]}...")

    if aria2c_cmd:
        ok = download_with_aria2c(s3_url, dest, aria2c_cmd)
    else:
        size = download_with_urllib_resume(s3_url, dest)
        ok = size > MIN_SIZE

    if ok and os.path.exists(dest) and os.path.getsize(dest) > MIN_SIZE:
        print(f"  [完成] {filename} ({os.path.getsize(dest)//1048576}MB)")
        return True
    else:
        print(f"  [失败] 文件不完整或下载中断")
        return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 检测 aria2c
    aria2c_cmd = find_aria2c()
    if aria2c_cmd:
        print(f"[OK] 找到 aria2c, 将使用 16 线程并行下载")
    else:
        print(f"[!] 未找到 aria2c, 使用单线程下载 (建议安装: conda install -c conda-forge aria2)")

    print("=" * 55)
    print("  HSIFoodIngr-64 数据集下载")
    print(f"  {'aria2c 16线程' if aria2c_cmd else 'urllib 单线程(续传)'}")
    print(f"  保存到: {DATA_DIR}")
    print("=" * 55)

    ok = 0
    for i, (fid, fname) in enumerate(FILES.items(), 1):
        print(f"\n[{i}/{len(FILES)}] {fname}")
        if download_file(fid, fname, aria2c_cmd):
            ok += 1

    print("\n" + "=" * 55)
    print(f"  完成: {ok}/{len(FILES)} 个")
    if ok == len(FILES):
        print("  全部下载成功! 下一步: 解压 .tar.gz 文件")
    print("=" * 55)


if __name__ == "__main__":
    main()
