"""
HSIFoodIngr-64 数据集下载脚本 (Harvard Dataverse)
====================================================
原理: Dataverse API 返回 303 -> S3签名URL, 再下载 S3 URL
用法: 在 yitu conda 环境中运行
      conda activate yitu
      python C:/Users/user/Desktop/download_data.py
"""
import os
import sys
import time
import http.client
import ssl
import urllib.request

# ====== 配置 ======
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

# 数据保存到项目 data 目录下
DATA_DIR = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data\HSIFoodIngr-64"


def progress(done, total):
    """显示下载进度条"""
    if total <= 0:
        pct = "??"
        bar = "?" * 10
    else:
        pct = f"{done * 100 / total:.1f}"
        filled = int(30 * done / total)
        bar = "=" * filled + "-" * (30 - filled)
    done_mb = done / 1048576
    total_mb = total / 1048576
    print(f"\r  [{bar}] {pct}% {done_mb:.0f}/{total_mb:.0f} MB", end="", flush=True)


def get_s3_url(file_id):
    """
    用 http.client 直接发 HTTP 请求(不自动跟随重定向)
    带上认证 header 获取 303 重定向的 S3 签名 URL
    """
    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(HOST, context=ctx, timeout=30)
    try:
        conn.request(
            "GET",
            f"/api/access/datafile/{file_id}",
            headers={"X-Dataverse-key": TOKEN}
        )
        resp = conn.getresponse()
        if resp.status in (301, 302, 303, 307):
            return resp.getheader("Location")
        else:
            resp.read()
            return None
    finally:
        conn.close()


def download_from_s3(s3_url, dest_path):
    """从 S3 签名 URL 下载文件(不需要任何认证)"""
    req = urllib.request.Request(s3_url, headers={
        "User-Agent": "Python/HSIFoodIngr64-Downloader"
    })
    resp = urllib.request.urlopen(req, timeout=60)
    total = int(resp.getheader("Content-Length", 0))
    with open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)  # 1MB 一块
            if not chunk:
                break
            f.write(chunk)
            progress(f.tell(), total)
    print()  # 换行
    return total


def download_file(file_id, filename):
    """完整下载流程: 获取 S3 URL -> 下载文件"""
    dest = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
        print(f"  [跳过] {filename} 已存在 ({os.path.getsize(dest) // 1048576}MB)")
        return True

    # 如果文件存在但太小(下载中断), 删掉重来
    if os.path.exists(dest):
        os.remove(dest)

    for attempt in range(3):
        try:
            if attempt > 0:
                print(f"  [重试] 第 {attempt + 1} 次尝试...")
                time.sleep(3)

            s3_url = get_s3_url(file_id)
            if not s3_url:
                print(f"  [错误] 无法获取下载链接 (file_id={file_id})")
                continue

            download_from_s3(s3_url, dest)
            size = os.path.getsize(dest)
            if size > 100 * 1024 * 1024:
                print(f"  [完成] {filename} ({size // 1048576}MB)")
                return True
            else:
                print(f"  [错误] 文件太小({size // 1048576}MB), 可能下载不完整")
                os.remove(dest)
        except Exception as e:
            print(f"\n  [异常] {e}")
            if os.path.exists(dest):
                os.remove(dest)

    return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 55)
    print("  HSIFoodIngr-64 数据集下载 (Harvard Dataverse)")
    print("=" * 55)
    print(f"  保存到: {DATA_DIR}")
    print(f"  共 {len(FILES)} 个分卷, 每个约 1.7~1.9GB")
    print(f"  总计约 14GB, 请确保网络稳定")
    print("=" * 55)
    print()

    ok = 0
    fail = 0
    for i, (fid, fname) in enumerate(FILES.items(), 1):
        print(f"[{i}/{len(FILES)}] {fname}")
        if download_file(fid, fname):
            ok += 1
        else:
            fail += 1
        print()

    print("=" * 55)
    print(f"  下载完成: 成功 {ok} 个, 失败 {fail} 个")
    if fail > 0:
        print("  重新运行脚本即可续传已失败的文件")
    else:
        print("  全部下载成功!")
        print(f"  下一步: 解压所有 .zip.tar.gz 文件到 {DATA_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
