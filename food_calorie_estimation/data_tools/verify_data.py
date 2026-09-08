"""
一图知卡 - 数据加载验证脚本
==============================
验证 HSIFoodIngr-64 和 Nutrition5k 数据集是否可用
用法:
    conda activate yitu
    python C:/Users/user/Desktop/verify_data.py
"""
import os
import sys
import glob
import tarfile
import zipfile

BASE = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data"
HSI_DIR = os.path.join(BASE, "HSIFoodIngr-64")
N5K_DIR = os.path.join(BASE, "Nutrition5k")


def check_hsi():
    print("=" * 50)
    print("  HSIFoodIngr-64 数据检查")
    print("=" * 50)

    if not os.path.exists(HSI_DIR):
        print(f"  [!] 目录不存在: {HSI_DIR}")
        return False

    # 查找 tar.gz 文件
    targz = sorted(glob.glob(os.path.join(HSI_DIR, "*.tar.gz")))
    # 查找已解压的 zip 文件
    zips = sorted(glob.glob(os.path.join(HSI_DIR, "*.zip")))
    # 查找已解压的目录
    dirs = [d for d in os.listdir(HSI_DIR) if os.path.isdir(os.path.join(HSI_DIR, d))]

    print(f"  tar.gz 文件: {len(targz)} 个")
    for f in targz:
        size_mb = os.path.getsize(f) // 1048576
        print(f"    {os.path.basename(f)}: {size_mb}MB")

    print(f"  zip 文件: {len(zips)} 个")
    for f in zips:
        size_mb = os.path.getsize(f) // 1048576
        print(f"    {os.path.basename(f)}: {size_mb}MB")

    print(f"  子目录: {len(dirs)} 个")
    for d in dirs:
        print(f"    {d}/")

    # 检查是否有完整数据(已解压的目录中有 .hdr 文件说明有 HSI 数据)
    hsi_files = []
    for root, _, files in os.walk(HSI_DIR):
        for f in files:
            if f.endswith('.hdr') or f.endswith('.raw'):
                hsi_files.append(os.path.join(root, f))
                if len(hsi_files) >= 5:
                    break
        if len(hsi_files) >= 5:
            break

    if hsi_files:
        print(f"\n  [OK] 找到 HSI 数据文件 (.hdr/.raw)")
        print(f"       示例: {hsi_files[0]}")
        return True
    elif targz:
        print(f"\n  [!] 有 {len(targz)} 个压缩包, 需要先解压")
        print(f"       解压命令: tar -xzf <文件> -C {HSI_DIR}")
        return False
    else:
        print(f"\n  [!] 没有可用数据")
        return False


def check_n5k():
    print("\n" + "=" * 50)
    print("  Nutrition5k 数据检查")
    print("=" * 50)

    if not os.path.exists(N5K_DIR):
        print(f"  [!] 目录不存在: {N5K_DIR}")
        return False

    # 查找关键文件/目录
    all_items = os.listdir(N5K_DIR)
    print(f"  内容: {all_items[:20]}")

    # 查找营养元数据
    meta_files = []
    for root, _, files in os.walk(N5K_DIR):
        for f in files:
            if f.endswith('.csv') or f.endswith('.json'):
                meta_files.append(os.path.join(root, f))
                if len(meta_files) >= 5:
                    break
        if len(meta_files) >= 5:
            break

    # 查找图像文件
    img_files = []
    for root, _, files in os.walk(N5K_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_files.append(os.path.join(root, f))
                if len(img_files) >= 3:
                    break
        if len(img_files) >= 3:
            break

    if meta_files:
        print(f"\n  [OK] 找到元数据文件: {len(meta_files)} 个")
        print(f"       示例: {meta_files[0]}")
    if img_files:
        print(f"  [OK] 找到图像文件: {len(img_files)} 个")
        print(f"       示例: {img_files[0]}")

    if meta_files and img_files:
        return True
    elif all_items:
        print(f"\n  [!] 目录非空但未找到标准格式数据")
        return False
    else:
        print(f"\n  [!] 空目录, 需要从 Kaggle 下载数据集")
        print(f"       https://www.kaggle.com/datasets/siddhantrout/nutrition5k-dataset")
        return False


def check_pytorch():
    print("\n" + "=" * 50)
    print("  PyTorch & GPU 检查")
    print("=" * 50)
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem // 1048576}MB")
        return True
    except ImportError:
        print("  [!] PyTorch 未安装")
        return False


def main():
    print("\n一图知卡 - 数据与环境验证\n")

    hsi_ok = check_hsi()
    n5k_ok = check_n5k()
    torch_ok = check_pytorch()

    print("\n" + "=" * 50)
    print("  总结")
    print("=" * 50)
    print(f"  HSIFoodIngr-64: {'OK' if hsi_ok else '待准备'}")
    print(f"  Nutrition5k:     {'OK' if n5k_ok else '待准备'}")
    print(f"  PyTorch/GPU:     {'OK' if torch_ok else '待安装'}")

    if hsi_ok and n5k_ok and torch_ok:
        print("\n  全部就绪! 可以开始训练了")
    else:
        print("\n  待办:")
        if not hsi_ok:
            print("    - 下载并解压 HSIFoodIngr-64 数据集")
        if not n5k_ok:
            print("    - 从 Kaggle 下载 Nutrition5k 数据集")
    print("=" * 50)


if __name__ == "__main__":
    main()
