"""
Extract all datasets for yitu-zhika project.
1. HSIFoodIngr-64: .zip.tar.gz → tar解压 → zip解压 → 原始HSI数据
2. Nutrition5k: Kaggle zip → pkl + xlsx

Usage:
    python extract_all_data.py
"""
import os
import sys
import zipfile
import tarfile
import glob
import time

DATA_ROOT = "C:/Users/user/Desktop/yitu-zhika/yitu-zhika code/data"

def extract_hsifood():
    """Extract HSIFoodIngr-64 .zip.tar.gz files."""
    hsi_dir = os.path.join(DATA_ROOT, "HSIFoodIngr-64")
    if not os.path.isdir(hsi_dir):
        print(f"[SKIP] HSIFoodIngr-64 directory not found")
        return
    
    # Find all .tar.gz files
    targz_files = sorted(glob.glob(os.path.join(hsi_dir, "*.tar.gz")))
    if not targz_files:
        # Maybe already extracted? Check for .zip files
        zip_files = sorted(glob.glob(os.path.join(hsi_dir, "*.zip")))
        if zip_files:
            print(f"[HSIFoodIngr-64] No .tar.gz found, but {len(zip_files)} .zip files exist")
            print("  Extracting zip files...")
            for zf_path in zip_files:
                extract_dir = os.path.join(hsi_dir, os.path.basename(zf_path).replace(".zip", ""))
                if os.path.isdir(extract_dir):
                    print(f"  [SKIP] {os.path.basename(zf_path)} already extracted")
                    continue
                try:
                    with zipfile.ZipFile(zf_path, 'r') as zf:
                        zf.extractall(extract_dir)
                    print(f"  [OK] {os.path.basename(zf_path)} → {os.path.basename(extract_dir)}/")
                except Exception as e:
                    print(f"  [ERR] {os.path.basename(zf_path)}: {e}")
            return
        print(f"[SKIP] No .tar.gz or .zip files found in HSIFoodIngr-64")
        return
    
    print(f"\n=== HSIFoodIngr-64: {len(targz_files)} .tar.gz files ===")
    
    for tgz_path in targz_files:
        fname = os.path.basename(tgz_path)
        # Expected: HSIFoodIngr-64_data_1.zip.tar.gz
        base_name = fname.replace(".tar.gz", "")  # HSIFoodIngr-64_data_1.zip
        
        # Step 1: Extract tar.gz → get .zip file
        print(f"\n[1/2] Extracting tar.gz: {fname}")
        try:
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(hsi_dir)
            print(f"  → tar extracted")
        except Exception as e:
            print(f"  [ERR] tar extraction failed: {e}")
            continue
        
        # Step 2: Extract .zip
        zip_path = os.path.join(hsi_dir, base_name)
        if not os.path.exists(zip_path):
            # Check if tar extracted to a subdirectory
            possible = glob.glob(os.path.join(hsi_dir, "**", base_name), recursive=True)
            if possible:
                zip_path = possible[0]
            else:
                print(f"  [WARN] Expected zip not found: {base_name}")
                # List what tar produced
                for dp, dn, fn in os.walk(hsi_dir):
                    for f in fn:
                        if f.endswith('.zip') and 'data_1' in f:
                            print(f"    Found: {os.path.join(dp, f)}")
                continue
        
        extract_dir = os.path.join(hsi_dir, base_name.replace(".zip", ""))
        if os.path.isdir(extract_dir):
            print(f"  [SKIP] {base_name} already extracted to {os.path.basename(extract_dir)}/")
            continue
        
        print(f"[2/2] Extracting zip: {base_name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            print(f"  → zip extracted to {os.path.basename(extract_dir)}/")
        except Exception as e:
            print(f"  [ERR] zip extraction failed: {e}")
        
        # Optional: remove intermediate .zip to save space
        # os.remove(zip_path)
    
    # Summary
    print(f"\n--- HSIFoodIngr-64 Summary ---")
    for dp, dn, fn in os.walk(hsi_dir):
        depth = dp.replace(hsi_dir, "").count(os.sep)
        if depth <= 2:
            for f in fn[:5]:  # Show first 5 files per dir
                fp = os.path.join(dp, f)
                sz = os.path.getsize(fp)
                rel = os.path.relpath(fp, hsi_dir)
                if sz > 10*1024*1024:
                    print(f"  {rel}: {sz//1048576}MB")


def extract_nutrition5k():
    """Extract Nutrition5k Kaggle zip."""
    n5k_dir = os.path.join(DATA_ROOT, "Nutrition5k")
    
    # Check if already extracted (pkl exists)
    pkl_files = glob.glob(os.path.join(n5k_dir, "**", "*.pkl"), recursive=True)
    xlsx_files = glob.glob(os.path.join(n5k_dir, "**", "*.xlsx"), recursive=True)
    
    if pkl_files or xlsx_files:
        print(f"\n[Nutrition5k] Already extracted:")
        for f in pkl_files + xlsx_files:
            sz = os.path.getsize(f)
            print(f"  {os.path.relpath(f, n5k_dir)}: {sz/1048576:.1f}MB")
        return
    
    # Find zip file
    zip_files = glob.glob(os.path.join(n5k_dir, "*.zip"))
    if not zip_files:
        # Check data root
        zip_files = glob.glob(os.path.join(DATA_ROOT, "nutrition5k*.zip"))
    if not zip_files:
        # Check Downloads
        zip_files = glob.glob(os.path.join("C:/Users/user/Downloads", "nutrition5k*.zip"))
    if not zip_files:
        print(f"\n[Nutrition5k] No zip found. Please download from Kaggle first.")
        print(f"  URL: https://www.kaggle.com/datasets/siddhantrout/nutrition5k-dataset")
        return
    
    zip_path = zip_files[0]
    print(f"\n=== Nutrition5k: Extracting {os.path.basename(zip_path)} ===")
    os.makedirs(n5k_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            print(f"  {len(members)} files in zip:")
            for m in members[:10]:
                info = zf.getinfo(m)
                print(f"    {m}: {info.file_size/1048576:.1f}MB")
            if len(members) > 10:
                print(f"    ... and {len(members)-10} more")
            
            zf.extractall(n5k_dir)
            print(f"  → Extracted to {n5k_dir}")
    except Exception as e:
        print(f"  [ERR] Extraction failed: {e}")
        return
    
    # Verify
    pkl_files = glob.glob(os.path.join(n5k_dir, "**", "*.pkl"), recursive=True)
    xlsx_files = glob.glob(os.path.join(n5k_dir, "**", "*.xlsx"), recursive=True)
    print(f"\n  Found: {len(pkl_files)} pkl, {len(xlsx_files)} xlsx")
    for f in pkl_files + xlsx_files:
        sz = os.path.getsize(f)
        print(f"    {os.path.relpath(f, n5k_dir)}: {sz/1048576:.1f}MB")


def main():
    print("=" * 60)
    print("yitu-zhika Data Extraction")
    print("=" * 60)
    
    extract_hsifood()
    extract_nutrition5k()
    
    print("\n" + "=" * 60)
    print("Done! Next step: run extract_nutrition5k.py to convert pkl → jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()
