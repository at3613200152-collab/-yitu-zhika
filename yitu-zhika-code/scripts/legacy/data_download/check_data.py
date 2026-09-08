"""Check data status for yitu-zhika project."""
import os

# Check data directories
data_root = "C:/Users/user/Desktop/yitu-zhika/yitu-zhika code/data"
for ds in ["Nutrition5k", "HSIFoodIngr-64"]:
    ds_path = os.path.join(data_root, ds)
    print(f"\n=== {ds} ===")
    if not os.path.isdir(ds_path):
        print("  NOT FOUND")
        continue
    total_size = 0
    file_count = 0
    for dp, dn, fn in os.walk(ds_path):
        for f in fn:
            fp = os.path.join(dp, f)
            sz = os.path.getsize(fp)
            total_size += sz
            file_count += 1
            rel = os.path.relpath(fp, ds_path)
            if sz > 50 * 1024 * 1024:
                print(f"  {rel}: {sz // 1048576}MB")
    if file_count == 0:
        print("  (empty directory)")
    else:
        print(f"  Total: {file_count} files, {total_size / 1048576:.0f}MB ({total_size / 1073741824:.2f}GB)")

# Check Downloads for nutrition5k files
print("\n=== Downloads (nutrition5k related) ===")
dl_dir = "C:/Users/user/Downloads"
if os.path.isdir(dl_dir):
    found = False
    for f in os.listdir(dl_dir):
        if "nutrition" in f.lower() or "Nutrition5k" in f:
            fp = os.path.join(dl_dir, f)
            sz = os.path.getsize(fp) if os.path.isfile(fp) else 0
            print(f"  {f}: {sz // 1048576}MB")
            found = True
    if not found:
        print("  No nutrition5k files found")
        # Show recent large files
        print("  Recent large files in Downloads:")
        files = []
        for f in os.listdir(dl_dir):
            fp = os.path.join(dl_dir, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp)
                mt = os.path.getmtime(fp)
                files.append((f, sz, mt))
        files.sort(key=lambda x: x[2], reverse=True)
        for f, sz, mt in files[:10]:
            print(f"    {f}: {sz // 1048576}MB")
