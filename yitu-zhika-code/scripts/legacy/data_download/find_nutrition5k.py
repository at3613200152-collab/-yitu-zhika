"""Find Nutrition5k pkl and xlsx files in data directory."""
import os

data_root = "C:/Users/user/Desktop/yitu-zhika/yitu-zhika code/data"
print(f"Scanning {data_root} ...")

for dp, dn, fn in os.walk(data_root):
    for f in fn:
        if f.endswith(('.pkl', '.xlsx', '.csv', '.zip')):
            fp = os.path.join(dp, f)
            sz = os.path.getsize(fp)
            print(f"  {os.path.relpath(fp, data_root)}: {sz/1048576:.1f}MB")

# Also check common download locations
for check_dir in ["C:/Users/user/Downloads", "C:/Users/user/Desktop"]:
    if os.path.isdir(check_dir):
        for f in os.listdir(check_dir):
            if 'nutrition' in f.lower() or f.endswith('.pkl') or 'dish' in f.lower() and f.endswith('.zip'):
                fp = os.path.join(check_dir, f)
                if os.path.isfile(fp):
                    sz = os.path.getsize(fp)
                    if sz > 1024:
                        print(f"  [Downloads] {f}: {sz/1048576:.1f}MB")
