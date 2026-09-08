import os, glob

# Check likely Nutrition5k download locations
paths = [
    "C:/Users/user/Downloads",
    "C:/Users/user/Desktop",
    "C:/Users/user/Desktop/yitu-zhika",
    "C:/Users/user/Desktop/yitu-zhika/yitu-zhika code/data",
    "C:/Users/user/Desktop/yitu-zhika/yitu-zhika code/data/Nutrition5k",
]

for p in paths:
    if os.path.isdir(p):
        items = os.listdir(p)
        print(f"\n{p} ({len(items)} items):")
        for i in items[:30]:
            fp = os.path.join(p, i)
            sz = os.path.getsize(fp) if os.path.isfile(fp) else 0
            tag = f" {sz/1048576:.0f}MB" if sz > 1048576 else ""
            print(f"  {i}{tag}")
    else:
        print(f"\n{p}: NOT EXISTS")

# Find any .zip > 100MB in Downloads (top level only)
dl = "C:/Users/user/Downloads"
if os.path.isdir(dl):
    print("\n=== Large zips in Downloads ===")
    for f in os.listdir(dl):
        if f.lower().endswith('.zip'):
            sz = os.path.getsize(os.path.join(dl, f))
            if sz > 100*1024*1024:
                print(f"  {f}: {sz/1048576:.0f}MB")
