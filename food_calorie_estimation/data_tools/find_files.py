"""Search for Nutrition5k and any large zip/pkl files."""
import os

search_dirs = [
    "C:/Users/user/Downloads",
    "C:/Users/user/Desktop",
    "C:/Users/user",
]

print("=== Searching for Nutrition5k files ===")
for sd in search_dirs:
    if not os.path.isdir(sd):
        continue
    for dp, dn, fn in os.walk(sd):
        # Skip deep nesting and hidden dirs
        depth = dp.replace(sd, "").count(os.sep)
        if depth > 3:
            continue
        for f in fn:
            fl = f.lower()
            if 'nutrition' in fl or fl.endswith('.pkl') or (fl.endswith('.zip') and os.path.getsize(os.path.join(dp,f)) > 100*1024*1024):
                fp = os.path.join(dp, f)
                sz = os.path.getsize(fp)
                print(f"  {fp}: {sz/1048576:.1f}MB")

print("\n=== Recent large files (>100MB) in Downloads ===")
dl = "C:/Users/user/Downloads"
if os.path.isdir(dl):
    files = []
    for f in os.listdir(dl):
        fp = os.path.join(dl, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            mt = os.path.getmtime(fp)
            if sz > 50*1024*1024:
                files.append((f, sz, mt))
    files.sort(key=lambda x: x[2], reverse=True)
    for f, sz, mt in files[:15]:
        print(f"  {f}: {sz/1048576:.1f}MB")

print("\n=== Recent large files on Desktop ===")
dk = "C:/Users/user/Desktop"
if os.path.isdir(dk):
    for f in os.listdir(dk):
        fp = os.path.join(dk, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            if sz > 50*1024*1024:
                print(f"  {f}: {sz/1048576:.1f}MB")
