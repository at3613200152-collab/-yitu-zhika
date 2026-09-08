import os

# Search for the 8 tar.gz files
data_base = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika code\data"
search_dirs = [data_base, r"C:\Users\user\Desktop\yitu-zhika", r"C:\Users\user\Desktop"]

for sd in search_dirs:
    if not os.path.isdir(sd):
        continue
    for dp, dn, fn in os.walk(sd):
        depth = dp.replace(sd, "").count(os.sep)
        if depth > 3:
            continue
        for f in fn:
            if 'HSIFood' in f or (f.endswith('.tar.gz') and os.path.getsize(os.path.join(dp,f)) > 100*1024*1024):
                fp = os.path.join(dp, f)
                sz = os.path.getsize(fp)
                print(f"  {fp}  ({sz/1048576:.0f}MB)")
