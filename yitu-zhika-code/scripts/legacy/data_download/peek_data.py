import os

def show(p, depth=0):
    if not os.path.isdir(p):
        print(f"{'  '*depth}{p}: NOT EXISTS")
        return
    items = os.listdir(p)
    print(f"{'  '*depth}{p} ({len(items)} items):")
    for i in items[:40]:
        fp = os.path.join(p, i)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            tag = f" {sz/1048576:.0f}MB" if sz > 1048576 else f" {sz/1024:.0f}KB" if sz > 1024 else ""
            print(f"{'  '*(depth+1)}{i}{tag}")
        elif os.path.isdir(fp):
            sub = os.listdir(fp)
            print(f"{'  '*(depth+1)}{i}/ ({len(sub)} items)")

data_base = "C:/Users/user/Desktop/yitu-zhika/yitu-zhika code/data"

show(os.path.join(data_base, "Nutrition5k"))
show(os.path.join(data_base, "Nutrition5k/extracted"))
show(os.path.join(data_base, "archive"))

# Also check if there's a Nutrition5k zip anywhere under data
print("\n=== zip/pkl/xlsx under data/ ===")
for dp, dn, fn in os.walk(data_base):
    for f in fn:
        fl = f.lower()
        if fl.endswith('.zip') or fl.endswith('.pkl') or fl.endswith('.xlsx') or fl.endswith('.tar.gz'):
            fp = os.path.join(dp, f)
            sz = os.path.getsize(fp)
            print(f"  {fp}: {sz/1048576:.0f}MB" if sz > 1048576 else f"  {fp}: {sz/1024:.0f}KB")
