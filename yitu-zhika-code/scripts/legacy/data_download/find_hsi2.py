import os

search_dirs = [
    r"C:\Users\user\Downloads",
    r"C:\Users\user",
    r"C:\Users\user\Desktop",
]

found = False
for sd in search_dirs:
    if not os.path.isdir(sd):
        continue
    for dp, dn, fn in os.walk(sd):
        # Skip deep nesting and hidden/git
        depth = dp.replace(sd, "").count(os.sep)
        if depth > 4:
            dn.clear()
            continue
        # Skip git dirs
        dn[:] = [d for d in dn if d not in ('.git', 'node_modules', '__pycache__', '.conda')]
        for f in fn:
            fl = f.lower()
            if 'hsifood' in fl or fl.endswith('.tar.gz'):
                fp = os.path.join(dp, f)
                sz = os.path.getsize(fp)
                print(f"  {fp}  ({sz/1048576:.0f}MB)")
                found = True

if not found:
    print("No HSIFood or .tar.gz files found anywhere!")
    # List large files in Downloads as a hint
    dl = r"C:\Users\user\Downloads"
    if os.path.isdir(dl):
        print(f"\nLarge files (>500MB) in Downloads:")
        for f in os.listdir(dl):
            fp = os.path.join(dl, f)
            if os.path.isfile(fp) and os.path.getsize(fp) > 500*1024*1024:
                print(f"  {f}: {os.path.getsize(fp)/1048576:.0f}MB")
