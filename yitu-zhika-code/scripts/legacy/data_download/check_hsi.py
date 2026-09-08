"""检查HSI数据目录结构"""
import os

base = r"C:\Users\user\Desktop\yitu-zhika\yitu-zhika-code\data\HSIFoodIngr-64"
print(f"Base: {base}")
print(f"Exists: {os.path.exists(base)}\n")

if os.path.exists(base):
    items = os.listdir(base)
    print(f"Top level ({len(items)} items):")
    for item in items[:20]:
        full = os.path.join(base, item)
        if os.path.isdir(full):
            sub_items = os.listdir(full)
            print(f"  📁 {item}/ ({len(sub_items)} items)")
            # show first few files in subdir
            for sub in sub_items[:5]:
                print(f"      {sub}")
            if len(sub_items) > 5:
                print(f"      ... +{len(sub_items)-5} more")
        else:
            size = os.path.getsize(full)
            print(f"  📄 {item} ({size/1024:.1f}KB)")
    
    # Count .hdr files recursively
    hdr_count = 0
    raw_count = 0
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith('.hdr'):
                hdr_count += 1
            elif f.endswith('.raw'):
                raw_count += 1
    print(f"\nTotal .hdr files: {hdr_count}")
    print(f"Total .raw files: {raw_count}")
    
    # Show full path of first .hdr
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith('.hdr'):
                print(f"\nFirst .hdr: {os.path.join(root, f)}")
                raw_match = os.path.join(root, f.replace('.hdr', '.raw'))
                print(f"Matching .raw exists: {os.path.exists(raw_match)}")
                break
        else:
            continue
        break
