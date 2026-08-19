"""Derived train annotation file: fill missing 'area' (= bbox w*h) and 'iscrowd' (= 0).

Dome's CocoDetection (ConvertCocoPolysToMask) reads obj['area'] unconditionally; 1,924,260 ki67_NET train annotations
(66 images) lack 'area'/'iscrowd' -> DataLoader KeyError at P4 launch #1. val/test are complete.
The original bundle file is NOT modified; the derived file is written next to the bundle
(/home/work/.mnt/combined_all_v1_bundle_derived/) and referenced from det/configs/dataset/combined_tnt_detection.yml.
"""
import json, os, sys, time
src = sys.argv[1] if len(sys.argv) > 1 else "/home/work/.mnt/combined_all_v1_bundle/train_coco.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "/home/work/.mnt/combined_all_v1_bundle_derived/train_coco_areafix.json"
os.makedirs(os.path.dirname(dst), exist_ok=True)
t = time.time(); d = json.load(open(src)); print(f"loaded {src} in {time.time()-t:.0f}s")
n_area = n_crowd = 0
for a in d["annotations"]:
    if "area" not in a:
        w, h = a["bbox"][2], a["bbox"][3]; a["area"] = float(w) * float(h); n_area += 1
    if "iscrowd" not in a:
        a["iscrowd"] = 0; n_crowd += 1
d.setdefault("info", {}); d["info"]["derived_from"] = src
d["info"]["derived_note"] = f"area filled for {n_area} anns (bbox w*h), iscrowd=0 for {n_crowd} anns; images/categories/ids unchanged"
t = time.time(); json.dump(d, open(dst, "w")); print(f"wrote {dst} in {time.time()-t:.0f}s: filled area {n_area}, iscrowd {n_crowd}; anns {len(d['annotations'])} images {len(d['images'])}")
# verify
t = time.time(); v = json.load(open(dst))
assert len(v["annotations"]) == len(d["annotations"]) and len(v["images"]) == len(d["images"]) and v["categories"] == d["categories"]
assert all("area" in a and "iscrowd" in a for a in v["annotations"])
print(f"verified reload in {time.time()-t:.0f}s: all annotations have area/iscrowd")
with open(os.path.join(os.path.dirname(dst), "README.txt"), "a") as f:
    f.write(f"train_coco_areafix.json: derived from {src} on 2026-08-18 by det/scripts/fix_train_area.py; {d['info']['derived_note']}\n")
