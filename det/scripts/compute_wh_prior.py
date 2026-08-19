"""P0: compute per-axis bbox median (w, h) from train_coco.json -> wh prior for the LSP log-wh head.

Usage: python det/scripts/compute_wh_prior.py [--ann /path/train_coco.json] [--out det/configs/wh_prior.json]
The strategy doc (§2, §4.1) forbids log(7.5); the prior must be the train bbox per-axis median.
"""
import argparse, json, math, os, sys, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--ann", default="/home/work/.mnt/combined_all_v1_bundle/train_coco.json")
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "configs", "wh_prior.json"))
a = p.parse_args()

t0 = time.time()
with open(a.ann) as f:
    d = json.load(f)
print(f"loaded {a.ann} in {time.time()-t0:.1f}s: images={len(d['images'])} anns={len(d['annotations'])} cats={d['categories']}")

w = np.fromiter((x["bbox"][2] for x in d["annotations"]), dtype=np.float64, count=len(d["annotations"]))
h = np.fromiter((x["bbox"][3] for x in d["annotations"]), dtype=np.float64, count=len(d["annotations"]))
cat = np.fromiter((x["category_id"] for x in d["annotations"]), dtype=np.int64, count=len(d["annotations"]))
img_of = np.fromiter((x["image_id"] for x in d["annotations"]), dtype=np.int64, count=len(d["annotations"]))

def q(x):
    return {k: float(np.percentile(x, v)) for k, v in [("p05", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95)]} | {"mean": float(x.mean())}

stats = {
    "ann_file": a.ann,
    "num_images": len(d["images"]),
    "num_annotations": int(len(w)),
    "median_w": float(np.median(w)),
    "median_h": float(np.median(h)),
    "log_median_w": float(math.log(np.median(w))),
    "log_median_h": float(math.log(np.median(h))),
    "w": q(w),
    "h": q(h),
    "per_class": {int(c): {"n": int((cat == c).sum()), "median_w": float(np.median(w[cat == c])), "median_h": float(np.median(h[cat == c]))} for c in np.unique(cat)},
}
# per-source (file_name prefix train/<source>/...)
src_of_img = {im["id"]: im["file_name"].split("/")[1] if "/" in im["file_name"] else "?" for im in d["images"]}
srcs = np.array([src_of_img[i] for i in img_of])
stats["per_source"] = {}
for s in np.unique(srcs):
    m = srcs == s
    stats["per_source"][str(s)] = {"n": int(m.sum()), "median_w": float(np.median(w[m])), "median_h": float(np.median(h[m]))}
imgs_with_ann = set(img_of.tolist())
stats["images_without_annotations"] = int(len(d["images"]) - len(imgs_with_ann))
sizes = {}
for im in d["images"]:
    k = f"{im['width']}x{im['height']}"
    sizes[k] = sizes.get(k, 0) + 1
stats["image_sizes"] = sizes

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps({k: v for k, v in stats.items() if k not in ("per_source",)}, indent=2))
print("per_source:", json.dumps(stats["per_source"], indent=1))
print(f"wrote {a.out}")
