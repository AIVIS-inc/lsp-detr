"""Create small COCO subsets (from val_coco.json, all 1536^2) for the DDP/EMA/val smoke (P3-4).
Writes det/logs/smoke_data/{train,val}_subset.json with file_name relative to the bundle root."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lsp_det  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--src", default="/home/work/.mnt/combined_all_v1_bundle/val_coco.json")
p.add_argument("--n-train", type=int, default=24)
p.add_argument("--n-val", type=int, default=16)
p.add_argument("--out-dir", default=os.path.join(lsp_det.DET_ROOT, "logs", "smoke_data"))
a = p.parse_args()

with open(a.src) as f:
    d = json.load(f)
by_img = {}
for an in d["annotations"]:
    by_img.setdefault(an["image_id"], []).append(an)
imgs = [im for im in d["images"] if 100 < len(by_img.get(im["id"], [])) < 2000]
rng = np.random.RandomState(1)
pick = rng.choice(len(imgs), size=a.n_train + a.n_val, replace=False)
tr = [imgs[i] for i in pick[: a.n_train]]
va = [imgs[i] for i in pick[a.n_train:]]
# include one image with very few boxes -> after centre filtering it can be empty in training? (val GT all centre) - keep as is
os.makedirs(a.out_dir, exist_ok=True)
for name, ims in (("train", tr), ("val", va)):
    ids = {im["id"] for im in ims}
    sub = {"images": ims, "annotations": [an for an in d["annotations"] if an["image_id"] in ids], "categories": d["categories"]}
    for k in ("info", "licenses"):
        if k in d:
            sub[k] = d[k]
    path = os.path.join(a.out_dir, f"{name}_subset.json")
    with open(path, "w") as f:
        json.dump(sub, f)
    print(f"{path}: {len(ims)} images, {len(sub['annotations'])} anns, boxes/img={[len(by_img[im['id']]) for im in ims]}")
