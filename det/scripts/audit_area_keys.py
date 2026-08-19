"""Audit missing 'area'/'iscrowd' keys in the bundle annotation files (Dome's CocoDetection requires obj['area'])."""
import json, time, collections, sys
splits = sys.argv[1:] or ["train", "val", "test"]
for split in splits:
    t = time.time(); d = json.load(open(f"/home/work/.mnt/combined_all_v1_bundle/{split}_coco.json"))
    n = len(d["annotations"]); no_area = sum(1 for a in d["annotations"] if "area" not in a)
    no_crowd = sum(1 for a in d["annotations"] if "iscrowd" not in a)
    bad_area = sum(1 for a in d["annotations"] if "area" in a and (a["area"] is None or a["area"] <= 0))
    img_src = {im["id"]: im["file_name"].split("/")[1] for im in d["images"]}
    by_src = collections.Counter(img_src[a["image_id"]] for a in d["annotations"] if "area" not in a)
    imgs = len({a["image_id"] for a in d["annotations"] if "area" not in a})
    print(split, "anns", n, "no_area", no_area, "imgs_with_no_area_anns", imgs, "no_iscrowd", no_crowd, "bad_area", bad_area,
          "by_source", dict(by_src), f"{time.time()-t:.0f}s", flush=True)
