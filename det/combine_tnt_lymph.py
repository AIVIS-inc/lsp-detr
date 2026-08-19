#!/usr/bin/env python3
"""Combine a tumour/non-tumour .zst with a lymphocyte/others .zst into one 3-class .zst.

Port of the platform combiner ``TILs-Inference/combine_hagen_tnt_lympho.py`` (MODE ``tnt-lymph``)
for this repo's WSI output, so an LSP-DETR run (``det/wsi_infer.py`` -> tumour / non-tumour) can be
refined by a lymphocyte detector (Dome-TILs-512 -> lymphocyte / others) over the same slide.

Rule - anchored on the TNT cell set, so every TNT detection appears exactly once:
    nt == False (tumour)      -> TUMOUR, untouched
    nt == True  (non-tumour)  -> THE corresponding lympho cell = the *single nearest* lympho
                                 detection of ANY class within ``--radius`` px:
                                     it is a lymphocyte -> LYMPHOCYTE
                                     it is "others"     -> NON-TUMOUR
                                     nothing in radius  -> NON-TUMOUR
    This is a one-cell correspondence (nearest detection, then read its class), NOT "is there any
    lymphocyte nearby" - the latter lets one lymphocyte relabel several neighbours and inflates the
    lymphocyte count on dense slides (combine_hagen_tnt_lympho.py:32-35). ``--rule any-within``
    selects that other variant (what combine_aimedbio_m4lymph.py does: the KD-tree holds only the
    lymphocytes and any anchor within the radius is relabelled); on this data it roughly doubles the
    lymphocyte count, so it is not the default.

Output class encoding (identical to the platform's ``*_tnt-lymph.zst``):
    tumour      categoryId "1"  termId 66d54cd789181badfeac2d69  nt=False
    non-tumour  categoryId "2"  termId 66d54d2a89181badfeac2d75  nt=True
    lymphocyte  categoryId "3"  termId 66d54cee89181badfeac2d6d  nt=True
``nt`` is ``label != tumour``, so the viewer's T/NT mode shows a lymphocyte as non-tumour (which it
is) while its normal mode shows it as a lymphocyte.

Coordinates: both inputs are decoded to image space (y down from the top of the slide; the tile
stores ``height - y``), matched there, and re-encoded in the *lympho* frame. When the two inputs
carry different extents - e.g. a Dome-DETR file written with the mapbox default 4096 - the TNT y is
shifted by ``lym_extent - tnt_extent`` first, exactly like the platform combiner.

Usage:
    python det/combine_tnt_lymph.py --tnt results/129S.tif_LSPDETR-TNT.zst \\
        --lym results/129S.tif_Dome-512.zst --out results/129S.tif_LSP-lymph.zst
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# --- output class encoding (combine_hagen_tnt_lympho.py:71-81) --------------------------------
CLASS = {
    "tumor":      ("1", "66d54cd789181badfeac2d69"),
    "non_tumor":  ("2", "66d54d2a89181badfeac2d75"),
    "lymphocyte": ("3", "66d54cee89181badfeac2d6d"),
}
LABELS = ("tumor", "non_tumor", "lymphocyte")
LYM_LYMPH_CAT = "3"      # lympho model: categoryId '3' = lymphocyte, anything else = others


# --------------------------------------------------------------------------------------- read
def _value(v):
    for f in ("string_value", "float_value", "double_value",
              "int_value", "uint_value", "sint_value", "bool_value"):
        if v.HasField(f):
            return getattr(v, f)
    return None


def parse_layer(path: str, want_key: str, with_ids: bool = False):
    """(extent, xy[N,2] float64 in image space, prop[N][, ids[N], image_id[N]]) - protobuf into numpy.

    ``mapbox_vector_tile.decode`` would materialise every detection as a tree of Python dicts
    (~1.3 kB each, i.e. GBs for a WSI); this reads the same numbers directly. ``prop`` holds the
    value of ``want_key`` per feature (None where the feature does not carry it).

    Geometry: a Point feature is [MoveTo, dx, dy] with the cursor reset at every feature, and the
    stored y is ``extent - image_y`` (what ``decode`` undoes), so image_y = extent - stored.
    """
    import zstandard as zstd
    from mapbox_vector_tile.Mapbox import vector_tile_pb2 as pb

    tile = pb.tile()
    tile.ParseFromString(zstd.ZstdDecompressor().decompress(open(path, "rb").read()))
    if not tile.layers:
        raise ValueError(f"{path}: no layers")
    layer = tile.layers[0]
    keys = list(layer.keys)
    key_idx = keys.index(want_key) if want_key in keys else -1
    values = [_value(v) for v in layer.values]

    img_idx = keys.index("imageId") if "imageId" in keys else -1
    n = len(layer.features)
    xy = np.empty((n, 2), dtype=np.float64)
    prop = np.empty(n, dtype=object)
    ids = np.arange(n, dtype=np.int64)
    image_id = np.zeros(n, dtype=np.int64)
    unzig = lambda v: (v >> 1) ^ (-(v & 1))  # noqa: E731
    for i, feat in enumerate(layer.features):
        g = feat.geometry
        if len(g) < 3 or (g[0] & 0x7) != 1:
            raise ValueError(f"{path}: feature {i} is not a MoveTo point geometry")
        xy[i, 0] = unzig(g[1])
        xy[i, 1] = layer.extent - unzig(g[2])          # -> image space (y down from the top)
        tags = feat.tags
        for t in range(0, len(tags) - 1, 2):
            if tags[t] == key_idx:
                prop[i] = values[tags[t + 1]]
            elif with_ids and tags[t] == img_idx:
                image_id[i] = int(values[tags[t + 1]] or 0)
        if with_ids and feat.HasField("id"):
            ids[i] = int(feat.id)
    if with_ids:
        return int(layer.extent), xy, prop, ids, image_id
    return int(layer.extent), xy, prop


# --------------------------------------------------------------------------------------- write
def write_zst(path: str, xy: np.ndarray, lab_code: np.ndarray, extent: int,
              ids: "np.ndarray | None" = None, image_id: "np.ndarray | None" = None) -> None:
    """Same container as det/wsi_infer.py and the platform combiners.

    ``xy`` is in image space, so it goes in with ``y_coord_down=False`` - the encoder then stores
    ``extent - y``, which is the platform's ``height - centre_y``. (Handing it a pre-flipped y with
    this flag would flip twice and render the slide upside down.)
    """
    import mapbox_vector_tile as mvt
    import zstandard as zstd

    feats = []
    for i in range(len(xy)):
        cid, term = CLASS[LABELS[lab_code[i]]]
        # `nt` must be a real python bool: a numpy bool_ makes the encoder drop the key entirely.
        feats.append({
            "geometry": {"type": "Point", "coordinates": [int(round(xy[i, 0])), int(round(xy[i, 1]))]},
            "properties": {"imageId": int(image_id[i]) if image_id is not None else 0,
                           "categoryId": cid, "termId": term,
                           "nt": bool(lab_code[i] != 0), "positivity_rank": 0},
            "id": int(ids[i]) if ids is not None else int(i),
        })
    blob = mvt.encode({"name": "default", "features": feats},
                      default_options={"quantize_bounds": None, "y_coord_down": False,
                                       "extents": int(extent)})
    packed = zstd.ZstdCompressor().compress(blob)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(packed)
    print(f"[zst] {path} ({len(packed) / 2**20:.1f} MB, {len(xy):,} features, extent={extent})", flush=True)


def verify_zst(path: str, xy: np.ndarray, lab_code: np.ndarray, sample: int = 2000) -> None:
    """Decode what we just wrote and check it lands back on the same image-space points/classes."""
    import mapbox_vector_tile as mvt
    import zstandard as zstd

    layer = mvt.decode(zstd.ZstdDecompressor().decompress(open(path, "rb").read()))["default"]
    feats = layer["features"]
    if len(feats) != len(xy):
        raise RuntimeError(f"verify: {len(feats)} features decoded, {len(xy)} written")
    idx = np.unique(np.linspace(0, len(feats) - 1, min(sample, len(feats))).astype(np.int64))
    dx = np.array([feats[i]["geometry"]["coordinates"][0] for i in idx]) - xy[idx, 0]
    dy = np.array([feats[i]["geometry"]["coordinates"][1] for i in idx]) - xy[idx, 1]
    if np.abs(dx).max() > 1.5 or np.abs(dy).max() > 1.5:
        flipped = np.abs(np.array([feats[i]["geometry"]["coordinates"][1] for i in idx])
                         - (layer["extent"] - xy[idx, 1])).max() <= 1.5
        raise RuntimeError(f"verify FAILED: max |dx|={np.abs(dx).max():.1f} max |dy|={np.abs(dy).max():.1f}"
                           + (" - the y axis is flipped" if flipped else ""))
    bad = [i for i in idx if feats[i]["properties"]["categoryId"] != CLASS[LABELS[lab_code[i]]][0]
           or feats[i]["properties"]["termId"] != CLASS[LABELS[lab_code[i]]][1]
           or feats[i]["properties"]["nt"] != bool(lab_code[i] != 0)]
    if bad:
        raise RuntimeError(f"verify FAILED: {len(bad)} of {len(idx)} sampled features carry the wrong class")
    print(f"[zst] verify: {len(idx)} sampled features decode back to the written points and classes "
          f"(max |dx|={np.abs(dx).max():.2f}, max |dy|={np.abs(dy).max():.2f})", flush=True)


# --------------------------------------------------------------------------------------- combine
def combine(tnt_path: str, lym_path: str, radius: float, lymph_cat: str, kd_workers: int = -1,
            rule: str = "nearest"):
    """Returns (xy, lab_code, extent, ids, image_id, stats) in the lympho frame."""
    from scipy.spatial import cKDTree

    t0 = time.time()
    tnt_extent, tnt_xy, tnt_nt_raw, tnt_ids, tnt_img = parse_layer(tnt_path, "nt", with_ids=True)
    print(f"[tnt] {os.path.basename(tnt_path)}  {len(tnt_xy):,} features  extent={tnt_extent}  "
          f"({time.time() - t0:.1f}s)", flush=True)
    t0 = time.time()
    lym_extent, lym_xy, lym_cat = parse_layer(lym_path, "categoryId")
    print(f"[lym] {os.path.basename(lym_path)}  {len(lym_xy):,} features  extent={lym_extent}  "
          f"({time.time() - t0:.1f}s)", flush=True)

    if any(v is None for v in tnt_nt_raw):
        raise ValueError(f"{tnt_path}: some features carry no 'nt' property")
    tnt_nt = np.asarray([bool(v) for v in tnt_nt_raw], dtype=bool)
    lym_is_lymph = np.asarray([str(v) == str(lymph_cat) for v in lym_cat], dtype=bool)
    print(f"[lym] lymphocyte (categoryId {lymph_cat!r}) = {int(lym_is_lymph.sum()):,}, "
          f"others = {int((~lym_is_lymph).sum()):,}", flush=True)
    print(f"[tnt] tumour (nt=False) = {int((~tnt_nt).sum()):,}, non-tumour (nt=True) = {int(tnt_nt.sum()):,}",
          flush=True)

    # Output frame = the lympho frame; shift the TNT y when the extents differ (a Dome-DETR file
    # written with the mapbox default 4096 decodes to negative y, cf. normalize_domedetr_zst.py).
    extent = lym_extent
    y_shift = float(extent - tnt_extent)
    if y_shift:
        print(f"[frame] extents differ ({tnt_extent} vs {extent}) -> shifting the TNT y by {y_shift:+.0f}",
              flush=True)
        tnt_xy = np.column_stack([tnt_xy[:, 0], tnt_xy[:, 1] + y_shift])

    lab = np.where(tnt_nt, 1, 0).astype(np.int8)          # 0 tumour, 1 non-tumour, 2 lymphocyte
    stats = {"rule": rule, "n_tnt": int(len(tnt_xy)), "n_tnt_tumour": int((~tnt_nt).sum()),
             "n_tnt_non_tumour": int(tnt_nt.sum()), "n_lym": int(len(lym_xy)),
             "n_lym_lymph": int(lym_is_lymph.sum()), "radius": float(radius),
             "tnt_extent": tnt_extent, "lym_extent": lym_extent, "y_shift": y_shift}

    nt_idx = np.flatnonzero(tnt_nt)
    if len(nt_idx) and len(lym_xy):
        t0 = time.time()
        if rule == "nearest":
            # THE corresponding cell: nearest lympho detection of ANY class, then read its class.
            dist, nn = cKDTree(lym_xy).query(tnt_xy[nt_idx], k=1, workers=kd_workers)
            hit = (dist <= radius) & lym_is_lymph[nn]
            claimed = nn[hit]
        else:
            # any-within: nearest LYMPHOCYTE only -> one lymphocyte may relabel many anchors.
            lym_only = np.flatnonzero(lym_is_lymph)
            dist, nn_l = cKDTree(lym_xy[lym_only]).query(tnt_xy[nt_idx], k=1, workers=kd_workers)
            hit = dist <= radius
            claimed = lym_only[nn_l[hit]]
        lab[nt_idx[hit]] = 2
        stats.update({
            "match_sec": round(time.time() - t0, 1),
            "nn_median_px": float(np.median(dist)), "nn_p90_px": float(np.percentile(dist, 90)),
            "nn_p99_px": float(np.percentile(dist, 99)),
            "matched_within_radius": int((dist <= radius).sum()),
            "matched_frac": float((dist <= radius).mean()),
            "relabelled_lymphocyte": int(hit.sum()),
            "distinct_lymphocytes_claimed": int(len(np.unique(claimed))),
            "duplicate_claims": int(len(claimed) - len(np.unique(claimed))),
        })
        what = "lympho detection" if rule == "nearest" else "lymphocyte"
        print(f"[match] rule={rule}: nearest {what} for {len(nt_idx):,} non-tumour cells: "
              f"median {stats['nn_median_px']:.2f} px, p90 {stats['nn_p90_px']:.1f}, "
              f"p99 {stats['nn_p99_px']:.1f}; {stats['matched_frac'] * 100:.1f}% within {radius:g} px "
              f"({stats['match_sec']}s)", flush=True)
        print(f"[match] -> lymphocyte {stats['relabelled_lymphocyte']:,} "
              f"(from {stats['distinct_lymphocytes_claimed']:,} distinct lympho detections, "
              f"{stats['duplicate_claims']:,} duplicate claims)", flush=True)

    counts = {LABELS[c]: int((lab == c).sum()) for c in range(3)}
    stats["class_totals"] = counts
    oor = int(((tnt_xy[:, 1] < 0) | (tnt_xy[:, 1] > extent)).sum()) if len(tnt_xy) else 0
    stats["outside_extent"] = oor
    if oor:
        print(f"[warn] {oor:,} anchor cells fall outside [0, {extent}] after the shift", flush=True)
    return tnt_xy, lab, extent, tnt_ids, tnt_img, stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tnt", required=True, help="tumour/non-tumour .zst (the anchor cell set)")
    p.add_argument("--lym", required=True, help="lymphocyte/others .zst")
    p.add_argument("--out", required=True, help="output .zst")
    p.add_argument("--radius", type=float, default=30.0, help="match radius in level-0 px (default 30)")
    p.add_argument("--rule", choices=["nearest", "any-within"], default="nearest",
                   help="nearest (default, canonical): the single nearest lympho detection of any class "
                        "decides; any-within: relabel whenever a lymphocyte is within --radius (inflates "
                        "the count - this is what the platform's 129S.tif_tnt-lymph.zst was built with, "
                        "at --radius 16)")
    p.add_argument("--lymph-cat", default=LYM_LYMPH_CAT,
                   help=f"categoryId meaning lymphocyte in --lym (default {LYM_LYMPH_CAT})")
    p.add_argument("--kd-workers", type=int, default=-1, help="threads for the KD-tree query (-1 = all)")
    p.add_argument("--stats-json", default=None, help="also write the run statistics to this path")
    p.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    args = p.parse_args(argv)

    for f in (args.tnt, args.lym):
        if not os.path.isfile(f):
            sys.exit(f"not a file: {f}")

    t_start = time.time()
    xy, lab, extent, ids, image_id, stats = combine(args.tnt, args.lym, args.radius, args.lymph_cat,
                                                    args.kd_workers, args.rule)
    write_zst(args.out, xy, lab, extent, ids, image_id)
    if args.verify:
        verify_zst(args.out, xy, lab)

    total = max(1, len(lab))
    print("[done] " + "  ".join(f"{k}={v:,} ({v / total * 100:.1f}%)" for k, v in stats["class_totals"].items())
          + f"  in {time.time() - t_start:.1f}s", flush=True)
    if args.stats_json:
        stats.update({"tnt": os.path.abspath(args.tnt), "lym": os.path.abspath(args.lym),
                      "out": os.path.abspath(args.out), "lymph_cat": args.lymph_cat,
                      "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "cli": " ".join(sys.argv)})
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[stats] {args.stats_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
