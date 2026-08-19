"""CPU tests for det/combine_tnt_lymph.py (no model, no GPU, no slide).

Builds two tiny synthetic .zst inputs in the platform's storage convention and checks the
TNT-anchored rule, the class encoding of the output, and the fact that the default rule reads the
class of THE nearest lympho detection rather than asking "is any lymphocyte nearby".
"""
import importlib.util
import os

import numpy as np
import pytest

zstd = pytest.importorskip("zstandard")
mvt = pytest.importorskip("mapbox_vector_tile")
pytest.importorskip("scipy")
from mapbox_vector_tile.Mapbox import vector_tile_pb2 as pb  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lsp_combine", os.path.join(_HERE, "..", "combine_tnt_lymph.py"))
comb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comb)

H = 1000  # tile extent == slide height


def _write_input(path, points, extent=H):
    """points: [(x, y_image, categoryId, nt)] -> a .zst in the platform convention (stores H - y)."""
    feats = [{"geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
              "properties": {"imageId": 0, "categoryId": cid, "termId": "t" * 24, "nt": bool(nt),
                             "positivity_rank": 0},
              "id": i}
             for i, (x, y, cid, nt) in enumerate(points)]
    blob = mvt.encode({"name": "default", "features": feats},
                      default_options={"quantize_bounds": None, "y_coord_down": False, "extents": extent})
    with open(path, "wb") as fp:
        fp.write(zstd.ZstdCompressor().compress(blob))


def _read_output(path):
    layer = mvt.decode(zstd.ZstdDecompressor().decompress(open(path, "rb").read()))["default"]
    return layer["extent"], layer["features"]


# anchors: (x, y, categoryId, nt)   nt=False -> tumour
TNT = [
    (100, 100, "1", False),   # 0 tumour, sitting right on a lymphocyte -> must stay tumour
    (200, 200, "5", True),    # 1 non-tumour, nearest lympho cell is a lymphocyte -> lymphocyte
    (300, 300, "5", True),    # 2 non-tumour, nearest is "others" (a lymphocyte is further but < radius)
    (400, 400, "5", True),    # 3 non-tumour, nothing within the radius -> stays non-tumour
]
LYM = [
    (100, 100, "3"),          # lymphocyte under the tumour anchor
    (202, 200, "3"),          # 2 px from anchor 1
    (301, 300, "2"),          # 1 px from anchor 2  -> "others" wins the nearest test
    (310, 300, "3"),          # 10 px from anchor 2 -> only "any-within" would catch it
    (480, 400, "3"),          # 80 px from anchor 3 -> outside a 30 px radius
]


def _run(tmp_path, rule="nearest", radius=30.0):
    tnt, lym, out = (str(tmp_path / f) for f in ("tnt.zst", "lym.zst", "out.zst"))
    _write_input(tnt, TNT)
    _write_input(lym, [(x, y, c, False) for x, y, c in LYM])
    xy, lab, extent, ids, img, stats = comb.combine(tnt, lym, radius, comb.LYM_LYMPH_CAT, 1, rule)
    comb.write_zst(out, xy, lab, extent, ids, img)
    return out, lab, stats


def test_nearest_rule_labels(tmp_path):
    out, lab, stats = _run(tmp_path)
    assert list(lab) == [0, 2, 1, 1]          # tumour, lymphocyte, non-tumour (nearest is "others"), non-tumour
    assert stats["class_totals"] == {"tumor": 1, "non_tumor": 2, "lymphocyte": 1}
    assert stats["y_shift"] == 0


def test_any_within_rule_differs(tmp_path):
    """The inflating variant also catches anchor 2, whose nearest neighbour is an "others" cell."""
    _, lab, _ = _run(tmp_path, rule="any-within")
    assert list(lab) == [0, 2, 2, 1]


def test_radius_gate(tmp_path):
    _, lab, _ = _run(tmp_path, radius=1.0)     # only anchor 2's "others" neighbour is within 1 px
    assert list(lab) == [0, 1, 1, 1]
    _, lab, _ = _run(tmp_path, rule="any-within", radius=100.0)
    assert list(lab) == [0, 2, 2, 2]           # now even the 80 px lymphocyte counts


def test_output_class_encoding_and_frame(tmp_path):
    out, lab, _ = _run(tmp_path)
    extent, feats = _read_output(out)
    assert extent == H and len(feats) == len(TNT)
    want = {0: ("1", "66d54cd789181badfeac2d69", False),      # LABELS[0] tumour
            1: ("2", "66d54d2a89181badfeac2d75", True),       # LABELS[1] non-tumour
            2: ("3", "66d54cee89181badfeac2d6d", True)}       # LABELS[2] lymphocyte
    for f, code in zip(feats, lab):
        cid, term, nt = want[int(code)]
        assert (f["properties"]["categoryId"], f["properties"]["termId"], f["properties"]["nt"]) == (cid, term, nt)
        assert f["properties"]["imageId"] == 0 and f["properties"]["positivity_rank"] == 0
    # anchors keep their position, and the tile stores extent - y like every platform writer
    assert [f["geometry"]["coordinates"] for f in feats] == [[x, y] for x, y, _, _ in TNT]
    tile = pb.tile()
    tile.ParseFromString(zstd.ZstdDecompressor().decompress(open(out, "rb").read()))
    unzig = lambda v: (v >> 1) ^ (-(v & 1))  # noqa: E731
    stored = [(unzig(f.geometry[1]), unzig(f.geometry[2])) for f in tile.layers[0].features]
    assert stored == [(x, H - y) for x, y, _, _ in TNT]
    assert list(tile.layers[0].keys) == ["imageId", "categoryId", "termId", "nt", "positivity_rank"]


def test_extent_mismatch_shifts_the_anchor_set(tmp_path):
    """A TNT file written with the mapbox default extent (4096) must be shifted into the lympho frame."""
    tnt, lym, out = (str(tmp_path / f) for f in ("tnt2.zst", "lym2.zst", "out2.zst"))
    _write_input(tnt, [(200, 200 - (H - 4096), "5", True)], extent=4096)   # same slide point, 4096 frame
    _write_input(lym, [(202, 200, "3", False)])
    xy, lab, extent, ids, img, stats = comb.combine(tnt, lym, 30.0, comb.LYM_LYMPH_CAT, 1)
    assert extent == H and stats["y_shift"] == H - 4096
    assert list(lab) == [2]                                  # lands on the lymphocyte after the shift
    comb.write_zst(out, xy, lab, extent, ids, img)
    _, feats = _read_output(out)
    assert feats[0]["geometry"]["coordinates"] == [200, 200]


def test_parse_layer_matches_mapbox_decode(tmp_path):
    tnt = str(tmp_path / "p.zst")
    _write_input(tnt, TNT)
    extent, xy, prop, ids, img = comb.parse_layer(tnt, "nt", with_ids=True)
    _, feats = _read_output(tnt)
    assert extent == H
    assert np.array_equal(xy, np.array([f["geometry"]["coordinates"] for f in feats], dtype=float))
    assert [bool(v) for v in prop] == [f["properties"]["nt"] for f in feats]
    assert list(ids) == [f["id"] for f in feats] and list(img) == [0] * len(TNT)
