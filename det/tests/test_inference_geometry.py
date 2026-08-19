"""CPU tests for det/inference.py tiling geometry (no model, no GPU)."""
import importlib.util
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lsp_inference", os.path.join(_HERE, "..", "inference.py"))
inf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inf)

T, S, F = 1536, 672, 672
M = (T - F) // 2


def _ownership(size):
    return [(o + M, o + M + F) for o in inf.tile_origins(size, T, S, F)]


def test_grid_covers_and_partitions():
    for size in [1, 50, 100, 512, 672, 673, 1000, 1024, 1536, 1537, 2048, 3375, 4096, 5000]:
        regs = _ownership(size)
        assert all(regs[i][1] == regs[i + 1][0] for i in range(len(regs) - 1)), size  # contiguous, no overlap
        assert regs[0][0] <= 0 and regs[-1][1] >= size, size                            # covers [0, size)
        assert abs((-regs[0][0]) - (regs[-1][1] - size)) <= 1, size                      # symmetric margins


def test_grid_matches_training_placement():
    # an image <= 672 is one tile placed like Dome's _handle_small_image (symmetric pad to 1536)
    assert inf.tile_origins(512, T, S, F) == [-(T - 512) // 2]
    assert inf.tile_origins(672, T, S, F) == [-M]
    # 1536 -> 3 tiles, centre tile owns exactly the centre 672
    assert _ownership(1536)[1] == (432, 1104)
    # single (tiling off): centred, identity for 1536
    assert inf.tile_origins(1536, T, S, T, single=True) == [0]
    assert inf.tile_origins(512, T, S, T, single=True) == [-512]


def test_reflect_pad_large_amounts_keep_content():
    x = torch.arange(3 * 5 * 7, dtype=torch.float32).view(1, 3, 5, 7)
    y = inf.reflect_pad(x, 20, 33, 9, 40)
    assert y.shape == (1, 3, 5 + 9 + 40, 7 + 20 + 33)
    assert torch.equal(y[:, :, 9:14, 20:27], x)
    z = inf.reflect_pad(torch.rand(1, 3, 1, 4), 3, 3, 3, 3)  # 1-px axis -> replicate fallback, no infinite loop
    assert z.shape == (1, 3, 7, 10)


def test_collect_inputs_dedup_and_limit(tmp_path):
    from PIL import Image
    for n in ["a.png", "b.png"]:
        Image.new("RGB", (8, 8)).save(tmp_path / n)
    lst = tmp_path / "l.txt"
    lst.write_text(f"{tmp_path/'a.png'}\n{tmp_path/'a.png'}\n{tmp_path/'b.png'}\n")
    args = inf.parse_args(["-r", "x.pth", "-i", str(lst), "-o", str(tmp_path)])
    items, root, _ = inf.collect_inputs(args)
    assert [os.path.basename(i["path"]) for i in items] == ["a.png", "b.png"]
    args = inf.parse_args(["-r", "x.pth", "-i", str(tmp_path), "-o", str(tmp_path), "--limit", "1"])
    items, root, _ = inf.collect_inputs(args)
    assert len(items) == 1 and root == str(tmp_path)
