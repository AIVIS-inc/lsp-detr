"""§5.2: CayleySTRING eval path must reflect the *current* S (no stale cached P)."""
import math

import torch

from lsp_det.lsp_trunk import CayleySTRING


def _snapshot_cayley():
    """Rebuild the HF snapshot variant (cached_property P) for comparison."""
    from functools import cached_property

    class SnapshotCayley(CayleySTRING):
        @cached_property
        def P(self):  # noqa: D401 - snapshot semantics
            return torch.matmul(self.I - self.S, torch.linalg.inv(self.I + self.S))

        def forward(self, x, positions):
            if self.training:
                return CayleySTRING.forward(self, x, positions)
            px = (x.float() @ self.P.T).contiguous()
            from einops import rearrange
            angles = torch.einsum("bnk,khc->bhnc", positions, self.freqs)
            freqs_cis = torch.polar(torch.ones_like(angles), angles)
            px_ = torch.view_as_complex(rearrange(px, "... (d two) -> ... d two", two=2))
            out = rearrange(torch.view_as_real(px_ * freqs_cis), "... d two -> ... (d two)")
            return out.type_as(x)

    return SnapshotCayley


def _run(pe, x, pos):
    return pe(x, pos)


def test_P_is_orthogonal_and_matches_train_path():
    torch.manual_seed(0)
    pe = CayleySTRING(dim=384, num_heads=12)
    P = pe.P()
    assert torch.allclose(P @ P.T, torch.eye(32), atol=1e-5)
    x = torch.randn(2, 12, 9, 32)
    pos = torch.rand(2, 9, 2) * 100
    pe.train(); y_tr = pe(x, pos)
    pe.eval(); y_ev = pe(x, pos)
    assert torch.allclose(y_tr, y_ev, atol=1e-4), (y_tr - y_ev).abs().max()


def test_eval_reflects_updated_S_after_optimizer_step():
    torch.manual_seed(0)
    x = torch.randn(2, 12, 9, 32)
    pos = torch.rand(2, 9, 2) * 100

    pe = CayleySTRING(dim=384, num_heads=12)
    Snap = _snapshot_cayley()
    snap = Snap(dim=384, num_heads=12)
    snap.load_state_dict(pe.state_dict())

    # first eval pass (this is where the snapshot caches P)
    pe.eval(); snap.eval()
    y0 = pe(x, pos); s0 = snap(x, pos)
    assert torch.allclose(y0, s0, atol=1e-4)  # elementwise vs einsum angle: float32 noise only

    # train steps that change S (same update applied to both modules)
    opt = torch.optim.SGD(pe.parameters(), lr=1.0)
    opt_s = torch.optim.SGD(snap.parameters(), lr=1.0)
    for _ in range(20):
        for m, o in ((pe, opt), (snap, opt_s)):
            m.train()
            o.zero_grad()
            m(x, pos).pow(2).sum().backward()
            o.step()
    delta = torch.randn(32, 32, generator=torch.Generator().manual_seed(1)) * 0.3
    with torch.no_grad():
        pe.parametrizations.S.original.add_(delta)
        snap.parametrizations.S.original.add_(delta)
    assert (pe.parametrizations.S.original - snap.parametrizations.S.original).abs().max() < 1e-5

    # second eval pass: ported module must reflect the new S; snapshot still uses the stale cached P
    pe.eval(); snap.eval()
    y1 = pe(x, pos); s1 = snap(x, pos)
    pe.train(); y1_ref = pe(x, pos)  # train path == ground truth for current S
    assert torch.allclose(y1, y1_ref, atol=1e-4), "ported eval path stale"
    assert not torch.allclose(s1, y1_ref, atol=1e-3), "snapshot should be stale here (sanity of the test)"
    assert not torch.allclose(y1, y0, atol=1e-3), "S did not change enough to be a meaningful test"


def test_state_dict_keys_match_checkpoint_layout():
    pe = CayleySTRING(dim=384, num_heads=12)
    keys = sorted(pe.state_dict().keys())
    assert keys == ["freqs", "parametrizations.S.original"], keys
    assert tuple(pe.state_dict()["freqs"].shape) == (2, 12, 16)
    assert tuple(pe.state_dict()["parametrizations.S.original"].shape) == (32, 32)
