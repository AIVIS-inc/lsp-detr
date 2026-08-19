"""P2: build the model from the resolved config (CPU) and audit optimizer param groups (§7).
Prints group name lists (--names) and asserts single assignment / LR / WD policy / frozen set."""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lsp_det  # noqa: F401
from src.core import YAMLConfig
from lsp_det.optim_audit import audit_param_groups

p = argparse.ArgumentParser()
p.add_argument("--config", default=os.path.join(lsp_det.DET_ROOT, "configs", "LSP-T-combined.yml"))
p.add_argument("--names", action="store_true")
p.add_argument("-u", "--update", nargs="+", default=[])
a = p.parse_args()

from src.core import yaml_utils
cfg = YAMLConfig(a.config, **yaml_utils.parse_cli(a.update))
model = cfg.model
groups = audit_param_groups(model, cfg.yaml_cfg["optimizer"], verbose=True, print_names=a.names,
                            expect={"backbone_lr": 1.25e-5, "min_groups": 3,
                                    "frozen_prefixes": ["backbone.embeddings.", "backbone.encoder.layers.0."]})
# extra checks specific to this port
names_by_group = {g["idx"]: set(g["names"]) for g in groups}
bb_norm = [k for k, _ in model.named_parameters() if "backbone" in k and "norm" in k and dict(model.named_parameters())[k].requires_grad]
assert all(k in names_by_group[1] for k in bb_norm), "backbone norms not all in group[1]"
dec_norm = [k for k, v in model.named_parameters() if "backbone" not in k and "norm" in k]
assert all(k in names_by_group[2] for k in dec_norm), "non-backbone norms not all in group[2]"
assert "feature_sampling.norm.weight" in names_by_group[2]
assert "decoder.class_head.bias" in names_by_group[3] and "decoder.wh_head.0.4.weight" in names_by_group[3]
print(f"backbone norm tensors at lr 1.25e-5/wd 0: {len(bb_norm)}; non-backbone norm tensors at wd 0: {len(dec_norm)}")
print("PASS optimizer group audit")
