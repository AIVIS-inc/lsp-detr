#!/usr/bin/env python
"""Fetch the hf-5class initial checkpoint (``hf-5class/model.safetensors``) used by every det/ run.

The weights are the RationAI/LSP-DETR Hugging Face release at revision
a32176184ec548279cff263e767c2d710ca31e7c (2025-07-09) - the revision whose ``config.json`` / ``modeling.py`` /
``preprocessor_config.json`` are the files tracked in ``hf-5class/``. The hub replaced ``model.safetensors``
on 2025-08-20 (sha256 99b0d385..., 180,178,896 bytes); that newer file is NOT what the P4/HER2 runs were
initialised from, so the revision is pinned and the download is verified against the original checksum.

Usage:
    python det/scripts/fetch_hf5class.py            # -> <repo>/hf-5class/model.safetensors (180 MB, public repo, no token)
    python det/scripts/fetch_hf5class.py --dest /elsewhere/model.safetensors
"""
import argparse
import hashlib
import os
import shutil
import sys

REPO_ID = "RationAI/LSP-DETR"
REVISION = "a32176184ec548279cff263e767c2d710ca31e7c"
FILENAME = "model.safetensors"
SHA256 = "3f5437eb889a864ff88ae121ed7581217778b30430657ce39751a7f5e4b96082"
SIZE = 180151024

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "hf-5class", FILENAME)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=DEFAULT_DEST, help=f"where to put the file (default {DEFAULT_DEST})")
    p.add_argument("--force", action="store_true", help="re-download even if --dest exists")
    a = p.parse_args()

    if os.path.isfile(a.dest) and not a.force:
        got = sha256_of(a.dest)
        if got == SHA256:
            print(f"[fetch_hf5class] OK, already present and verified: {a.dest}")
            return 0
        print(f"[fetch_hf5class] {a.dest} exists but sha256 {got[:16]}... != expected {SHA256[:16]}... "
              f"(size {os.path.getsize(a.dest):,} vs {SIZE:,}); re-run with --force to replace it", file=sys.stderr)
        return 2

    from huggingface_hub import hf_hub_download  # pulled in by transformers

    print(f"[fetch_hf5class] downloading {REPO_ID}@{REVISION[:10]}:{FILENAME} ...")
    cached = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=REVISION)
    os.makedirs(os.path.dirname(os.path.abspath(a.dest)), exist_ok=True)
    tmp = a.dest + ".part"
    shutil.copyfile(cached, tmp)   # a real copy: the repo directory must not depend on the hub cache
    got = sha256_of(tmp)
    if got != SHA256 or os.path.getsize(tmp) != SIZE:
        os.remove(tmp)
        print(f"[fetch_hf5class] checksum mismatch after download: sha256 {got} size {os.path.getsize(cached):,} "
              f"(expected {SHA256} / {SIZE:,}); nothing written", file=sys.stderr)
        return 1
    os.replace(tmp, a.dest)
    print(f"[fetch_hf5class] OK: {a.dest} ({SIZE:,} bytes, sha256 {SHA256[:16]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
