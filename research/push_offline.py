"""Push offline (no-internet) GPU kernel using solution.py + dataset."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(r"G:\ml\gpuchals\newone")
STAGING = Path(r"G:\ml\kaggle_staging")
DATASET = "nakuls1nha/errata-provenance-cards-data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--mode", choices=["validate", "full"], default="validate")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    out = STAGING / args.slug
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    src = (ROOT / "solution.py").read_text(encoding="utf-8")
    # Strip future import placement issues by keeping solution as code_file directly,
    # with a tiny argv bootstrap prepended carefully.
    bootstrap = f'''from __future__ import annotations
import glob, sys
from pathlib import Path

hits = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True), key=len)
assert hits, "train.csv missing"
DATA = str(Path(hits[0]).parent)
OUT = "/kaggle/working/submission.csv"
sys.argv = ["solution.py", DATA, OUT, "--mode", "{args.mode}", "--epochs", "{args.epochs}", "--time-limit", "3200"]
print("ARGS", sys.argv, flush=True)

'''
    # Remove the solution's own future import to keep it at file top only.
    body = src.replace("from __future__ import annotations\n", "")
    # Ensure main runs (solution has __main__ guard; __name__ will be __main__).
    (out / "solution.py").write_text(bootstrap + body, encoding="utf-8")

    meta = {
        "id": f"nakuls1nha/{args.slug}",
        "title": args.slug.replace("-", " ").title(),
        "code_file": "solution.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    api = KaggleApi()
    api.authenticate()
    print(api.kernels_push(str(out)))


if __name__ == "__main__":
    main()
