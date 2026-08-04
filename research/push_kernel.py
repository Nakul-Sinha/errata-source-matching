"""Push a single-file GPU kernel that reads the Kaggle dataset."""
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
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    out = STAGING / args.slug
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Embed train_ce.py into solution.py so Kaggle only needs one code file.
    train_src = (ROOT / "research" / "train_ce.py").read_text(encoding="utf-8")
    # Drop future import + __main__ guard; wrapper owns the file header.
    train_src = train_src.replace("from __future__ import annotations\n", "")
    marker = '\nif __name__ == "__main__":\n'
    if marker in train_src:
        train_src = train_src.split(marker, 1)[0]
    solution = f'''\
from __future__ import annotations
import glob, sys, shutil
from pathlib import Path

hits = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True), key=len)
assert hits, "train.csv not found under /kaggle/input"
DATA = str(Path(hits[0]).parent)
OUT = "/kaggle/working/ce_out"
sys.argv = [
    "train_ce.py",
    "--data", DATA,
    "--out", OUT,
    "--mode", "{args.mode}",
    "--model", "{args.model}",
    "--epochs", "{args.epochs}",
    "--batch-size", "{args.batch_size}",
    "--grad-accum", "{args.grad_accum}",
    "--max-length", "{args.max_length}",
    "--max-chars", "1600",
    "--lr", "{args.lr}",
    "--val-year", "2011",
]
print("ARGS", sys.argv, flush=True)

{train_src}

main()
sub = Path(OUT) / "submission.csv"
if sub.exists():
    shutil.copy2(sub, Path("/kaggle/working/submission.csv"))
    print("copied submission", flush=True)
'''
    (out / "solution.py").write_text(solution, encoding="utf-8")
    meta = {
        "id": f"nakuls1nha/{args.slug}",
        "title": args.slug.replace("-", " ").title(),
        "code_file": "solution.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        # Avoid P100 (sm_60) — current Kaggle PyTorch wheels omit that arch.
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    api = KaggleApi()
    api.authenticate()
    result = api.kernels_push(str(out))
    print(result)


if __name__ == "__main__":
    main()
