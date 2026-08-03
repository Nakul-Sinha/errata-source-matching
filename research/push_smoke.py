import json
import traceback
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
root = Path(r"G:\Datacurve\kaggle_staging\errata-push-smoke")
root.mkdir(parents=True, exist_ok=True)
(root / "solution.py").write_text('print("hello")\n', encoding="utf-8")
meta = {
    "id": "nakuls1nha/errata-push-smoke",
    "title": "Errata Push Smoke",
    "code_file": "solution.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
(root / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
try:
    r = api.kernels_push(str(root))
    print("RESULT", r)
except Exception:
    traceback.print_exc()
