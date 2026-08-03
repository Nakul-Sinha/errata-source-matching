import json
import sys
from pathlib import Path

slug = sys.argv[1] if len(sys.argv) > 1 else "errata-minilm-ft-val3"
out = Path(r"G:\Datacurve\kaggle_output") / slug
out.mkdir(parents=True, exist_ok=True)

# Force UTF-8 for kaggle's kernels_output writer on Windows.
import builtins

_real_open = builtins.open


def _utf8_open(file, mode="r", *args, **kwargs):
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
        kwargs.setdefault("errors", "replace")
    return _real_open(file, mode, *args, **kwargs)


builtins.open = _utf8_open

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.kernels_output(f"nakuls1nha/{slug}", str(out), quiet=True)
for p in sorted(out.iterdir()):
    print("FILE", p.name, p.stat().st_size)
    if p.suffix == ".log" and p.stat().st_size:
        txt = p.read_text(encoding="utf-8", errors="replace")
        try:
            rows = json.loads(txt)
            for r in rows:
                d = r.get("data", "")
                if any(
                    k in d
                    for k in (
                        "Error",
                        "Traceback",
                        "ARGS",
                        "device",
                        "BEST",
                        "val",
                        "CUDA",
                        "OOM",
                        "epoch",
                    )
                ):
                    print(d[:1000])
        except Exception:
            print(txt[:3000])
