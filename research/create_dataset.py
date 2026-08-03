"""Create the private Kaggle dataset for this challenge."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(r"G:\Datacurve\kaggle_staging\errata-provenance-data")
ROOT.mkdir(parents=True, exist_ok=True)
src = Path(r"G:\Datacurve\gpuchals\newone\dataset\public")
for name in ("train.csv", "test.csv", "sample_submission.csv"):
    (ROOT / name).write_bytes((src / name).read_bytes())

meta = {
    "title": "Errata Provenance Cards Data (Private)",
    "id": "nakuls1nha/errata-provenance-cards-data",
    "licenses": [{"name": "other"}],
}
(ROOT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

api = KaggleApi()
api.authenticate()
try:
    print(api.dataset_create_new(str(ROOT), public=False, quiet=False, dir_mode="zip"))
except Exception:
    traceback.print_exc()
    print("create failed; trying version on existing...")
    try:
        print(api.dataset_create_version(str(ROOT), version_notes="init", dir_mode="zip"))
    except Exception:
        traceback.print_exc()
