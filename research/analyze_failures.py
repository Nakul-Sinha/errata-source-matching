"""Analyze when distinctive title tokens are present but ranking fails."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import (
    REF_COLS,
    STOP,
    build_query,
    format_candidate,
    label_index,
    overlap_features,
    parse_ref,
    tokenize,
    title_group_ids,
    decode_prediction,
)
import torch

train = pd.read_csv(r"G:\ml\gpuchals\newone\dataset\public\train.csv")
train["label"] = train.apply(label_index, axis=1)
train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
va = train[train.src_year >= 2011].reset_index(drop=True)

present = absent = 0
present_correct = absent_correct = 0
oracle_maxfeat = 0
for _, row in va.iterrows():
    q = build_query(row, 1800)
    ql = q.lower()
    card = json.loads(row["provenance_card"])
    gold_title = card["source_title"]
    toks = [t for t in tokenize(gold_title) if len(t) >= 5 and t not in STOP]
    hit = any(t in ql for t in toks)
    feats = np.array([overlap_features(q, format_candidate(str(row[c]))) for c in REF_COLS])
    # score = length-weighted distinctive channel (index 4) + substring (3)
    scores = feats[:, 3] * 5 + feats[:, 4] * 5 + feats[:, 2] * 3
    groups = title_group_ids([str(row[c]) for c in REF_COLS])
    pred = int(decode_prediction(torch.tensor(scores)[None], torch.tensor(groups)[None])[0])
    pred_title = parse_ref(row[REF_COLS[pred]])[0]
    ok = pred_title == gold_title
    # oracle: is gold title the unique max on substring feature among groups?
    # collapse by title max feature
    title_best = {}
    for i, c in enumerate(REF_COLS):
        t, _ = parse_ref(row[c])
        title_best[t] = max(title_best.get(t, -1e9), float(scores[i]))
    oracle = max(title_best, key=title_best.get) == gold_title
    oracle_maxfeat += int(oracle)
    if hit:
        present += 1
        present_correct += int(ok)
    else:
        absent += 1
        absent_correct += int(ok)

print("distinctive token present in query", present, "acc", present_correct / max(1, present))
print("absent", absent, "acc", absent_correct / max(1, absent))
print("oracle title by max feat score", oracle_maxfeat / len(va))
print("overall feat decode title acc", (present_correct + absent_correct) / len(va))
