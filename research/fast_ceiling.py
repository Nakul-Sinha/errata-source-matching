"""Fast ceilings: coverage oracle + supervised token-weight feature MLP."""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import (
    REF_COLS,
    STOP,
    build_query,
    decode_prediction,
    distinctive,
    label_index,
    metric_from_preds,
    overlap_features,
    parse_ref,
    title_group_ids,
    tokenize,
)

DATA = Path(r"G:\Datacurve\gpuchals\newone\dataset\public")
train = pd.read_csv(DATA / "train.csv")
train["label"] = train.apply(label_index, axis=1)
train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
tr = train[train.src_year < 2011].reset_index(drop=True)
va = train[train.src_year >= 2011].reset_index(drop=True)
print("split", len(tr), len(va), flush=True)


def coverage(query: str, title: str) -> float:
    toks = distinctive(tokenize(title))
    if not toks:
        return 0.0
    ql = query.lower()
    hits = [t for t in toks if t in ql]
    return len(hits) / len(toks) + 0.5 * sum(len(t) for t in hits) / max(1, sum(len(t) for t in toks))


# coverage oracle
ok = 0
for _, row in va.iterrows():
    q = build_query(row, 2200)
    gold = json.loads(row["provenance_card"])["source_title"]
    best = {}
    for c in REF_COLS:
        t, _ = parse_ref(row[c])
        best[t] = max(best.get(t, -1.0), coverage(q, t))
    ok += int(max(best, key=best.get) == gold)
print("coverage_oracle_title", ok / len(va), flush=True)

# supervised token weights from train labels (gold vs distractor match counts)
pos, neg = Counter(), Counter()
for _, row in tr.iterrows():
    qset = set(tokenize(build_query(row, 2200)))
    gi = int(row["label"])
    for i, c in enumerate(REF_COLS):
        t, _ = parse_ref(row[c])
        for tok in distinctive(tokenize(t)):
            if tok in qset:
                (pos if i == gi else neg)[tok] += 1


def tw(tok: str) -> float:
    return float(math.log(pos[tok] + 0.5) - math.log(neg[tok] + 0.5))


def wscore(query: str, title: str) -> float:
    ql = query.lower()
    s = 0.0
    for tok in distinctive(tokenize(title)):
        if tok in ql:
            s += max(0.05, tw(tok)) * (len(tok) / 5.0)
    return s


ok = 0
for _, row in va.iterrows():
    q = build_query(row, 2200)
    gold = json.loads(row["provenance_card"])["source_title"]
    best = {}
    for c in REF_COLS:
        t, _ = parse_ref(row[c])
        best[t] = max(best.get(t, -1e9), wscore(q, t) + 0.3 * coverage(q, t))
    ok += int(max(best, key=best.get) == gold)
print("supervised_token_weight_title", ok / len(va), flush=True)


def row_feats(row):
    q = build_query(row, 2200)
    xs = []
    for c in REF_COLS:
        raw = str(row[c])
        t, y = parse_ref(raw)
        base = overlap_features(q, raw)
        xs.append(base + [wscore(q, t), coverage(q, t)])
    return np.asarray(xs, dtype=np.float32)


print("building matrices...", flush=True)
Xtr = np.stack([row_feats(r) for _, r in tr.iterrows()])
ytr = tr["label"].to_numpy()
Gtr = [title_group_ids([str(r[c]) for c in REF_COLS]) for _, r in tr.iterrows()]
Xva = np.stack([row_feats(r) for _, r in va.iterrows()])
yva = va["label"].to_numpy()
Gva = [title_group_ids([str(r[c]) for c in REF_COLS]) for _, r in va.iterrows()]
print("X", Xtr.shape, flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(
    nn.Linear(Xtr.shape[-1], 96),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(96, 48),
    nn.GELU(),
    nn.Linear(48, 1),
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
Xt, yt = torch.tensor(Xtr, device=device), torch.tensor(ytr, device=device)
Xv = torch.tensor(Xva, device=device)

best_title = 0.0
best_score = 0.0
for epoch in range(80):
    model.train()
    perm = torch.randperm(len(Xt), device=device)
    for i in range(0, len(Xt), 128):
        idx = perm[i : i + 128]
        scores = model(Xt[idx]).squeeze(-1)
        loss = scores.new_zeros(())
        for b in range(scores.size(0)):
            gi = int(idx[b])
            g = torch.tensor(Gtr[gi], device=device)
            max_g = int(g.max()) + 1
            gs = scores.new_full((max_g,), -1e4)
            for u in g.unique():
                gs[int(u)] = torch.logsumexp(scores[b][g == u], 0)
            lab = int(g[yt[idx[b]]].item())
            loss = loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([lab], device=device))
        loss = loss / scores.size(0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if (epoch + 1) % 5 == 0 or epoch < 3:
        model.eval()
        with torch.no_grad():
            scores = model(Xv).squeeze(-1).cpu()
            preds = []
            for i in range(len(Xv)):
                g = torch.tensor(Gva[i])
                preds.append(int(decode_prediction(scores[i : i + 1], g.unsqueeze(0))[0]))
            pred = np.array(preds)
            # fake flat conf for metric; then calibrate roughly by margin
            conf = np.full(len(pred), 50.0)
            # better conf: softmax of title groups
            confs = []
            for i in range(len(pred)):
                g = torch.tensor(Gva[i])
                # rebuild group scores
                from solution import title_group_scores

                gs = title_group_scores(scores[i : i + 1], g.unsqueeze(0))
                p = F.softmax(gs, dim=-1)[0, Gva[i][pred[i]]].item() * 100
                confs.append(p)
            conf = np.array(confs)
            m = metric_from_preds(pred, conf, va)
            best_title = max(best_title, m["title_rate"])
            best_score = max(best_score, m["score"])
            print(f"epoch {epoch+1} {m} best_score={best_score:.4f}", flush=True)

print("DONE best_title", best_title, "best_score", best_score, flush=True)
