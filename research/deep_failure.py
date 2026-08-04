"""Deeper failure analysis + simple learned-token-importance ceiling."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
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
    format_candidate,
    label_index,
    parse_ref,
    tokenize,
    title_group_ids,
    decode_prediction,
    metric_from_preds,
)

DATA = Path(r"G:\ml\gpuchals\newone\dataset\public")
train = pd.read_csv(DATA / "train.csv")
train["label"] = train.apply(label_index, axis=1)
train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
tr = train[train.src_year < 2011].reset_index(drop=True)
va = train[train.src_year >= 2011].reset_index(drop=True)


def distinctive(title: str) -> list[str]:
    return [t for t in tokenize(title) if len(t) >= 4 and t not in STOP]


# Coverage oracle: pick title maximizing fraction of distinctive tokens found in query
def coverage_score(query: str, title: str) -> float:
    toks = distinctive(title)
    if not toks:
        return 0.0
    ql = query.lower()
    hits = sum(1 for t in toks if t in ql)
    # length-weighted
    w_hits = sum(len(t) for t in toks if t in ql)
    w_all = sum(len(t) for t in toks)
    return hits / len(toks) + 0.5 * (w_hits / max(1, w_all))


correct = 0
present_n = present_ok = 0
for _, row in va.iterrows():
    q = build_query(row, 1800)
    card = json.loads(row["provenance_card"])
    gold = card["source_title"]
    best_t, best_s = None, -1.0
    titles = {}
    for c in REF_COLS:
        t, y = parse_ref(row[c])
        titles[t] = max(titles.get(t, -1.0), coverage_score(q, t))
    pred = max(titles, key=titles.get)
    ok = pred == gold
    correct += int(ok)
    toks = distinctive(gold)
    hit = any(t in q.lower() for t in toks)
    if hit:
        present_n += 1
        present_ok += int(ok)

print("coverage oracle title_rate", correct / len(va))
print("coverage present acc", present_ok / max(1, present_n), "n", present_n)

# Learned token IDF-like weights FROM LABELS only (not classic TF-IDF vectorizer):
# For each token, P(token in gold title's distinctive set | token in query) vs distractors.
# Actually: learn log-odds that a candidate token matching query indicates gold.
token_pos = Counter()
token_neg = Counter()
for _, row in tr.iterrows():
    q = build_query(row, 1800)
    ql = set(tokenize(q))
    gold_i = int(row["label"])
    for i, c in enumerate(REF_COLS):
        t, _ = parse_ref(row[c])
        for tok in distinctive(t):
            if tok in ql:
                if i == gold_i:
                    token_pos[tok] += 1
                else:
                    token_neg[tok] += 1


def token_weight(tok: str) -> float:
    p = token_pos[tok] + 0.5
    n = token_neg[tok] + 0.5
    # higher when more associated with gold than distractor matches
    return math.log(p / n) if False else float(np.log(p) - np.log(n))


import math

weights = {t: token_weight(t) for t in set(token_pos) | set(token_neg)}
print("num weighted tokens", len(weights), "top", sorted(weights, key=weights.get, reverse=True)[:15])


def weighted_score(query: str, title: str) -> float:
    ql = set(tokenize(query))
    s = 0.0
    for tok in distinctive(title):
        if tok in ql or tok in query.lower():
            s += max(0.1, weights.get(tok, 0.5)) * (len(tok) / 5.0)
    return s


correct = 0
scores_all = []
for _, row in va.iterrows():
    q = build_query(row, 1800)
    card = json.loads(row["provenance_card"])
    gold = card["source_title"]
    titles = {}
    for c in REF_COLS:
        t, _ = parse_ref(row[c])
        titles[t] = max(titles.get(t, -1e9), weighted_score(q, t))
    pred = max(titles, key=titles.get)
    correct += int(pred == gold)
print("label-derived token weight title_rate", correct / len(va))

# Neural: small MLP on expanded handcrafted feats + learned token match score
# Also try: for each candidate, vector of [coverage, weighted, char4, acr, ...]


def feat_vec(query: str, raw: str) -> np.ndarray:
    title, year = parse_ref(raw)
    cand = format_candidate(raw)
    q_toks = set(tokenize(query))
    c_toks = distinctive(title)
    ql = query.lower()
    hits = [t for t in c_toks if t in ql]
    acr = set(re.findall(r"\b[A-Z]{2,8}\b", title))
    acr = {a.lower() for a in acr}
    acr_hits = len(acr & q_toks)
    qg = {ql[i : i + 4] for i in range(max(0, len(ql) - 3))}
    cl = title.lower()
    cg = {cl[i : i + 4] for i in range(max(0, len(cl) - 3))}
    return np.array(
        [
            len(hits) / max(1, len(c_toks)),
            sum(len(t) for t in hits) / max(1, sum(len(t) for t in c_toks)),
            weighted_score(query, title),
            len(qg & cg) / max(1, len(cg)),
            acr_hits / max(1, len(acr)),
            float(bool(hits)),
            float(len(hits) >= 2),
            float(year in query),
            math.log1p(len(hits)),
            math.log1p(sum(len(t) for t in hits)),
        ],
        dtype=np.float32,
    )


def build_xy(df):
    X, y, groups = [], [], []
    for _, row in df.iterrows():
        q = build_query(row, 1800)
        raws = [str(row[c]) for c in REF_COLS]
        X.append([feat_vec(q, r) for r in raws])
        y.append(int(row["label"]))
        groups.append(title_group_ids(raws))
    return np.asarray(X, np.float32), np.asarray(y, np.int64), groups


Xtr, ytr, Gtr = build_xy(tr)
Xva, yva, Gva = build_xy(va)
print("X", Xtr.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(
    nn.Linear(Xtr.shape[-1], 64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 1),
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
Xt, yt = torch.tensor(Xtr, device=device), torch.tensor(ytr, device=device)
Xv, yv = torch.tensor(Xva, device=device), torch.tensor(yva, device=device)

best = 0.0
for epoch in range(60):
    model.train()
    perm = torch.randperm(len(Xt), device=device)
    total = 0.0
    for i in range(0, len(Xt), 64):
        idx = perm[i : i + 64]
        scores = model(Xt[idx]).squeeze(-1)
        # title group loss
        loss = 0.0
        for b in range(scores.size(0)):
            gi = int(idx[b].item())
            g = torch.tensor(Gtr[gi], device=device)
            # logsumexp by group
            max_g = int(g.max()) + 1
            gs = scores.new_full((max_g,), -1e4)
            for u in g.unique():
                gs[int(u)] = torch.logsumexp(scores[b][g == u], 0)
            # title label
            lab = int(g[yt[idx[b]]].item())
            loss = loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([lab], device=device))
        loss = loss / scores.size(0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss)
    model.eval()
    with torch.no_grad():
        scores = model(Xv).squeeze(-1).cpu()
        preds = []
        for i in range(len(Xv)):
            g = torch.tensor(Gva[i])
            pred = int(decode_prediction(scores[i].unsqueeze(0), g.unsqueeze(0))[0])
            preds.append(pred)
        preds = np.array(preds)
        # title rate
        title_ok = 0
        for i, (_, row) in enumerate(va.iterrows()):
            card = json.loads(row["provenance_card"])
            t, _ = parse_ref(row[REF_COLS[preds[i]]])
            title_ok += int(t == card["source_title"])
        tr_rate = title_ok / len(va)
        both = float((preds == yva).mean())
        score = 0.85 * tr_rate + 0.05 * both + 0.10 * 0.0  # rough
        if tr_rate > best:
            best = tr_rate
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1} title={tr_rate:.4f} both={both:.4f} best_title={best:.4f}", flush=True)

print("FINAL best title_rate", best)
