"""Probe stateless overlap-feature ranking on the time holdout (diagnostic only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import (
    REF_COLS,
    build_query,
    label_index,
    overlap_features,
    format_candidate,
    parse_ref,
)

train = pd.read_csv(r"G:\Datacurve\gpuchals\newone\dataset\public\train.csv")
train["label"] = train.apply(label_index, axis=1)
train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
tr = train[train.src_year < 2011].reset_index(drop=True)
va = train[train.src_year >= 2011].reset_index(drop=True)


def matrix(df):
    X, y = [], []
    for _, row in df.iterrows():
        q = build_query(row, 1800)
        feats = [overlap_features(q, format_candidate(str(row[c]))) for c in REF_COLS]
        X.append(feats)
        y.append(int(row["label"]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


Xtr, ytr = matrix(tr)
Xva, yva = matrix(va)
print("shapes", Xtr.shape, Xva.shape)

# greedy: argmax of content-overlap feature index 2
pred = Xva[:, :, 2].argmax(1)
print("feat2 argmax acc", float((pred == yva).mean()))
pred = Xva.sum(-1).argmax(1)
print("feat sum argmax acc", float((pred == yva).mean()))

# learned linear listwise on features only
device = "cuda" if torch.cuda.is_available() else "cpu"
W = nn.Linear(Xtr.shape[-1], 1).to(device)
opt = torch.optim.AdamW(W.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
Xt = torch.tensor(Xtr, device=device)
yt = torch.tensor(ytr, device=device)
Xv = torch.tensor(Xva, device=device)
yv = torch.tensor(yva, device=device)
for epoch in range(40):
    W.train()
    # mini-batches
    perm = torch.randperm(len(Xt), device=device)
    total = 0.0
    for i in range(0, len(Xt), 64):
        idx = perm[i : i + 64]
        scores = W(Xt[idx]).squeeze(-1)
        loss = loss_fn(scores, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss)
    if (epoch + 1) % 10 == 0:
        W.eval()
        with torch.no_grad():
            pred = W(Xv).squeeze(-1).argmax(-1)
            acc = float((pred == yv).mean())
        print(f"epoch {epoch+1} loss={total:.3f} val_acc={acc:.4f}", flush=True)
