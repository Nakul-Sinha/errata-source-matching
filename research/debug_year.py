"""Inspect title/year match inconsistency after a short fine-tune."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, r"G:\ml\gpuchals\newone\research")
from train_ce import (
    REF_COLS,
    CrossEncoder,
    PairDataset,
    label_index,
    make_collate,
    parse_ref,
    seed_everything,
)

seed_everything(0)
device = "cuda"
train = pd.read_csv(r"G:\ml\gpuchals\newone\dataset\public\train.csv")
train["label"] = train.apply(label_index, axis=1)
train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
tr = train[train.src_year < 2011].sample(400, random_state=0).reset_index(drop=True)
va = train[train.src_year >= 2011].reset_index(drop=True)

tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").to(device)
collate = make_collate(tok, 256)
dl = DataLoader(
    PairDataset(tr, tr.label.tolist(), 1200),
    batch_size=2,
    shuffle=True,
    collate_fn=collate,
)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()
model.train()
for epoch in range(3):
    total = 0.0
    for enc, labels, _ in dl:
        bsz = enc["input_ids"].size(0)
        flat = {k: v.view(bsz * 16, -1).to(device) for k, v in enc.items()}
        labels = labels.to(device)
        logits = model(flat["input_ids"], flat["attention_mask"]).view(bsz, 16)
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss)
    print(f"epoch {epoch+1} loss={total/len(dl):.4f}", flush=True)

model.eval()
loader = DataLoader(
    PairDataset(va, va.label.tolist(), 1200), batch_size=4, collate_fn=collate
)
preds = []
with torch.no_grad():
    for enc, labels, _ in loader:
        bsz = enc["input_ids"].size(0)
        flat = {k: v.view(bsz * 16, -1).to(device) for k, v in enc.items()}
        pred = model(flat["input_ids"], flat["attention_mask"]).view(bsz, 16).argmax(-1).cpu().numpy()
        preds.extend(pred.tolist())

preds = np.array(preds)
labels = va["label"].to_numpy()
print("label_acc", float((preds == labels).mean()))

title_ok = year_ok = both = 0
mismatch_examples = []
for i, (_, row) in enumerate(va.iterrows()):
    card = json.loads(row["provenance_card"])
    title, year = parse_ref(row[REF_COLS[preds[i]]])
    t = title == card["source_title"]
    y = year == card["source_year"]
    title_ok += int(t)
    year_ok += int(y)
    both += int(t and y)
    if t and not y and len(mismatch_examples) < 5:
        mismatch_examples.append((card, title, year, row[REF_COLS[preds[i]]], row[REF_COLS[labels[i]]]))
    if (not t) and y and len(mismatch_examples) < 5:
        mismatch_examples.append(("year_only", card, title, year))

print("title", title_ok / len(va), "year", year_ok / len(va), "both", both / len(va))
print("mismatch samples:")
for m in mismatch_examples:
    print(m)

# How often do multiple candidates share the same title?
share = 0
for _, row in va.iterrows():
    titles = [parse_ref(row[c])[0] for c in REF_COLS]
    if len(titles) != len(set(titles)):
        share += 1
print("rows with duplicate titles among 16", share, "/", len(va))
