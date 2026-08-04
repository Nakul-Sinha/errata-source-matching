"""Compare random-split vs time-split ceilings with a quick MiniLM fine-tune."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, r"G:\ml\gpuchals\newone\research")
from train_ce import (
    CrossEncoder,
    PairDataset,
    label_index,
    make_collate,
    seed_everything,
)

DATA = r"G:\ml\gpuchals\newone\dataset\public"
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def run(tr, va, tag, epochs=3, lr=5e-5):
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = CrossEncoder(MODEL).to(device)
    collate = make_collate(tok, 320)
    dl = DataLoader(
        PairDataset(tr, tr.label.tolist(), 1400),
        batch_size=2,
        shuffle=True,
        collate_fn=collate,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = len(dl) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for enc, labels, _ in dl:
            bsz = enc["input_ids"].size(0)
            flat = {k: v.view(bsz * 16, -1).to(device) for k, v in enc.items()}
            labels = labels.to(device)
            loss = loss_fn(
                model(flat["input_ids"], flat["attention_mask"]).view(bsz, 16), labels
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            tot += float(loss)
        print(f"{tag} epoch {ep+1} loss={tot/len(dl):.4f}", flush=True)

    @torch.no_grad()
    def acc(df):
        model.eval()
        correct = 0
        loader = DataLoader(
            PairDataset(df, df.label.tolist(), 1400), batch_size=4, collate_fn=collate
        )
        for enc, labels, _ in loader:
            bsz = enc["input_ids"].size(0)
            flat = {k: v.view(bsz * 16, -1).to(device) for k, v in enc.items()}
            pred = (
                model(flat["input_ids"], flat["attention_mask"])
                .view(bsz, 16)
                .argmax(-1)
                .cpu()
            )
            correct += int((pred == labels).sum())
        return correct / len(df)

    print(f"{tag} train_acc={acc(tr):.4f} val_acc={acc(va):.4f}", flush=True)


def main():
    seed_everything(0)
    train = pd.read_csv(f"{DATA}/train.csv")
    train["label"] = train.apply(label_index, axis=1)
    train["src_year"] = train["provenance_card"].apply(
        lambda s: int(json.loads(s)["source_year"])
    )

    # subsample for speed
    tr_time = train[train.src_year < 2011].sample(800, random_state=0).reset_index(drop=True)
    va_time = train[train.src_year >= 2011].sample(300, random_state=0).reset_index(drop=True)
    run(tr_time, va_time, "TIME")

    rng = train.sample(1100, random_state=1).reset_index(drop=True)
    tr_rand = rng.iloc[:800].reset_index(drop=True)
    va_rand = rng.iloc[800:].reset_index(drop=True)
    run(tr_rand, va_rand, "RAND")


if __name__ == "__main__":
    main()
