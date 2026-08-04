"""Sanity: can the CE overfit 32 labeled rows?"""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_ce import (
    REF_COLS,
    CrossEncoder,
    PairDataset,
    build_query,
    label_index,
    make_collate,
    parse_ref,
)

DATA = r"G:\ml\gpuchals\newone\dataset\public"
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main():
    device = "cuda"
    train = pd.read_csv(f"{DATA}/train.csv")
    train["label"] = train.apply(label_index, axis=1)
    df = train.sample(32, random_state=0).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = CrossEncoder(MODEL).to(device)
    collate = make_collate(tok, 256)
    loader = DataLoader(
        PairDataset(df, df["label"].tolist(), 1000),
        batch_size=2,
        shuffle=True,
        collate_fn=collate,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(20):
        model.train()
        total = 0.0
        for enc, labels, _ in loader:
            bsz = enc["input_ids"].size(0)
            flat = {k: v.view(bsz * 16, -1).to(device) for k, v in enc.items()}
            labels = labels.to(device)
            logits = model(flat["input_ids"], flat["attention_mask"]).view(bsz, 16)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        # acc
        model.eval()
        correct = 0
        with torch.no_grad():
            for enc, labels, _ in DataLoader(
                PairDataset(df, df["label"].tolist(), 1000),
                batch_size=4,
                collate_fn=collate,
            ):
                bsz = enc["input_ids"].size(0)
                flat = {k: v.view(bsz * 16, -1).to(device) for k, v in enc.items()}
                pred = model(flat["input_ids"], flat["attention_mask"]).view(bsz, 16).argmax(-1).cpu()
                correct += int((pred == labels).sum())
        print(
            f"epoch {epoch+1} loss={total/len(loader):.4f} acc={correct/len(df):.3f} "
            f"t={time.time()-t0:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
