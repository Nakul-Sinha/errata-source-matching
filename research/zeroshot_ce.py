"""Zero-shot / light-finetune probe with a pretrained MS MARCO cross-encoder."""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
DATA = r"G:\Datacurve\gpuchals\newone\dataset\public"
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def parse_ref(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def build_query(row, max_chars=1400):
    note = str(row.get("submitter_note") or "").strip()
    corr = str(row.get("proposed_correction") or "").strip()
    excerpt = str(row.get("original_excerpt") or "").strip()
    head = f"NOTE: {note}\nCORRECTION: {corr}\nEXCERPT: "
    budget = max(200, max_chars - len(head))
    if len(excerpt) > budget:
        keep = budget // 2
        excerpt = excerpt[:keep] + " ... " + excerpt[-keep:]
    return head + excerpt


def label_index(row):
    card = json.loads(row["provenance_card"])
    for i, col in enumerate(REF_COLS):
        t, y = parse_ref(row[col])
        if t == card["source_title"] and y == card["source_year"]:
            return i
    raise ValueError(row["id"])


def score(title_ok, year_ok, conf):
    title_rate = float(title_ok.mean())
    year_rate = float(year_ok.mean())
    y = title_ok.astype(np.float64)
    p = np.clip(conf / 100.0, 0, 1)
    brier = float(((p - y) ** 2).mean())
    base = float(y.mean())
    ref = base * (1 - base) if 0 < base < 1 else (1 / 16) * (15 / 16)
    cal = float(np.clip(1 - brier / ref, 0, 1))
    return {
        "title": title_rate,
        "year": year_rate,
        "cal": cal,
        "score": 0.85 * title_rate + 0.05 * year_rate + 0.10 * cal,
    }


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_csv(f"{DATA}/train.csv")
    train["label"] = train.apply(label_index, axis=1)
    train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
    va = train[train["src_year"] >= 2011].reset_index(drop=True)
    print("val", len(va), "device", device, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()

    all_logits = []
    with torch.no_grad():
        for i, row in va.iterrows():
            q = build_query(row)
            cands = [str(row[c]) for c in REF_COLS]
            enc = tok(
                [q] * 16,
                cands,
                truncation=True,
                padding=True,
                max_length=320,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            all_logits.append(logits)
            if (i + 1) % 50 == 0:
                print(f"scored {i+1}/{len(va)} {time.time()-t0:.0f}s", flush=True)

    logits = np.stack(all_logits, 0)
    # temperature sweep
    best = None
    y = va["label"].to_numpy()
    for t in np.linspace(0.2, 5.0, 40):
        probs = torch.softmax(torch.tensor(logits) / t, -1).numpy()
        pred = probs.argmax(1)
        conf = probs.max(1) * 100
        title_ok = (pred == y).astype(np.int32)
        year_ok = np.zeros(len(va), np.int32)
        for i, (_, row) in enumerate(va.iterrows()):
            card = json.loads(row["provenance_card"])
            _, year = parse_ref(row[REF_COLS[pred[i]]])
            year_ok[i] = int(year == card["source_year"])
        m = score(title_ok, year_ok, conf)
        m["t"] = float(t)
        if best is None or m["score"] > best["score"]:
            best = m
    print("BEST zero-shot", best, flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
