"""Difficulty probe: TF-IDF and token-overlap baselines on a time holdout."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train = pd.read_csv(r"G:\ml\gpuchals\newone\dataset\public\train.csv")
REFS = [c for c in train.columns if c.startswith("reference_title")]


def parse_ref(s: str) -> tuple[str, str]:
    m = re.match(r"^(.*) \((\d{4})\)$", str(s))
    return (m.group(1), m.group(2)) if m else (str(s), "")


def query(row) -> str:
    parts = [
        str(row["original_excerpt"] or ""),
        str(row["proposed_correction"] or ""),
        str(row["submitter_note"] or ""),
    ]
    return " ".join(parts)


def label_idx(row) -> int:
    card = json.loads(row["provenance_card"])
    for i, c in enumerate(REFS):
        t, y = parse_ref(row[c])
        if t == card["source_title"] and y == card["source_year"]:
            return i
    return -1


def metric(title_rate: float, year_rate: float, confs: np.ndarray, ys: np.ndarray) -> dict:
    p = confs / 100.0
    brier = float(((p - ys) ** 2).mean())
    base = float(ys.mean())
    ref = base * (1 - base) if 0 < base < 1 else (1 / 16) * (15 / 16)
    cal = float(np.clip(1 - brier / ref, 0, 1))
    score = 0.85 * title_rate + 0.05 * year_rate + 0.10 * cal
    return dict(title=title_rate, year=year_rate, cal=cal, score=score)


train["label"] = train.apply(label_idx, axis=1)
train["year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
tr = train[train["year"] <= 2010].reset_index(drop=True)
va = train[train["year"] >= 2011].reset_index(drop=True)
print("split", len(tr), len(va))


def score_tfidf(fit_df, eval_df):
    docs = []
    for _, r in fit_df.iterrows():
        docs.append(query(r))
        for c in REFS:
            docs.append(str(r[c]))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80_000)
    vec.fit(docs)
    correct = year_ok = 0
    confs, ys = [], []
    for _, r in eval_df.iterrows():
        q = vec.transform([query(r)])
        cands = [str(r[c]) for c in REFS]
        sims = cosine_similarity(q, vec.transform(cands))[0]
        pred = int(sims.argmax())
        lab = int(r["label"])
        y = int(pred == lab)
        correct += y
        _, py = parse_ref(cands[pred])
        card = json.loads(r["provenance_card"])
        year_ok += int(py == card["source_year"])
        x = sims - sims.max()
        e = np.exp(x * 10)
        p = e / e.sum()
        confs.append(float(p[pred] * 100))
        ys.append(y)
    n = len(eval_df)
    out = metric(correct / n, year_ok / n, np.asarray(confs), np.asarray(ys, dtype=float))
    out["n"] = n
    return out


print("tfidf timehold", score_tfidf(tr, va))


def tok(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z]{3,}|\d{4}|[A-Z]{2,}", s))


correct = 0
for _, r in va.iterrows():
    q = tok(query(r))
    scores = [len(q & tok(str(r[c]))) for c in REFS]
    pred = int(np.argmax(scores))
    correct += pred == r["label"]
print("token overlap val", correct / len(va))

hits = 0
for _, r in va.iterrows():
    card = json.loads(r["provenance_card"])
    title_toks = [
        w
        for w in re.findall(r"[A-Za-z]{4,}", card["source_title"])
        if w.lower()
        not in {
            "protocol",
            "specification",
            "version",
            "internet",
            "network",
            "format",
            "message",
        }
    ]
    q = query(r).lower()
    hits += any(w.lower() in q for w in title_toks[:8])
print("any distinctive title token in query", hits / len(va))
print("chance title", 1 / 16)
