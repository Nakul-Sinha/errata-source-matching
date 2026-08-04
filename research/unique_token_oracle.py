"""Oracle: pick title that uniquely owns a query-matched distinctive token."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import REF_COLS, STOP, build_query, distinctive, label_index, parse_ref, tokenize
import pandas as pd

train = pd.read_csv(r"G:\Datacurve\gpuchals\newone\dataset\public\train.csv")
train["label"] = train.apply(label_index, axis=1)
train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
va = train[train.src_year >= 2011].reset_index(drop=True)

unique_ok = coverage_ok = randomish = 0
unique_fire = 0
for _, row in va.iterrows():
    q = build_query(row, 2200).lower()
    gold = json.loads(row["provenance_card"])["source_title"]
    # map token -> titles containing it
    tok2titles = defaultdict(set)
    titles = set()
    for c in REF_COLS:
        t, _ = parse_ref(row[c])
        titles.add(t)
        for tok in distinctive(tokenize(t)):
            tok2titles[tok].add(t)
    # unique evidence tokens present in query
    votes = defaultdict(int)
    for tok, ts in tok2titles.items():
        if tok in q and len(ts) == 1:
            only = next(iter(ts))
            votes[only] += len(tok)
    if votes:
        unique_fire += 1
        pred = max(votes, key=votes.get)
        unique_ok += int(pred == gold)
    # fallback coverage
    cov = {}
    for t in titles:
        toks = distinctive(tokenize(t))
        hits = [x for x in toks if x in q]
        cov[t] = len(hits) / max(1, len(toks))
    coverage_ok += int(max(cov, key=cov.get) == gold)
    # combine: unique votes else coverage
    if votes:
        pred = max(votes, key=votes.get)
    else:
        pred = max(cov, key=cov.get)
    randomish += int(pred == gold)

print("unique_token_fires", unique_fire / len(va), "acc_when_fire", unique_ok / max(1, unique_fire))
print("coverage_always", coverage_ok / len(va))
print("unique_then_coverage", randomish / len(va))

# How often does gold have a unique token in query?
gold_unique = 0
for _, row in va.iterrows():
    q = build_query(row, 2200).lower()
    gold = json.loads(row["provenance_card"])["source_title"]
    tok2titles = defaultdict(set)
    for c in REF_COLS:
        t, _ = parse_ref(row[c])
        for tok in distinctive(tokenize(t)):
            tok2titles[tok].add(t)
    if any(tok in q and tok2titles[tok] == {gold} for tok in tok2titles):
        gold_unique += 1
print("gold_has_unique_token_in_query", gold_unique / len(va))
