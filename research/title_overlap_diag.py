"""Diagnose title/token overlap across time split and retrieval oracle."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "dataset" / "public"
REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
TOKEN_RE = re.compile(r"[A-Za-z]{3,}|\d{4}")
STOP = set(
    "year protocol specification version internet network format message the and for "
    "with from that this standard requirements framework profile into over".split()
)


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def toks(s):
    return [t for t in TOKEN_RE.findall(str(s).lower()) if t not in STOP]


def build_query(row):
    return f"{row['submitter_note']}\n{row['proposed_correction']}\n{row['original_excerpt']}"


def main():
    train = pd.read_csv(DATA / "train.csv")
    train["card"] = train["provenance_card"].apply(json.loads)
    train["gold_title"] = train["card"].apply(lambda c: c["source_title"])
    train["gold_year"] = train["card"].apply(lambda c: str(c["source_year"]))
    train["src_year"] = train["gold_year"].astype(int)
    train["query"] = train.apply(build_query, axis=1)

    tr = train[train.src_year < 2011]
    va = train[train.src_year >= 2011]
    print("split", len(tr), len(va))

    tr_titles = set(tr.gold_title)
    va_titles = set(va.gold_title)
    print("unique titles train/val/overlap", len(tr_titles), len(va_titles), len(tr_titles & va_titles))
    print("val gold title seen in train", va.gold_title.isin(tr_titles).mean())

    # candidate title in val that appears as train gold
    seen = 0
    for _, row in va.iterrows():
        cands = [parse(row[c])[0] for c in REF_COLS]
        if any(t in tr_titles for t in cands):
            seen += 1
    print("val rows with any cand title in train golds", seen / len(va))

    # gold title in candidates that was train gold
    gold_in_train = va.gold_title.isin(tr_titles).mean()
    print("val gold title was train gold", gold_in_train)

    # Token Jaccard retrieval oracle: retrieve top-k train queries, vote gold titles among candidates
    tr_toks = [set(toks(q)) for q in tr["query"]]
    tr_gold = tr["gold_title"].tolist()
    hits = []
    for _, row in va.iterrows():
        qt = set(toks(row["query"]))
        if not qt:
            hits.append(0)
            continue
        scores = []
        for i, tt in enumerate(tr_toks):
            inter = len(qt & tt)
            if inter == 0:
                scores.append(0.0)
            else:
                scores.append(inter / math_sqrt(len(qt) * len(tt)))
        top = np.argsort(scores)[-32:][::-1]
        votes = Counter(tr_gold[i] for i in top if scores[i] > 0)
        cands = [parse(row[c])[0] for c in REF_COLS]
        # pick cand with most votes
        best = None
        best_v = -1
        for t in cands:
            v = votes.get(t, 0)
            if v > best_v:
                best_v = v
                best = t
        # also try: among cands, max overlap with retrieved titles' tokens
        hits.append(int(best == row["gold_title"]))
    print("jaccard retrieval vote title acc", np.mean(hits))

    # Phrase presence oracle: if any 2+ consecutive distinctive title tokens appear as phrase
    phrase_hit = 0
    phrase_fire = 0
    for _, row in va.iterrows():
        q = row["query"].lower()
        gold = row["gold_title"]
        gt = toks(gold)
        found = False
        for n in (3, 2):
            for i in range(0, max(0, len(gt) - n + 1)):
                phrase = " ".join(gt[i : i + n])
                if len(phrase) >= 8 and phrase in q:
                    found = True
                    break
            if found:
                break
        if found:
            phrase_fire += 1
            # would we pick gold among cands with phrase?
            scores = []
            for c in REF_COLS:
                t, _ = parse(row[c])
                ct = toks(t)
                sc = 0
                for n in (3, 2):
                    for i in range(0, max(0, len(ct) - n + 1)):
                        phrase = " ".join(ct[i : i + n])
                        if len(phrase) >= 8 and phrase in q:
                            sc += n
                scores.append(sc)
            pred = [parse(row[c])[0] for c in REF_COLS][int(np.argmax(scores))]
            phrase_hit += int(pred == gold)
    print("phrase fire rate", phrase_fire / len(va), "acc when fire", phrase_hit / max(1, phrase_fire))

    # Distinctive token IDF from train titles only, score cands
    dfreq = Counter()
    for t in tr_titles:
        dfreq.update(set(toks(t)))
    N = max(1, len(tr_titles))
    idf = {t: math_log((N + 1) / (c + 1)) + 1 for t, c in dfreq.items()}

    ok = 0
    for _, row in va.iterrows():
        qset = set(toks(row["query"]))
        scores = []
        for c in REF_COLS:
            t, _ = parse(row[c])
            sc = sum(idf.get(tok, 0.5) for tok in set(toks(t)) & qset)
            scores.append(sc)
        pred = [parse(row[c])[0] for c in REF_COLS][int(np.argmax(scores))]
        ok += int(pred == row["gold_title"])
    print("train-title-idf overlap title acc", ok / len(va))


def math_sqrt(x):
    return float(np.sqrt(x))


def math_log(x):
    return float(np.log(x))


if __name__ == "__main__":
    main()
