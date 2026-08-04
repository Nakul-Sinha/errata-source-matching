"""Fuzzy / soft-match oracles beyond exact token overlap."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "dataset" / "public"
REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
STOP = {
    "year", "protocol", "specification", "version", "internet", "network",
    "format", "message", "standard", "requirements", "framework", "profile",
}


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def toks(s):
    return [t for t in re.findall(r"[a-z]{4,}|\d{4}", s.lower()) if t not in STOP]


def build_query(row):
    return f"{row['submitter_note']}\n{row['proposed_correction']}\n{row['original_excerpt']}"


def edit_leq1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    # check one edit
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) < len(b):
        a, b = b, a
    # a longer by 1
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            diff += 1
            if diff > 1:
                return False
            i += 1
        else:
            i += 1
            j += 1
    return True


def score_fuzzy(query, title):
    ql = query.lower()
    qset = set(toks(ql))
    cd = toks(title)
    sc = 0.0
    for t in cd:
        if t in ql:
            sc += len(t) * 2
            continue
        # prefix match length>=5
        if len(t) >= 6 and any(t[:6] in q for q in qset):
            sc += len(t) * 0.8
            continue
        # edit distance 1 against any query token of similar length
        for q in qset:
            if abs(len(q) - len(t)) <= 1 and len(t) >= 5 and edit_leq1(t, q):
                sc += len(t) * 1.2
                break
        # substring of title token in query (partial)
        if len(t) >= 7 and t[:5] in ql:
            sc += 2.0
    # phrase
    for i in range(len(cd) - 1):
        if cd[i] + " " + cd[i + 1] in ql:
            sc += 8
    return sc


def main():
    train = pd.read_csv(DATA / "train.csv")
    train["card"] = train["provenance_card"].apply(json.loads)
    train["gold"] = train["card"].apply(lambda c: c["source_title"])
    train["src_year"] = train["card"].apply(lambda c: int(c["source_year"]))
    va = train[train.src_year >= 2011].reset_index(drop=True)
    ok = 0
    for _, row in va.iterrows():
        q = build_query(row)
        scores = [score_fuzzy(q, parse(row[c])[0]) for c in REF_COLS]
        pred = parse(row[REF_COLS[int(np.argmax(scores))]])[0]
        ok += int(pred == row["gold"])
    print("fuzzy title acc", ok / len(va))

    # combine with exact coverage
    ok2 = 0
    for _, row in va.iterrows():
        q = build_query(row)
        scores = []
        for c in REF_COLS:
            t = parse(row[c])[0]
            cd = toks(t)
            hits = [x for x in cd if x in q.lower()]
            cov = sum(len(x) for x in hits) / max(1.0, sum(len(x) for x in cd))
            scores.append(3 * cov + 0.15 * score_fuzzy(q, t) + 2 * (1 if " ".join(cd[:2]) in q.lower() else 0))
        pred = parse(row[REF_COLS[int(np.argmax(scores))]])[0]
        ok2 += int(pred == row["gold"])
    print("cov+fuzzy title acc", ok2 / len(va))


if __name__ == "__main__":
    main()
