"""Look for exploitable structure in errata text vs gold titles."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "dataset" / "public"
REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def main():
    train = pd.read_csv(DATA / "train.csv")
    train["card"] = train["provenance_card"].apply(json.loads)
    train["gold_title"] = train["card"].apply(lambda c: c["source_title"])
    train["gold_year"] = train["card"].apply(lambda c: str(c["source_year"]))
    train["src_year"] = train["gold_year"].astype(int)
    va = train[train.src_year >= 2011]

    # How often is full gold title in note / corr / excerpt?
    for part in ["submitter_note", "proposed_correction", "original_excerpt"]:
        hit = 0
        for _, row in va.iterrows():
            text = str(row[part]).lower()
            if row["gold_title"].lower() in text:
                hit += 1
        print(f"full title in {part}: {hit/len(va):.4f}")

    # Longest title token in note
    def longest_hit(text, title):
        toks = re.findall(r"[A-Za-z]{5,}", title)
        text = text.lower()
        hits = [t for t in toks if t.lower() in text]
        return max((len(t) for t in hits), default=0)

    buckets = Counter()
    for _, row in va.iterrows():
        lh = longest_hit(str(row["submitter_note"]) + " " + str(row["proposed_correction"]), row["gold_title"])
        buckets[lh // 5] += 1
    print("longest hit length//5 in note+corr:", dict(sorted(buckets.items())))

    # Are candidates near-duplicates? title uniqueness per row
    n_unique = []
    for _, row in va.iterrows():
        titles = [parse(row[c])[0] for c in REF_COLS]
        n_unique.append(len(set(titles)))
    print("mean unique titles/row", sum(n_unique) / len(n_unique), "min", min(n_unique), "max", max(n_unique))

    # Gold title rank by simple coverage among title-groups
    from collections import defaultdict

    def score_title(q, title):
        ql = q.lower()
        toks = [t for t in re.findall(r"[A-Za-z]{4,}", title.lower())]
        hits = [t for t in toks if t in ql]
        return sum(len(t) for t in hits)

    correct_group = 0
    for _, row in va.iterrows():
        q = f"{row['submitter_note']}\n{row['proposed_correction']}\n{row['original_excerpt']}"
        # best title by coverage
        best_t, best_s = None, -1
        for c in REF_COLS:
            t, _ = parse(row[c])
            s = score_title(q, t)
            if s > best_s:
                best_s = s
                best_t = t
        correct_group += int(best_t == row["gold_title"])
    print("coverage title-group acc", correct_group / len(va))

    # Show 5 failures where gold has ZERO distinctive tokens in query
    print("\n--- zero-hit examples ---")
    shown = 0
    for _, row in va.iterrows():
        q = f"{row['submitter_note']}\n{row['proposed_correction']}\n{row['original_excerpt']}".lower()
        toks = [t for t in re.findall(r"[A-Za-z]{5,}", row["gold_title"].lower())]
        hits = [t for t in toks if t in q]
        if hits:
            continue
        print("GOLD:", row["gold_title"])
        print("NOTE:", str(row["submitter_note"])[:200])
        print("CORR:", str(row["proposed_correction"])[:200])
        print("EXCERPT:", str(row["original_excerpt"])[:250])
        print("CANDS:")
        for c in REF_COLS[:6]:
            print(" ", row[c])
        print("---")
        shown += 1
        if shown >= 4:
            break


if __name__ == "__main__":
    main()
