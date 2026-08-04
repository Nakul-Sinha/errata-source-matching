"""CountVectorizer char/word ngrams (NOT TF-IDF) + title-group ranking."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse

DATA = Path(__file__).resolve().parents[1] / "dataset" / "public"
REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def build_query(row):
    return f"{row['submitter_note']}\n{row['proposed_correction']}\n{row['original_excerpt']}"


def main():
    train = pd.read_csv(DATA / "train.csv")
    train["card"] = train["provenance_card"].apply(json.loads)
    train["gold_title"] = train["card"].apply(lambda c: c["source_title"])
    train["gold_year"] = train["card"].apply(lambda c: str(c["source_year"]))
    train["src_year"] = train["gold_year"].astype(int)
    train["query"] = train.apply(build_query, axis=1)
    tr = train[train.src_year < 2011].reset_index(drop=True)
    va = train[train.src_year >= 2011].reset_index(drop=True)

    # Fit count vectorizers on train queries + titles only
    corpus = list(tr["query"])
    for _, row in tr.iterrows():
        for c in REF_COLS:
            corpus.append(parse(row[c])[0])

    word = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000)
    char = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000)
    word.fit(corpus)
    char.fit(corpus)

    def score_row(row):
        q = row["query"]
        qw = word.transform([q])
        qc = char.transform([q])
        scores = []
        titles = []
        for c in REF_COLS:
            t, y = parse(row[c])
            titles.append(t)
            tw = word.transform([t])
            tc = char.transform([t])
            # cosine on counts
            def cos(a, b):
                num = a.multiply(b).sum()
                na = np.sqrt(a.multiply(a).sum())
                nb = np.sqrt(b.multiply(b).sum())
                return float(num / (na * nb + 1e-9))

            scores.append(0.45 * cos(qw, tw) + 0.55 * cos(qc, tc) + (0.05 if y in q else 0))
        scores = np.asarray(scores)
        # title group
        best_t = None
        best_s = -1
        for t in set(titles):
            mask = [i for i, tt in enumerate(titles) if tt == t]
            s = float(np.logaddexp.reduce(scores[mask]))
            if s > best_s:
                best_s = s
                best_t = t
        cands = [i for i, t in enumerate(titles) if t == best_t]
        return cands[int(np.argmax(scores[cands]))]

    ok = both = 0
    for _, row in va.iterrows():
        pred = score_row(row)
        t, y = parse(row[REF_COLS[pred]])
        ok += int(t == row["gold_title"])
        both += int(t == row["gold_title"] and y == row["gold_year"])
    n = len(va)
    print("countvec", {"title_rate": ok / n, "both_rate": both / n, "score": 0.85 * ok / n + 0.05 * both / n})

    # Learn a logistic on concatenated count overlaps — pairwise
    # For speed: use handcrafted cos features into LR like rich feat
    X = []
    y = []
    for _, row in tr.iterrows():
        q = row["query"]
        qw, qc = word.transform([q]), char.transform([q])
        for j, c in enumerate(REF_COLS):
            t, yr = parse(row[c])
            tw, tc = word.transform([t]), char.transform([t])

            def cos(a, b):
                num = a.multiply(b).sum()
                na = np.sqrt(a.multiply(a).sum())
                nb = np.sqrt(b.multiply(b).sum())
                return float(num / (na * nb + 1e-9))

            X.append([cos(qw, tw), cos(qc, tc), float(yr in q)])
            gold_t = row["gold_title"]
            y.append(1 if t == gold_t else 0)
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(np.asarray(X), np.asarray(y))

    ok = both = 0
    for _, row in va.iterrows():
        q = row["query"]
        qw, qc = word.transform([q]), char.transform([q])
        scores = []
        titles = []
        for c in REF_COLS:
            t, yr = parse(row[c])
            titles.append(t)
            tw, tc = word.transform([t]), char.transform([t])

            def cos(a, b):
                num = a.multiply(b).sum()
                na = np.sqrt(a.multiply(a).sum())
                nb = np.sqrt(b.multiply(b).sum())
                return float(num / (na * nb + 1e-9))

            scores.append(clf.predict_proba([[cos(qw, tw), cos(qc, tc), float(yr in q)]])[0, 1])
        scores = np.asarray(scores)
        best_t = max(set(titles), key=lambda t: max(scores[i] for i, tt in enumerate(titles) if tt == t))
        cands = [i for i, t in enumerate(titles) if t == best_t]
        pred = cands[int(np.argmax(scores[cands]))]
        t, yv = parse(row[REF_COLS[pred]])
        ok += int(t == row["gold_title"])
        both += int(t == row["gold_title"] and yv == row["gold_year"])
    print("countvec+LR", {"title_rate": ok / n, "both_rate": both / n, "score": 0.85 * ok / n + 0.05 * both / n})


if __name__ == "__main__":
    main()
