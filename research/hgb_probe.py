"""HistGradientBoosting on rich features (not TF-IDF)."""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = Path(__file__).resolve().parents[1] / "dataset" / "public"
REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d{2,4}")
STOP = {
    "year", "protocol", "specification", "version", "internet", "network",
    "format", "message", "standard", "requirements", "framework", "profile",
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
    "using", "based", "system", "data", "information", "control", "services",
}


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def tokenize(text):
    return [t.lower().replace("\u2019", "'") for t in TOKEN_RE.findall(str(text))]


def distinctive(tokens):
    return [t for t in tokens if len(t) >= 4 and t not in STOP]


def build_query(row, max_chars=2200):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(300, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


def feats(query, raw):
    title, year = parse(raw)
    cd = distinctive(tokenize(title))
    ql = query.lower()
    hits = [t for t in cd if t in ql]
    acr = {a.lower() for a in re.findall(r"\b[A-Z]{2,12}\b", title)}
    qs = set(tokenize(query))
    phrase2 = sum(1 for i in range(max(0, len(cd) - 1)) if " ".join(cd[i : i + 2]) in ql)
    phrase3 = sum(1 for i in range(max(0, len(cd) - 2)) if " ".join(cd[i : i + 3]) in ql)
    qg = {ql[i : i + 4] for i in range(max(0, len(ql) - 3))}
    cg = {title.lower()[i : i + 4] for i in range(max(0, len(title) - 3))}
    inter = len(qg & cg)
    cov = sum(len(t) for t in hits) / max(1.0, sum(len(t) for t in cd))
    y = int(year)
    years = [int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
    return [
        len(hits) / max(1, len(cd)),
        cov,
        len(acr & qs) / max(1, len(acr)),
        float(bool(acr & qs)),
        float(phrase2),
        float(phrase3),
        float(phrase2 + phrase3 > 0),
        inter / max(1, len(cg)),
        math.log1p(inter),
        float(year in query),
        float(bool(years) and min(abs(y - yq) for yq in years) <= 2),
        float(sum(1 for t in hits if len(t) >= 8)),
        float(len(hits) >= 2),
        float(len(hits) >= 3),
        float(title.lower() in ql),
        math.log1p(len(hits)),
        float(max((len(t) for t in hits), default=0) >= 7),
        (y - 1990) / 40.0,
        len(hits) / max(1, len(set(distinctive(tokenize(query))))),
        float(bool(hits)),
    ]


def main():
    train = pd.read_csv(DATA / "train.csv")
    train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
    train["query"] = [build_query(r) for _, r in train.iterrows()]
    train["label"] = train.apply(
        lambda row: next(
            i
            for i, c in enumerate(REF_COLS)
            if parse(row[c])[0] == json.loads(row["provenance_card"])["source_title"]
            and parse(row[c])[1] == str(json.loads(row["provenance_card"])["source_year"])
        ),
        axis=1,
    )
    tr = train[train.src_year < 2011].reset_index(drop=True)
    va = train[train.src_year >= 2011].reset_index(drop=True)
    print("split", len(tr), len(va), flush=True)

    Xtr, ytr = [], []
    for _, row in tr.iterrows():
        for j, c in enumerate(REF_COLS):
            Xtr.append(feats(row["query"], row[c]))
            ytr.append(1 if j == row["label"] else 0)
    Xtr = np.asarray(Xtr, dtype=np.float32)
    ytr = np.asarray(ytr)
    print("fitting HGB...", flush=True)
    clf = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=250, l2_regularization=0.1, random_state=0
    )
    clf.fit(Xtr, ytr)

    ok = both = 0
    for _, row in va.iterrows():
        scores = []
        for c in REF_COLS:
            f = np.asarray(feats(row["query"], row[c]), dtype=np.float32).reshape(1, -1)
            scores.append(clf.predict_proba(f)[0, 1])
        scores = np.asarray(scores)
        raws = [str(row[c]) for c in REF_COLS]
        titles = [parse(r)[0] for r in raws]
        gscores = defaultdict(lambda: -1e9)
        for j, t in enumerate(titles):
            gscores[t] = max(gscores[t], scores[j])
        best_t = max(gscores, key=gscores.get)
        cands = [j for j, t in enumerate(titles) if t == best_t]
        pred = cands[int(np.argmax(scores[cands]))]
        card = json.loads(row["provenance_card"])
        t, y = parse(row[REF_COLS[pred]])
        ok += int(t == card["source_title"])
        both += int(t == card["source_title"] and y == str(card["source_year"]))
    n = len(va)
    tr_rate = ok / n
    br = both / n
    print("hgb", {"title_rate": tr_rate, "both_rate": br, "score": 0.85 * tr_rate + 0.05 * br}, flush=True)


if __name__ == "__main__":
    main()
