"""Compare feature-MLP scores across split protocols."""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[1] / "dataset" / "public"
REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d{2,4}")
STOP = {
    "year", "protocol", "specification", "version", "internet", "network",
    "format", "message", "standard", "requirements", "framework", "profile",
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
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


def title_groups(raws):
    m, out = {}, []
    for r in raws:
        t, _ = parse(r)
        if t not in m:
            m[t] = len(m)
        out.append(m[t])
    return out


def metric(preds, confs, df):
    rows = df.reset_index(drop=True)
    title_ok = both = 0
    ybin = []
    for i, row in rows.iterrows():
        card = json.loads(row["provenance_card"])
        t, y = parse(row[REF_COLS[preds[i]]])
        title_ok += int(t == card["source_title"])
        both += int(t == card["source_title"] and y == str(card["source_year"]))
        ybin.append(float(t == card["source_title"] and y == str(card["source_year"])))
    n = len(rows)
    base = both / n
    conf = np.asarray(confs) / 100.0
    ybin = np.asarray(ybin)
    brier = float(np.mean((conf - ybin) ** 2))
    brier_base = float(base * (1 - base))
    cal = 0.0 if brier_base < 1e-12 else max(0.0, 1.0 - brier / brier_base)
    tr = title_ok / n
    return {"title_rate": tr, "both_rate": both / n, "cal": cal, "score": 0.85 * tr + 0.05 * (both / n) + 0.10 * cal}


def run_split(tr, va, tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(df):
        X = np.zeros((len(df), 16, 20), dtype=np.float32)
        G = np.zeros((len(df), 16), dtype=np.int64)
        Yt = np.zeros(len(df), dtype=np.int64)
        for i, row in df.iterrows():
            raws = [str(row[c]) for c in REF_COLS]
            G[i] = title_groups(raws)
            for j, raw in enumerate(raws):
                X[i, j] = feats(row["query"], raw)
            Yt[i] = G[i, int(row["label"])]
        return X, G, Yt

    Xtr, Gtr, Yt = build(tr)
    Xva, Gva, _ = build(va)
    model = nn.Sequential(nn.Linear(20, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02)
    best = -1
    best_m = None
    for ep in range(8):
        model.train()
        idx = np.random.permutation(len(tr))
        for start in range(0, len(tr), 64):
            bi = idx[start : start + 64]
            s = model(torch.tensor(Xtr[bi], device=device)).squeeze(-1)
            g = torch.tensor(Gtr[bi], device=device)
            y = torch.tensor(Yt[bi], device=device)
            bsz = s.size(0)
            max_g = int(g.max()) + 1
            gs = s.new_full((bsz, max_g), -1e4)
            for b in range(bsz):
                for gi in g[b].unique():
                    gs[b, int(gi)] = torch.logsumexp(s[b, g[b] == gi], 0)
            loss = F.cross_entropy(gs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = model(torch.tensor(Xva, device=device)).squeeze(-1).cpu()
            preds, confs = [], []
            for i in range(len(va)):
                s = sv[i]
                g = Gva[i].tolist()
                uniq = sorted(set(g))
                gs = torch.stack([torch.logsumexp(s[torch.tensor(g) == u], 0) for u in uniq])
                p = F.softmax(gs, 0)
                bg = uniq[int(p.argmax())]
                mask = [j for j, gg in enumerate(g) if gg == bg]
                preds.append(mask[int(s[mask].argmax())])
                confs.append(float(p.max() * 100))
            m = metric(preds, confs, va)
        if m["score"] > best:
            best = m["score"]
            best_m = m
    print(tag, "ntr", len(tr), "nva", len(va), best_m, flush=True)
    return best_m


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    train = pd.read_csv(DATA / "train.csv")
    train["label"] = train.apply(
        lambda row: next(
            i
            for i, c in enumerate(REF_COLS)
            if parse(row[c])[0] == json.loads(row["provenance_card"])["source_title"]
            and parse(row[c])[1] == str(json.loads(row["provenance_card"])["source_year"])
        ),
        axis=1,
    )
    train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
    train["query"] = [build_query(r) for _, r in train.iterrows()]

    for vy in (2010, 2011, 2012, 2013):
        tr = train[train.src_year < vy].reset_index(drop=True)
        va = train[train.src_year >= vy].reset_index(drop=True)
        if len(va) < 50 or len(tr) < 500:
            continue
        run_split(tr, va, f"time>={vy}")

    # random split
    idx = np.random.permutation(len(train))
    cut = int(0.85 * len(train))
    tr = train.iloc[idx[:cut]].reset_index(drop=True)
    va = train.iloc[idx[cut:]].reset_index(drop=True)
    run_split(tr, va, "random15")


if __name__ == "__main__":
    main()
