"""Richer OOD-stable features + larger MLP; no TF-IDF."""
from __future__ import annotations

import json
import math
import random
import re
import time
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
    "using", "based", "system", "data", "information", "control", "services",
}


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def tokenize(text):
    return [t.lower().replace("\u2019", "'") for t in TOKEN_RE.findall(str(text))]


def distinctive(tokens):
    return [t for t in tokens if len(t) >= 4 and t not in STOP]


def build_query(row, max_chars=2400):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(300, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


def char_ngrams(s, n=4):
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def lcs_len(a: str, b: str, cap=80) -> int:
    a, b = a[:cap], b[:cap]
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            v = prev[j - 1] + 1 if ca == cb else 0
            cur.append(v)
            if v > best:
                best = v
        prev = cur
    return best


def title_group_ids(raws):
    m, out = {}, []
    for raw in raws:
        t, _ = parse(raw)
        if t not in m:
            m[t] = len(m)
        out.append(m[t])
    return out


def feats(query: str, raw: str) -> list[float]:
    title, year = parse(raw)
    q_toks = tokenize(query)
    c_toks = tokenize(title)
    q_set, c_set = set(q_toks), set(c_toks)
    qd = set(distinctive(q_toks))
    cd = distinctive(c_toks)
    cd_set = set(cd)
    ql = query.lower()
    tl = title.lower()
    hits = [t for t in cd if t in ql]
    exact = [t for t in cd if t in q_set]
    # acronyms
    acr = {a.lower() for a in re.findall(r"\b[A-Z]{2,12}\b", title)}
    acr_hits = len(acr & q_set)
    # char 4grams
    qg, cg = char_ngrams(ql, 4), char_ngrams(tl, 4)
    inter_g = len(qg & cg)
    # phrases
    phrase2 = phrase3 = 0
    for n, bucket in ((2, "p2"), (3, "p3")):
        for i in range(0, max(0, len(cd) - n + 1)):
            ph = " ".join(cd[i : i + n])
            if len(ph) >= 7 and ph in ql:
                if n == 2:
                    phrase2 += 1
                else:
                    phrase3 += 1
    # numbers in title
    nums = re.findall(r"\d{2,4}", title)
    num_hits = sum(1 for n in nums if n in query)
    # unique-ish: tokens appearing in title that are rare in query... skip corpus
    # first-token / last significant
    first_hit = float(bool(cd) and cd[0] in ql)
    # coverage weighted by length
    cov = sum(len(t) for t in hits) / max(1.0, sum(len(t) for t in cd))
    # soft: fraction of title chars covered by hits
    # LCS between note head and title
    note = query.split("\n", 1)[0][:200]
    lcs = lcs_len(note.lower(), tl, 60) / max(1, min(60, len(tl)))
    # hashed binary features for top hit tokens presence — use length buckets
    long_hits = sum(1 for t in hits if len(t) >= 8)
    mid_hits = sum(1 for t in hits if 5 <= len(t) < 8)
    # year proximity features
    years_in_q = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
    y = int(year)
    year_exact = float(year in query)
    year_near = 0.0
    if years_in_q:
        year_near = float(min(abs(y - yq) for yq in years_in_q) <= 2)
    # Jaccard distinctive
    jac = len(qd & cd_set) / max(1, len(qd | cd_set))
    # char jaccard
    cjac = inter_g / max(1, len(qg | cg))
    # title fully contained?
    contained = float(tl in ql)
    # slash / hyphen variants
    variants = 0
    for t in cd:
        if "-" in t or "/" in t:
            for alt in (t.replace("-", " "), t.replace("/", " "), t.replace("-", ""), t.replace("/", "")):
                if alt and alt in ql:
                    variants += 1
    return [
        len(hits) / max(1, len(cd)),
        len(exact) / max(1, len(cd)),
        cov,
        math.log1p(len(hits)),
        math.log1p(sum(len(t) for t in hits)),
        acr_hits / max(1, len(acr)),
        float(acr_hits > 0),
        phrase2,
        phrase3,
        float(phrase2 + phrase3 > 0),
        num_hits / max(1, len(nums)),
        first_hit,
        long_hits,
        mid_hits,
        year_exact,
        year_near,
        jac,
        cjac,
        math.log1p(inter_g),
        lcs,
        contained,
        variants / max(1, len(cd)),
        float(len(hits) >= 2),
        float(len(hits) >= 3),
        float(long_hits >= 1),
        # relative year as float normalized
        (y - 1990) / 40.0,
    ]


def metric(pred_idx, conf, df):
    title_ok = year_ok = both = 0
    for i, row in df.iterrows():
        # df may not be contiguous
        pass
    title_ok = year_ok = both = 0
    rows = df.reset_index(drop=True)
    for i, row in rows.iterrows():
        card = json.loads(row["provenance_card"])
        raw = row[REF_COLS[pred_idx[i]]]
        t, y = parse(raw)
        title_ok += int(t == card["source_title"])
        year_ok += int(y == str(card["source_year"]))
        both += int(t == card["source_title"] and y == str(card["source_year"]))
    n = len(rows)
    # cal
    base = both / n
    conf = np.asarray(conf, dtype=np.float64) / 100.0
    ybin = np.array(
        [
            int(
                parse(rows.iloc[i][REF_COLS[pred_idx[i]]])[0]
                == json.loads(rows.iloc[i]["provenance_card"])["source_title"]
                and parse(rows.iloc[i][REF_COLS[pred_idx[i]]])[1]
                == str(json.loads(rows.iloc[i]["provenance_card"])["source_year"])
            )
            for i in range(n)
        ],
        dtype=np.float64,
    )
    # Actually cal uses exact card match; use both
    ybin = np.zeros(n)
    for i in range(n):
        card = json.loads(rows.iloc[i]["provenance_card"])
        t, y = parse(rows.iloc[i][REF_COLS[pred_idx[i]]])
        ybin[i] = float(t == card["source_title"] and y == str(card["source_year"]))
    brier = float(np.mean((conf - ybin) ** 2))
    brier_base = float(base * (1 - base))
    cal = 0.0 if brier_base < 1e-12 else max(0.0, 1.0 - brier / brier_base)
    tr = title_ok / n
    yr = year_ok / n
    score = 0.85 * tr + 0.05 * (both / n) + 0.10 * cal
    return {
        "title_rate": tr,
        "year_rate": yr,
        "both_rate": both / n,
        "cal": cal,
        "score": score,
    }


def decode(scores, groups):
    # scores (16,), groups (16,)
    uniq = sorted(set(groups.tolist()))
    gs = []
    for u in uniq:
        mask = groups == u
        gs.append(float(torch.logsumexp(scores[mask], 0)))
    best_g = uniq[int(np.argmax(gs))]
    mask = np.where(groups.numpy() == best_g)[0]
    return int(mask[int(scores[mask].argmax())])


def main():
    t0 = time.time()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    tr = train[train.src_year < 2011].reset_index(drop=True)
    va = train[train.src_year >= 2011].reset_index(drop=True)
    print("split", len(tr), len(va), "device", device, flush=True)

    def build_X(df):
        X = np.zeros((len(df), 16, 26), dtype=np.float32)
        groups = np.zeros((len(df), 16), dtype=np.int64)
        labels = df["label"].to_numpy()
        gold_titles = [json.loads(s)["source_title"] for s in df["provenance_card"]]
        for i, row in df.iterrows():
            raws = [str(row[c]) for c in REF_COLS]
            groups[i] = title_group_ids(raws)
            for j, raw in enumerate(raws):
                X[i, j] = feats(row["query"], raw)
        return X, groups, labels, gold_titles

    print("building features...", flush=True)
    Xtr, Gtr, Ytr, _ = build_X(tr)
    Xva, Gva, Yva, gold_va = build_X(va)
    print("X", Xtr.shape, flush=True)

    # handcrafted score oracle
    hand = Xva[:, :, 0] + 2 * Xva[:, :, 2] + 1.5 * Xva[:, :, 7] + 2 * Xva[:, :, 8] + Xva[:, :, 16] * 3
    pred = hand.argmax(1)
    # title group decode hand
    pred_g = []
    for i in range(len(va)):
        s = torch.tensor(hand[i])
        pred_g.append(decode(s, torch.tensor(Gva[i])))
    print("hand title-group", metric(pred_g, np.full(len(va), 50), va), flush=True)

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 128),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = MLP(26).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02)
    best = 0.0
    best_m = None
    for epoch in range(40):
        model.train()
        idx = np.random.permutation(len(tr))
        tot = 0.0
        for start in range(0, len(tr), 64):
            bi = idx[start : start + 64]
            x = torch.tensor(Xtr[bi], device=device)
            g = torch.tensor(Gtr[bi], device=device)
            scores = model(x)
            loss = 0.0
            for b in range(len(bi)):
                raws = [str(tr.loc[bi[b], c]) for c in REF_COLS]
                titles = [parse(r)[0] for r in raws]
                gold = json.loads(tr.loc[bi[b], "provenance_card"])["source_title"]
                uniq = []
                gmap = []
                for t in titles:
                    if t not in uniq:
                        uniq.append(t)
                    gmap.append(uniq.index(t))
                gmap_t = torch.tensor(gmap, device=device)
                s = scores[b]
                gs = s.new_full((len(uniq),), -1e4)
                for u in range(len(uniq)):
                    gs[u] = torch.logsumexp(s[gmap_t == u], 0)
                loss = loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([uniq.index(gold)], device=device))
                loss = loss + 0.2 * F.cross_entropy(
                    s.unsqueeze(0), torch.tensor([int(Ytr[bi[b]])], device=device)
                )
            loss = loss / len(bi)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(Xva, device=device)
            sv = model(xv).cpu()
            preds = []
            confs = []
            for i in range(len(va)):
                s = sv[i]
                g = torch.tensor(Gva[i])
                # group softmax conf
                uniq = sorted(set(Gva[i].tolist()))
                gs = []
                for u in uniq:
                    gs.append(float(torch.logsumexp(s[g == u], 0)))
                gs_t = torch.tensor(gs)
                p = F.softmax(gs_t, 0)
                best_g = uniq[int(p.argmax())]
                mask = np.where(Gva[i] == best_g)[0]
                preds.append(int(mask[int(s[mask].argmax())]))
                confs.append(float(p.max() * 100))
            m = metric(preds, confs, va)
            if m["score"] > best:
                best = m["score"]
                best_m = m
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"epoch {epoch+1} {m} best={best:.4f} t={time.time()-t0:.0f}s", flush=True)
    print("DONE", best_m, flush=True)


if __name__ == "__main__":
    main()
