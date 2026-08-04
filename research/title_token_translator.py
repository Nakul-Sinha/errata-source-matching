"""Supervised query→title-token translator + candidate scoring (no TF-IDF)."""
from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
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


def build_query(row, max_chars=2200):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(300, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


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
    train["gold_title"] = train["provenance_card"].apply(lambda s: json.loads(s)["source_title"])
    tr = train[train.src_year < 2011].reset_index(drop=True)
    va = train[train.src_year >= 2011].reset_index(drop=True)
    print("split", len(tr), len(va), device, flush=True)

    # vocab: query tokens + title tokens
    q_counts = Counter()
    t_counts = Counter()
    for _, row in tr.iterrows():
        q_counts.update(distinctive(tokenize(row["query"]))[:200])
        t_counts.update(distinctive(tokenize(row["gold_title"])))
    q_vocab = ["<pad>", "<unk>"] + [t for t, f in q_counts.most_common(12000) if f >= 2]
    t_vocab = ["<pad>"] + [t for t, f in t_counts.most_common(4000) if f >= 2]
    q2i = {t: i for i, t in enumerate(q_vocab)}
    t2i = {t: i for i, t in enumerate(t_vocab)}
    print("q_vocab", len(q2i), "t_vocab", len(t2i), flush=True)

    def bow(text, vocab, limit=180):
        v = np.zeros(len(vocab), dtype=np.float32)
        for tok in distinctive(tokenize(text))[:limit]:
            v[vocab.get(tok, 1)] += 1.0
        # log1p normalize
        v = np.log1p(v)
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    Xtr = np.stack([bow(q, q2i) for q in tr["query"]])
    Ytr = np.stack([bow(t, t2i) for t in tr["gold_title"]])
    Xva = np.stack([bow(q, q2i) for q in va["query"]])

    class Translator(nn.Module):
        def __init__(self, din, dout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(din, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, dout),
            )

        def forward(self, x):
            return self.net(x)

    model = Translator(len(q2i), len(t2i)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    for ep in range(20):
        model.train()
        idx = np.random.permutation(len(tr))
        tot = 0.0
        for start in range(0, len(tr), 64):
            bi = idx[start : start + 64]
            x = torch.tensor(Xtr[bi], device=device)
            y = torch.tensor(Ytr[bi], device=device)
            pred = model(x)
            # cosine embedding + soft multilabel
            pred_n = F.normalize(pred, dim=-1)
            y_n = F.normalize(y, dim=-1)
            loss = (1 - (pred_n * y_n).sum(-1)).mean()
            loss = loss + F.binary_cross_entropy_with_logits(pred, (y > 0).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        if (ep + 1) % 2 == 0:
            print(f"trans ep{ep+1} loss={tot/(len(tr)/64):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred_va = model(torch.tensor(Xva, device=device)).cpu().numpy()

    # score candidates by cosine of predicted title bow vs candidate title bow
    # plus lexical features blend
    class FeatMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def lex6(query, raw):
        title, year = parse(raw)
        cd = distinctive(tokenize(title))
        ql = query.lower()
        hits = [t for t in cd if t in ql]
        cov = sum(len(t) for t in hits) / max(1.0, sum(len(t) for t in cd))
        return [
            cov,
            len(hits) / max(1, len(cd)),
            float(year in query),
            float(len(hits) >= 2),
            math.log1p(len(hits)),
            float(any(" ".join(cd[i : i + 2]) in ql for i in range(max(0, len(cd) - 1)))),
        ]

    # build candidate title bows
    def score_df(df, pred_title):
        ok_title = []
        preds = []
        confs = []
        for i, row in df.iterrows():
            pt = pred_title[i]
            pt = pt / (np.linalg.norm(pt) + 1e-6)
            scores = []
            for c in REF_COLS:
                title, year = parse(row[c])
                tb = bow(title, t2i)
                cos = float(np.dot(pt, tb))
                lx = lex6(row["query"], row[c])
                scores.append(3.0 * cos + 4.0 * lx[0] + 2.0 * lx[5] + lx[2])
            scores = np.asarray(scores)
            raws = [str(row[c]) for c in REF_COLS]
            groups = title_groups(raws)
            uniq = sorted(set(groups))
            gs = []
            for u in uniq:
                mask = [j for j, g in enumerate(groups) if g == u]
                gs.append(float(np.logaddexp.reduce(scores[mask])))
            best_g = uniq[int(np.argmax(gs))]
            mask = [j for j, g in enumerate(groups) if g == best_g]
            pred = mask[int(np.argmax(scores[mask]))]
            preds.append(pred)
            # conf
            gsp = np.exp(np.asarray(gs) - max(gs))
            gsp = gsp / gsp.sum()
            confs.append(float(gsp.max() * 100))
        return metric(preds, confs, df)

    m = score_df(va, pred_va)
    print("translator+lex", m, flush=True)

    # Also: translator-only
    ok = 0
    for i, row in va.iterrows():
        pt = pred_va[i]
        pt = pt / (np.linalg.norm(pt) + 1e-6)
        scores = []
        for c in REF_COLS:
            tb = bow(parse(row[c])[0], t2i)
            scores.append(float(np.dot(pt, tb)))
        pred = int(np.argmax(scores))
        card = json.loads(row["provenance_card"])
        ok += int(parse(row[REF_COLS[pred]])[0] == card["source_title"])
    print("translator_only title", ok / len(va), flush=True)

    # Train a small mixer on translator cos + lex features
    # precompute
    def build_mix(df, pred_title):
        X = np.zeros((len(df), 16, 7), dtype=np.float32)
        G = np.zeros((len(df), 16), dtype=np.int64)
        Yt = np.zeros(len(df), dtype=np.int64)
        for i, row in df.iterrows():
            pt = pred_title[i]
            pt = pt / (np.linalg.norm(pt) + 1e-6)
            raws = [str(row[c]) for c in REF_COLS]
            G[i] = title_groups(raws)
            for j, c in enumerate(REF_COLS):
                title, year = parse(row[c])
                tb = bow(title, t2i)
                cos = float(np.dot(pt, tb))
                lx = lex6(row["query"], row[c])
                X[i, j] = [cos] + lx
            if "label" in row:
                Yt[i] = G[i, int(row["label"])]
        return X, G, Yt

    # need pred_tr
    with torch.no_grad():
        pred_tr = model(torch.tensor(Xtr, device=device)).cpu().numpy()
    Xtr_m, Gtr, Yt_tr = build_mix(tr, pred_tr)
    Xva_m, Gva, Yt_va = build_mix(va, pred_va)

    mix = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(mix.parameters(), lr=2e-3)
    best = 0.0
    best_m = None
    for ep in range(16):
        mix.train()
        idx = np.random.permutation(len(tr))
        for start in range(0, len(tr), 64):
            bi = idx[start : start + 64]
            s = mix(torch.tensor(Xtr_m[bi], device=device)).squeeze(-1)
            g = torch.tensor(Gtr[bi], device=device)
            y = torch.tensor(Yt_tr[bi], device=device)
            # title group CE
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
        mix.eval()
        with torch.no_grad():
            sv = mix(torch.tensor(Xva_m, device=device)).squeeze(-1).cpu()
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
        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"mix ep{ep+1} {m} best={best:.4f} t={time.time()-t0:.0f}s", flush=True)
    print("DONE", best_m, flush=True)


if __name__ == "__main__":
    main()
