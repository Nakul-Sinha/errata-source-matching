"""Cascade: rich lexical MLP when coverage high; offline CE when low.

local_files_only — no download. Tests if semantic rescue of zero-overlap rows
can push time-split score toward 0.55 while preserving lexical wins.
"""
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
from transformers import AutoModel, AutoTokenizer

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
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def tokenize(text):
    return [t.lower().replace("\u2019", "'") for t in TOKEN_RE.findall(str(text))]


def distinctive(tokens):
    return [t for t in tokens if len(t) >= 4 and t not in STOP]


def build_query(row, max_chars=2000):
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


def feats(query, raw):
    title, year = parse(raw)
    q_toks = tokenize(query)
    c_toks = tokenize(title)
    q_set = set(q_toks)
    cd = distinctive(c_toks)
    ql = query.lower()
    tl = title.lower()
    hits = [t for t in cd if t in ql]
    acr = {a.lower() for a in re.findall(r"\b[A-Z]{2,12}\b", title)}
    acr_hits = len(acr & q_set)
    phrase2 = sum(1 for i in range(max(0, len(cd) - 1)) if " ".join(cd[i : i + 2]) in ql)
    phrase3 = sum(1 for i in range(max(0, len(cd) - 2)) if " ".join(cd[i : i + 3]) in ql)
    qg, cg = char_ngrams(ql, 4), char_ngrams(tl, 4)
    inter = len(qg & cg)
    cov = sum(len(t) for t in hits) / max(1.0, sum(len(t) for t in cd))
    long_hits = sum(1 for t in hits if len(t) >= 8)
    y = int(year)
    years_in_q = [int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
    return [
        len(hits) / max(1, len(cd)),
        cov,
        acr_hits / max(1, len(acr)),
        float(acr_hits > 0),
        float(phrase2),
        float(phrase3),
        float(phrase2 + phrase3 > 0),
        inter / max(1, len(cg)),
        math.log1p(inter),
        float(year in query),
        float(bool(years_in_q) and min(abs(y - yq) for yq in years_in_q) <= 2),
        float(long_hits),
        float(len(hits) >= 2),
        float(len(hits) >= 3),
        float(tl in ql),
        math.log1p(len(hits)),
        float(max((len(t) for t in hits), default=0) >= 7),
        (y - 1990) / 40.0,
        len(hits) / max(1, len(set(distinctive(q_toks)))),
        float(bool(hits)),
    ]


def coverage_signal(query, raws):
    """Max distinctive coverage across candidates — gate for cascade."""
    best = 0.0
    for raw in raws:
        title, _ = parse(raw)
        cd = distinctive(tokenize(title))
        ql = query.lower()
        hits = [t for t in cd if t in ql]
        cov = sum(len(t) for t in hits) / max(1.0, sum(len(t) for t in cd))
        best = max(best, cov, 0.15 * len(hits))
    return best


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


class FeatMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0]).squeeze(-1)


def decode(scores, groups):
    uniq = sorted(set(groups))
    gs = []
    g_t = torch.tensor(groups, device=scores.device)
    for u in uniq:
        gs.append(torch.logsumexp(scores[g_t == u], 0))
    gs = torch.stack(gs)
    p = F.softmax(gs, 0)
    best_g = uniq[int(p.argmax())]
    mask = [j for j, g in enumerate(groups) if g == best_g]
    return mask[int(scores[mask].argmax())], float(p.max() * 100)


def main():
    t0 = time.time()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
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
    print("split", len(tr), len(va), device, flush=True)

    # features
    def build_X(df):
        X = np.zeros((len(df), 16, 20), dtype=np.float32)
        G = np.zeros((len(df), 16), dtype=np.int64)
        cov = np.zeros(len(df), dtype=np.float32)
        for i, row in df.iterrows():
            raws = [str(row[c]) for c in REF_COLS]
            G[i] = title_groups(raws)
            cov[i] = coverage_signal(row["query"], raws)
            for j, raw in enumerate(raws):
                X[i, j] = feats(row["query"], raw)
        return X, G, cov

    Xtr, Gtr, Covtr = build_X(tr)
    Xva, Gva, Covva = build_X(va)
    print("cov val percentiles", np.percentile(Covva, [10, 25, 50, 75, 90]), flush=True)

    # train feat MLP
    feat = FeatMLP(20).to(device)
    opt = torch.optim.AdamW(feat.parameters(), lr=2e-3, weight_decay=0.02)
    best_state, best_sc = None, -1
    for epoch in range(12):
        feat.train()
        idx = np.random.permutation(len(tr))
        for start in range(0, len(tr), 64):
            bi = idx[start : start + 64]
            scores = feat(torch.tensor(Xtr[bi], device=device))
            loss = 0.0
            for b, i in enumerate(bi):
                s = scores[b]
                raws = [str(tr.loc[i, c]) for c in REF_COLS]
                titles = [parse(r)[0] for r in raws]
                gold = json.loads(tr.loc[i, "provenance_card"])["source_title"]
                uniq, gmap = [], []
                for t in titles:
                    if t not in uniq:
                        uniq.append(t)
                    gmap.append(uniq.index(t))
                gmap_t = torch.tensor(gmap, device=device)
                gs = s.new_full((len(uniq),), -1e4)
                for u in range(len(uniq)):
                    gs[u] = torch.logsumexp(s[gmap_t == u], 0)
                loss = loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([uniq.index(gold)], device=device))
            loss = loss / len(bi)
            opt.zero_grad()
            loss.backward()
            opt.step()
        feat.eval()
        with torch.no_grad():
            sv = feat(torch.tensor(Xva, device=device)).cpu()
            preds, confs = [], []
            for i in range(len(va)):
                p, c = decode(sv[i], Gva[i].tolist())
                preds.append(p)
                confs.append(c)
            m = metric(preds, confs, va)
        if m["score"] > best_sc:
            best_sc = m["score"]
            best_state = {k: v.cpu().clone() for k, v in feat.state_dict().items()}
        print(f"feat ep{epoch+1} {m} best={best_sc:.4f}", flush=True)
    feat.load_state_dict(best_state)
    feat.eval()
    with torch.no_grad():
        feat_va = feat(torch.tensor(Xva, device=device)).cpu().numpy()
        feat_tr = feat(torch.tensor(Xtr, device=device)).cpu().numpy()

    # Analyze feat accuracy by coverage bucket
    preds = []
    for i in range(len(va)):
        p, _ = decode(torch.tensor(feat_va[i]), Gva[i].tolist())
        preds.append(p)
    for thr in (0.05, 0.1, 0.2, 0.3, 0.5):
        hi = Covva >= thr
        lo = ~hi
        def acc(mask, preds):
            if mask.sum() == 0:
                return float("nan"), 0
            ok = 0
            for i in np.where(mask)[0]:
                card = json.loads(va.loc[i, "provenance_card"])
                t, _ = parse(va.loc[i, REF_COLS[preds[i]]])
                ok += int(t == card["source_title"])
            return ok / mask.sum(), int(mask.sum())
        a_hi, n_hi = acc(hi, preds)
        a_lo, n_lo = acc(lo, preds)
        print(f"cov>={thr}: hi_acc={a_hi:.3f} n={n_hi} | lo_acc={a_lo:.3f} n={n_lo}", flush=True)

    # Train CE only on LOW coverage train rows (force semantic learning)
    ce = CE().to(device)
    for name, p in ce.encoder.named_parameters():
        if "embeddings" in name or any(f"layer.{i}." in name for i in range(3)):
            p.requires_grad = False
    opt = torch.optim.AdamW([p for p in ce.parameters() if p.requires_grad], lr=3e-5, weight_decay=0.01)

    low_idx = [i for i in range(len(tr)) if Covtr[i] < 0.25]
    print("low-cov train rows", len(low_idx), flush=True)

    def encode_batch(queries, cands):
        pq, pc = [], []
        for q, cs in zip(queries, cands):
            for c in cs:
                pq.append(q)
                pc.append(c)
        enc = tok(pq, pc, padding=True, truncation=True, max_length=288, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    for epoch in range(4):
        ce.train()
        random.shuffle(low_idx)
        tot = 0.0
        steps = 0
        for start in range(0, len(low_idx), 2):
            batch = low_idx[start : start + 2]
            queries = [tr.loc[i, "query"] for i in batch]
            cands, groups, golds, labs = [], [], [], []
            for i in batch:
                row = tr.loc[i]
                raws = [str(row[c]) for c in REF_COLS]
                cands.append([parse(r)[0] + " (" + parse(r)[1] + ")" for r in raws])
                groups.append(title_groups(raws))
                golds.append(json.loads(row["provenance_card"])["source_title"])
                labs.append(int(row["label"]))
            enc = encode_batch(queries, cands)
            scores = ce(enc["input_ids"], enc["attention_mask"]).view(len(batch), 16)
            loss = 0.0
            for bi in range(len(batch)):
                s = scores[bi]
                raws = [str(tr.loc[batch[bi], c]) for c in REF_COLS]
                titles = [parse(r)[0] for r in raws]
                gmap = groups[bi]
                uniq = sorted(set(gmap))
                gs = s.new_full((len(uniq),), -1e4)
                gt = torch.tensor(gmap, device=device)
                for u, ug in enumerate(uniq):
                    gs[u] = torch.logsumexp(s[gt == ug], 0)
                gold_g = gmap[titles.index(golds[bi])]
                loss = loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([uniq.index(gold_g)], device=device))
            loss = loss / len(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            steps += 1
        print(f"ce ep{epoch+1} loss={tot/max(1,steps):.4f} t={time.time()-t0:.0f}s", flush=True)

    # CE logits on val
    ce.eval()
    ce_va = np.zeros((len(va), 16), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(va), 4):
            batch = list(range(start, min(len(va), start + 4)))
            queries = [va.loc[i, "query"] for i in batch]
            cands = []
            for i in batch:
                raws = [str(va.loc[i, c]) for c in REF_COLS]
                cands.append([parse(r)[0] + " (" + parse(r)[1] + ")" for r in raws])
            enc = encode_batch(queries, cands)
            scores = ce(enc["input_ids"], enc["attention_mask"]).view(len(batch), 16).cpu().numpy()
            ce_va[batch] = scores

    # Cascade over thresholds
    for thr in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
        preds, confs = [], []
        n_ce = 0
        for i in range(len(va)):
            if Covva[i] >= thr:
                p, c = decode(torch.tensor(feat_va[i]), Gva[i].tolist())
            else:
                p, c = decode(torch.tensor(ce_va[i]), Gva[i].tolist())
                n_ce += 1
            preds.append(p)
            confs.append(c)
        m = metric(preds, confs, va)
        print(f"cascade thr={thr} n_ce={n_ce} {m}", flush=True)

    # Also: feat + small CE residual only on low cov
    for thr in (0.15, 0.25):
        for alpha in (0.3, 0.5, 1.0):
            preds, confs = [], []
            for i in range(len(va)):
                f = feat_va[i]
                if Covva[i] < thr:
                    # z-blend
                    c = ce_va[i]
                    f = (f - f.mean()) / (f.std() + 1e-6)
                    c = (c - c.mean()) / (c.std() + 1e-6)
                    s = (1 - alpha) * f + alpha * c
                else:
                    s = f
                p, cfd = decode(torch.tensor(s), Gva[i].tolist())
                preds.append(p)
                confs.append(cfd)
            m = metric(preds, confs, va)
            print(f"blend thr={thr} a={alpha} {m}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
