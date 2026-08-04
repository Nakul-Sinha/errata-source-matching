"""Push lexical+learned feature ceiling with slate-relative + more matchers."""
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
    "document", "section", "should", "must", "may", "page", "errata",
}


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def tokenize(text):
    return [t.lower().replace("\u2019", "'") for t in TOKEN_RE.findall(str(text))]


def distinctive(tokens):
    return [t for t in tokens if len(t) >= 4 and t not in STOP]


def build_query(row, max_chars=2600):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(400, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


def char_ngrams(s, n):
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def stemish(t: str) -> str:
    for suf in ("tion", "sion", "ing", "ed", "ly", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def raw_feats(query, raw, tok_w):
    title, year = parse(raw)
    q_toks = tokenize(query)
    c_toks = tokenize(title)
    q_set = set(q_toks)
    qd = set(distinctive(q_toks))
    cd = distinctive(c_toks)
    ql = query.lower()
    tl = title.lower()
    hits = [t for t in cd if t in ql]
    # stemmed soft hits
    q_stems = {stemish(t) for t in distinctive(q_toks)}
    soft = [t for t in cd if stemish(t) in q_stems or t in ql]
    acr = {a.lower() for a in re.findall(r"\b[A-Z]{2,12}\b", title)}
    # also build acronym from title initials
    words = re.findall(r"[A-Za-z]+", title)
    if len(words) >= 3:
        initials = "".join(w[0] for w in words if w[0].isupper() or len(w) >= 4).lower()
        if 2 <= len(initials) <= 8:
            acr.add(initials)
    acr_hits = len(acr & q_set) + sum(1 for a in acr if a in ql)
    phrase2 = sum(
        1
        for i in range(max(0, len(cd) - 1))
        if len(cd[i]) >= 4 and (cd[i] + " " + cd[i + 1]) in ql
    )
    phrase3 = sum(
        1
        for i in range(max(0, len(cd) - 2))
        if (" ".join(cd[i : i + 3]) in ql)
    )
    qg4, cg4 = char_ngrams(ql, 4), char_ngrams(tl, 4)
    qg5, cg5 = char_ngrams(ql, 5), char_ngrams(tl, 5)
    inter4 = len(qg4 & cg4)
    inter5 = len(qg5 & cg5)
    cov = sum(len(t) for t in hits) / max(1.0, sum(len(t) for t in cd))
    soft_cov = sum(len(t) for t in soft) / max(1.0, sum(len(t) for t in cd))
    long_hits = sum(1 for t in hits if len(t) >= 8)
    tw = sum(tok_w.get(t, 0.0) for t in set(hits)) if tok_w else 0.0
    # unique hit: title token in query that appears in few other cands — filled later
    y = int(year)
    years_in_q = [int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
    return np.array(
        [
            len(hits) / max(1, len(cd)),
            cov,
            soft_cov,
            len(soft) / max(1, len(cd)),
            acr_hits / max(1, len(acr)),
            float(acr_hits > 0),
            float(phrase2),
            float(phrase3),
            float(phrase2 + phrase3 > 0),
            inter4 / max(1, len(cg4)),
            inter5 / max(1, len(cg5)),
            math.log1p(inter4),
            math.log1p(inter5),
            float(year in query),
            float(bool(years_in_q) and min(abs(y - yq) for yq in years_in_q) <= 1),
            float(long_hits),
            float(len(hits) >= 2),
            float(len(hits) >= 3),
            float(tl in ql),
            math.log1p(max(tw, 0.0)),
            float(tw > 0),
            math.log1p(len(hits)),
            float(max((len(t) for t in hits), default=0) >= 7),
            (y - 1990) / 40.0,
            # note-focused coverage
            sum(1 for t in cd if t in query.split("\n", 1)[0].lower()) / max(1, len(cd)),
            sum(1 for t in cd if t in "\n".join(query.split("\n")[:2]).lower()) / max(1, len(cd)),
        ],
        dtype=np.float32,
    )


def learn_tok_w(tr):
    pos, neg = Counter(), Counter()
    for _, row in tr.iterrows():
        q = build_query(row).lower()
        gold = json.loads(row["provenance_card"])["source_title"]
        for c in REF_COLS:
            title, _ = parse(row[c])
            toks = [t for t in distinctive(tokenize(title)) if t in q]
            (pos if title == gold else neg).update(toks)
    return {
        t: math.log1p(pos[t]) - math.log1p(neg[t]) + 0.2 * math.log1p(pos[t])
        for t in set(pos) | set(neg)
    }


def metric(pred_idx, conf, df):
    rows = df.reset_index(drop=True)
    title_ok = both = 0
    ybin = []
    for i, row in rows.iterrows():
        card = json.loads(row["provenance_card"])
        t, y = parse(row[REF_COLS[pred_idx[i]]])
        title_ok += int(t == card["source_title"])
        both += int(t == card["source_title"] and y == str(card["source_year"]))
        ybin.append(float(t == card["source_title"] and y == str(card["source_year"])))
    n = len(rows)
    base = both / n
    conf = np.asarray(conf) / 100.0
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
    tr = train[train.src_year < 2011].reset_index(drop=True)
    va = train[train.src_year >= 2011].reset_index(drop=True)
    print("split", len(tr), len(va), device, flush=True)
    tok_w = learn_tok_w(tr)

    def build(df):
        base = np.zeros((len(df), 16, 26), dtype=np.float32)
        groups = np.zeros((len(df), 16), dtype=np.int64)
        for i, row in df.iterrows():
            raws = [str(row[c]) for c in REF_COLS]
            # groups
            m = {}
            for j, raw in enumerate(raws):
                t, _ = parse(raw)
                if t not in m:
                    m[t] = len(m)
                groups[i, j] = m[t]
                base[i, j] = raw_feats(row["query"], raw, tok_w)
            # unique-token bonus: hits not shared with other cands
            for j, raw in enumerate(raws):
                title, _ = parse(raw)
                cd = set(distinctive(tokenize(title)))
                others = set()
                for k, raw2 in enumerate(raws):
                    if k == j:
                        continue
                    others |= set(distinctive(tokenize(parse(raw2)[0])))
                uniq = [t for t in cd - others if t in row["query"].lower()]
                # append relative later
        # slate-relative expansion
        rel = np.zeros((len(df), 16, 10), dtype=np.float32)
        for i in range(len(df)):
            for f in range(26):
                col = base[i, :, f]
                mx, mn, mean = col.max(), col.min(), col.mean()
                # store only for key features into rel channels
            # key feature indices for relative
            for j in range(16):
                for rki, fi in enumerate([0, 1, 2, 4, 6, 7, 9, 10, 15, 19]):
                    col = base[i, :, fi]
                    v = base[i, j, fi]
                    rel[i, j, rki] = (v - col.mean()) / (col.std() + 1e-6)
        # unique hit features
        uniqf = np.zeros((len(df), 16, 2), dtype=np.float32)
        for i, row in df.iterrows():
            raws = [str(row[c]) for c in REF_COLS]
            ql = row["query"].lower()
            tok_sets = [set(distinctive(tokenize(parse(r)[0]))) for r in raws]
            for j in range(16):
                others = set().union(*[tok_sets[k] for k in range(16) if k != j]) if True else set()
                uniq = [t for t in tok_sets[j] - others if t in ql]
                uniqf[i, j, 0] = len(uniq)
                uniqf[i, j, 1] = sum(len(t) for t in uniq) / 20.0
        X = np.concatenate([base, rel, uniqf], axis=-1)
        return X, groups

    print("building...", flush=True)
    Xtr, Gtr = build(tr)
    Xva, Gva = build(va)
    print("X", Xtr.shape, flush=True)
    d = Xtr.shape[-1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 160),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(160, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    best_global = 0.0
    best_m = None
    all_va = []
    for seed in range(5):
        seed_everything = lambda s: (random.seed(s), np.random.seed(s), torch.manual_seed(s))
        seed_everything(seed * 11 + 3)
        model = MLP().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.03)
        best_state, best_sc = None, -1
        for epoch in range(20):
            model.train()
            idx = np.random.permutation(len(tr))
            for start in range(0, len(tr), 64):
                bi = idx[start : start + 64]
                x = torch.tensor(Xtr[bi], device=device)
                scores = model(x)
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
            model.eval()
            with torch.no_grad():
                sv = model(torch.tensor(Xva, device=device)).cpu()
                preds, confs = [], []
                for i in range(len(va)):
                    s = sv[i]
                    g = torch.tensor(Gva[i])
                    uniq = sorted(set(Gva[i].tolist()))
                    gs = torch.stack([torch.logsumexp(s[g == u], 0) for u in uniq])
                    p = F.softmax(gs, 0)
                    bg = uniq[int(p.argmax())]
                    mask = np.where(Gva[i] == bg)[0]
                    preds.append(int(mask[int(s[mask].argmax())]))
                    confs.append(float(p.max() * 100))
                m = metric(preds, confs, va)
            if m["score"] > best_sc:
                best_sc = m["score"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch < 6 or (epoch + 1) % 4 == 0:
                print(f"seed{seed} ep{epoch+1} {m} best={best_sc:.4f}", flush=True)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            all_va.append(model(torch.tensor(Xva, device=device)).cpu().numpy())
        best_global = max(best_global, best_sc)
        print(f"seed{seed} done best={best_sc:.4f} t={time.time()-t0:.0f}s", flush=True)

    ens = np.mean(all_va, axis=0)
    preds, confs = [], []
    for i in range(len(va)):
        s = torch.tensor(ens[i])
        g = torch.tensor(Gva[i])
        uniq = sorted(set(Gva[i].tolist()))
        gs = torch.stack([torch.logsumexp(s[g == u], 0) for u in uniq])
        p = F.softmax(gs, 0)
        bg = uniq[int(p.argmax())]
        mask = np.where(Gva[i] == bg)[0]
        preds.append(int(mask[int(s[mask].argmax())]))
        confs.append(float(p.max() * 100))
    m = metric(preds, confs, va)
    print("ENSEMBLE", m, "best_single", best_global, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
