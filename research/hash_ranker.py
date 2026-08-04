"""Feature-hashing char/word n-gram ranker (NOT TF-IDF). Offline, from-scratch."""
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

DIM = 2**18  # hashing dim


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def tokenize(text):
    return [t.lower().replace("\u2019", "'") for t in TOKEN_RE.findall(str(text))]


def build_query(row, max_chars=2200):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(300, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


def murmur(s: str) -> int:
    # FNV-1a
    h = 2166136261
    for ch in s.encode("utf-8", errors="ignore"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def hash_feats(query: str, title: str, year: str) -> dict[int, float]:
    """Pairwise interaction hashing: query grams × title grams + unary."""
    q = query.lower()
    t = title.lower()
    feats: dict[int, float] = {}

    def add(key: str, val: float = 1.0):
        i = murmur(key) % DIM
        feats[i] = feats.get(i, 0.0) + val

    q_toks = tokenize(q)
    t_toks = tokenize(t)
    q_set = set(q_toks)
    # unary title tokens present in query
    for tok in t_toks:
        if len(tok) < 3:
            continue
        if tok in q_set or tok in q:
            add(f"hit:{tok}", 1.0 + 0.1 * len(tok))
            add(f"hitlen:{min(len(tok), 12)}")
    # bigrams of title in query
    for i in range(len(t_toks) - 1):
        bg = t_toks[i] + "_" + t_toks[i + 1]
        if t_toks[i] in q and t_toks[i + 1] in q:
            add(f"bg:{bg}", 2.0)
        phrase = t_toks[i] + " " + t_toks[i + 1]
        if phrase in q:
            add(f"ph:{bg}", 3.0)
    # char ngrams intersection
    def cgrams(s, n):
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}

    for n in (3, 4, 5):
        inter = cgrams(q, n) & cgrams(t, n)
        add(f"cg{n}_cnt", math.log1p(len(inter)))
        for g in list(inter)[:40]:
            add(f"cg{n}:{g}", 0.5)
    # acronyms
    for a in re.findall(r"\b[A-Z]{2,10}\b", title):
        if a.lower() in q_set:
            add(f"acr:{a.lower()}", 2.0)
    if year in query:
        add("year_exact", 1.5)
    add("bias", 1.0)
    # normalize
    norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    return {k: v / norm for k, v in feats.items()}


def to_sparse_batch(feat_dicts, device):
    """Return dense (B,16,D) is too big — use EmbeddingBag style: indices+values."""
    # We'll score with a weight vector via sparse matmul manually
    return feat_dicts


class HashLinear(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.w = nn.Embedding(dim, 1)
        nn.init.zeros_(self.w.weight)

    def score(self, feat_dict: dict[int, float]) -> torch.Tensor:
        if not feat_dict:
            return self.w.weight.new_zeros(())
        idx = torch.tensor(list(feat_dict.keys()), dtype=torch.long, device=self.w.weight.device)
        val = torch.tensor(list(feat_dict.values()), dtype=torch.float32, device=self.w.weight.device)
        return (self.w(idx).squeeze(-1) * val).sum()


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
    print("split", len(tr), len(va), device, flush=True)

    print("precomputing features...", flush=True)
    def row_feats(df):
        out = []
        for _, row in df.iterrows():
            fs = []
            for c in REF_COLS:
                title, year = parse(row[c])
                fs.append(hash_feats(row["query"], title, year))
            out.append(fs)
        return out

    Ftr = row_feats(tr)
    Fva = row_feats(va)
    print("done feats", time.time() - t0, flush=True)

    model = HashLinear().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-5)
    best = 0.0
    best_m = None
    for epoch in range(25):
        model.train()
        idx = list(range(len(tr)))
        random.shuffle(idx)
        tot = 0.0
        for start in range(0, len(idx), 32):
            batch = idx[start : start + 32]
            loss = 0.0
            for i in batch:
                scores = torch.stack([model.score(Ftr[i][j]) for j in range(16)])
                raws = [str(tr.loc[i, c]) for c in REF_COLS]
                titles = [parse(r)[0] for r in raws]
                gold = json.loads(tr.loc[i, "provenance_card"])["source_title"]
                uniq = []
                gmap = []
                for t in titles:
                    if t not in uniq:
                        uniq.append(t)
                    gmap.append(uniq.index(t))
                gmap_t = torch.tensor(gmap, device=device)
                gs = scores.new_full((len(uniq),), -1e4)
                for u in range(len(uniq)):
                    gs[u] = torch.logsumexp(scores[gmap_t == u], 0)
                loss = loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([uniq.index(gold)], device=device))
                loss = loss + 0.2 * F.cross_entropy(
                    scores.unsqueeze(0), torch.tensor([int(tr.loc[i, "label"])], device=device)
                )
            loss = loss / len(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        # eval
        model.eval()
        preds, confs = [], []
        with torch.no_grad():
            for i in range(len(va)):
                scores = torch.stack([model.score(Fva[i][j]) for j in range(16)])
                raws = [str(va.loc[i, c]) for c in REF_COLS]
                titles = [parse(r)[0] for r in raws]
                uniq = []
                gmap = []
                for t in titles:
                    if t not in uniq:
                        uniq.append(t)
                    gmap.append(uniq.index(t))
                gmap_t = torch.tensor(gmap, device=device)
                gs = scores.new_full((len(uniq),), -1e4)
                for u in range(len(uniq)):
                    gs[u] = torch.logsumexp(scores[gmap_t == u], 0)
                p = F.softmax(gs, 0)
                best_g = int(p.argmax())
                mask = [j for j, g in enumerate(gmap) if g == best_g]
                preds.append(mask[int(scores[mask].argmax())])
                confs.append(float(p.max() * 100))
        m = metric(preds, confs, va)
        if m["score"] > best:
            best = m["score"]
            best_m = m
        print(f"epoch {epoch+1} {m} best={best:.4f} loss={tot/(len(idx)/32):.4f} t={time.time()-t0:.0f}s", flush=True)
    print("DONE", best_m, flush=True)


if __name__ == "__main__":
    main()
