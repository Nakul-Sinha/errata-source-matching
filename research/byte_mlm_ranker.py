"""From-scratch byte-level Transformer: MLM pretrain then slate ranking.

No HF, no TF-IDF, no internet. Designed for OOD-stable compositional matching.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[1] / "dataset" / "public"
REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
import re

REF_RE = re.compile(r"^(.*) \((\d{4})\)$")

PAD, MASK, CLS, SEP = 0, 1, 2, 3
# bytes 0-255 mapped to 4..259


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def build_query(row, max_chars=1200):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(200, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


def to_bytes(text: str, limit: int, add_cls=False) -> list[int]:
    raw = text.encode("utf-8", errors="ignore")[:limit]
    ids = [b + 4 for b in raw]
    if add_cls:
        ids = [CLS] + ids + [SEP]
    return ids or [CLS, SEP]


class ByteTM(nn.Module):
    def __init__(self, d=160, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(260, d, padding_idx=PAD)
        self.pos = nn.Embedding(512, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, ids, mask):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
        x = self.emb(ids) + self.pos(pos.clamp(max=511))
        x = self.enc(x, src_key_padding_mask=~mask)
        return self.norm(x)


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
    tr = train[train.src_year < 2011].reset_index(drop=True)
    va = train[train.src_year >= 2011].reset_index(drop=True)
    print("split", len(tr), len(va), device, flush=True)

    enc = ByteTM().to(device)
    mlm = nn.Linear(enc.d, 260).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(mlm.parameters()), lr=3e-4, weight_decay=0.01)

    corpus = list(tr["query"])
    for _, row in tr.iterrows():
        for c in REF_COLS:
            t, y = parse(row[c])
            corpus.append(f"{t} ({y})")
    print("mlm corpus", len(corpus), flush=True)

    for ep in range(8):
        random.shuffle(corpus)
        tot = 0.0
        steps = 0
        for start in range(0, len(corpus), 24):
            batch = corpus[start : start + 24]
            seqs = [to_bytes(t, 256, True)[:256] for t in batch]
            L = max(len(s) for s in seqs)
            ids = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
            mask = torch.zeros(len(seqs), L, dtype=torch.bool, device=device)
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s, device=device)
                mask[i, : len(s)] = True
            labels = ids.clone()
            prob = torch.full(ids.shape, 0.15, device=device) * mask.float()
            sel = torch.bernoulli(prob).bool()
            sel[:, 0] = False
            inp = ids.clone()
            inp[sel] = MASK
            h = enc(inp, mask)
            logits = mlm(h)
            loss = F.cross_entropy(logits[sel], labels[sel]) if sel.any() else h.sum() * 0
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(mlm.parameters()), 1.0)
            opt.step()
            tot += float(loss)
            steps += 1
        print(f"mlm ep{ep+1} loss={tot/max(1,steps):.4f} t={time.time()-t0:.0f}s", flush=True)

    # ranking head on CLS of [CLS] q [SEP] title [SEP]
    head = nn.Sequential(nn.Linear(enc.d, enc.d), nn.GELU(), nn.Linear(enc.d, 1)).to(device)
    # simple lexical feat residual
    feat_w = nn.Linear(6, 1).to(device)
    opt2 = torch.optim.AdamW(
        [
            {"params": enc.parameters(), "lr": 1e-4},
            {"params": head.parameters(), "lr": 1e-3},
            {"params": feat_w.parameters(), "lr": 2e-3},
        ],
        weight_decay=0.01,
    )

    def lex(query, raw):
        title, year = parse(raw)
        ql = query.lower()
        import re as _re
        cd = [t for t in _re.findall(r"[a-z]{4,}", title.lower())]
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

    def pack(queries, cands):
        # return ids (B,16,L), mask, feats (B,16,6)
        B = len(queries)
        seqs = []
        feats = []
        for q, cs in zip(queries, cands):
            row_f = []
            for c in cs:
                q_ids = to_bytes(q, 200)
                c_ids = to_bytes(c, 80)
                seq = ([CLS] + q_ids + [SEP] + c_ids + [SEP])[:320]
                seqs.append(seq)
            # feats filled by caller
        # reshape later
        return seqs

    best = 0.0
    for epoch in range(10):
        enc.train()
        head.train()
        idx = list(range(len(tr)))
        random.shuffle(idx)
        tot = 0.0
        steps = 0
        for start in range(0, len(idx), 4):
            batch = idx[start : start + 4]
            queries = [tr.loc[i, "query"] for i in batch]
            cands = []
            f_all = []
            groups = []
            golds = []
            for i in batch:
                row = tr.loc[i]
                raws = [str(row[c]) for c in REF_COLS]
                cands.append([parse(r)[0] + " (" + parse(r)[1] + ")" for r in raws])
                f_all.append([lex(row["query"], r) for r in raws])
                groups.append(title_groups(raws))
                golds.append(json.loads(row["provenance_card"])["source_title"])
            # pack
            flat = []
            for q, cs in zip(queries, cands):
                for c in cs:
                    q_ids = to_bytes(q, 180)
                    c_ids = to_bytes(c, 70)
                    flat.append(([CLS] + q_ids + [SEP] + c_ids + [SEP])[:280])
            L = max(len(s) for s in flat)
            ids = torch.zeros(len(flat), L, dtype=torch.long, device=device)
            mask = torch.zeros(len(flat), L, dtype=torch.bool, device=device)
            for i, s in enumerate(flat):
                ids[i, : len(s)] = torch.tensor(s, device=device)
                mask[i, : len(s)] = True
            h = enc(ids, mask)[:, 0]
            ce_scores = head(h).view(len(batch), 16)
            f = torch.tensor(f_all, dtype=torch.float32, device=device)
            scores = ce_scores + feat_w(f).squeeze(-1) * 3.0
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
            opt2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()) + list(feat_w.parameters()), 1.0)
            opt2.step()
            tot += float(loss)
            steps += 1
        # eval
        enc.eval()
        head.eval()
        preds, confs = [], []
        with torch.no_grad():
            for start in range(0, len(va), 4):
                batch = list(range(start, min(len(va), start + 4)))
                queries = [va.loc[i, "query"] for i in batch]
                cands, f_all, groups = [], [], []
                for i in batch:
                    row = va.loc[i]
                    raws = [str(row[c]) for c in REF_COLS]
                    cands.append([parse(r)[0] + " (" + parse(r)[1] + ")" for r in raws])
                    f_all.append([lex(row["query"], r) for r in raws])
                    groups.append(title_groups(raws))
                flat = []
                for q, cs in zip(queries, cands):
                    for c in cs:
                        q_ids = to_bytes(q, 180)
                        c_ids = to_bytes(c, 70)
                        flat.append(([CLS] + q_ids + [SEP] + c_ids + [SEP])[:280])
                L = max(len(s) for s in flat)
                ids = torch.zeros(len(flat), L, dtype=torch.long, device=device)
                mask = torch.zeros(len(flat), L, dtype=torch.bool, device=device)
                for i, s in enumerate(flat):
                    ids[i, : len(s)] = torch.tensor(s, device=device)
                    mask[i, : len(s)] = True
                h = enc(ids, mask)[:, 0]
                ce_scores = head(h).view(len(batch), 16)
                f = torch.tensor(f_all, dtype=torch.float32, device=device)
                scores = ce_scores + feat_w(f).squeeze(-1) * 3.0
                for bi in range(len(batch)):
                    s = scores[bi]
                    gmap = groups[bi]
                    uniq = sorted(set(gmap))
                    gs = torch.stack([torch.logsumexp(s[torch.tensor(gmap, device=device) == u], 0) for u in uniq])
                    p = F.softmax(gs, 0)
                    bg = uniq[int(p.argmax())]
                    mask_j = [j for j, g in enumerate(gmap) if g == bg]
                    preds.append(mask_j[int(s[mask_j].argmax())])
                    confs.append(float(p.max() * 100))
        m = metric(preds, confs, va)
        best = max(best, m["score"])
        print(f"rank ep{epoch+1} {m} best={best:.4f} loss={tot/max(1,steps):.4f} t={time.time()-t0:.0f}s", flush=True)
    print("DONE best", best, flush=True)


if __name__ == "__main__":
    main()
