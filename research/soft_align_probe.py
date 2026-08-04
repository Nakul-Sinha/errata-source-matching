"""Soft token-alignment ranker from scratch (dot-product attention match)."""
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
}


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def tokenize(text):
    return [t.lower().replace("\u2019", "'") for t in TOKEN_RE.findall(str(text))]


def build_query(row, max_chars=2000):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(300, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


class SoftAlign(nn.Module):
    def __init__(self, vocab, d=96, max_q=180, max_c=36):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), d, padding_idx=0)
        self.pos_q = nn.Embedding(max_q, d)
        self.pos_c = nn.Embedding(max_c, d)
        self.d = d
        self.max_q = max_q
        self.max_c = max_c
        self.vocab = vocab
        self.feat_mlp = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 1))
        self.out = nn.Linear(d * 3 + 1, 1)
        self.drop = nn.Dropout(0.1)

    def encode_ids(self, ids, pos_emb):
        # ids: (B, L)
        mask = ids != 0
        x = self.emb(ids) + pos_emb(torch.arange(ids.size(1), device=ids.device).clamp(max=pos_emb.num_embeddings - 1))
        x = x * mask.unsqueeze(-1)
        return x, mask

    def forward(self, q_ids, c_ids, feats):
        # q: (B,Lq), c: (B,16,Lc), feats: (B,16,8)
        B, Lq = q_ids.shape
        q, qm = self.encode_ids(q_ids, self.pos_q)  # B,Lq,d
        c_ids_f = c_ids.view(B * 16, -1)
        c, cm = self.encode_ids(c_ids_f, self.pos_c)  # B*16,Lc,d
        c = c.view(B, 16, -1, self.d)
        cm = cm.view(B, 16, -1)
        # attention: for each cand token, max sim to query tokens
        # scores_bt = einsum q with c
        # q: B,Lq,d ; c: B,16,Lc,d
        sim = torch.einsum("bqd,bckd->bcqk", q, c) / math.sqrt(self.d)  # B,16,Lq,Lc
        sim = sim.masked_fill(~qm.unsqueeze(1).unsqueeze(-1), -1e4)
        sim = sim.masked_fill(~cm.unsqueeze(2), -1e4)
        # soft align: cand->query attention pool
        attn = F.softmax(sim, dim=2)  # over query
        # aligned query for each cand token
        # attn: B,16,Lq,Lc ; q: B,Lq,d -> want B,16,Lc,d
        q_exp = q.unsqueeze(1).unsqueeze(-2)  # B,1,Lq,1,d — messy
        # use max-sim pooling features instead (faster/stabler)
        max_over_q = sim.max(2).values  # B,16,Lc
        max_over_q = max_over_q.masked_fill(~cm, 0)
        mean_match = max_over_q.sum(-1) / cm.sum(-1).clamp(min=1)  # B,16
        max_match = max_over_q.max(-1).values
        # also pool embeddings
        q_pool = q.sum(1) / qm.sum(1).clamp(min=1).unsqueeze(-1)
        c_pool = c.sum(2) / cm.sum(-1).clamp(min=1).unsqueeze(-1)
        q_pool = F.normalize(q_pool, dim=-1)
        c_pool = F.normalize(c_pool, dim=-1)
        cos = (q_pool.unsqueeze(1) * c_pool).sum(-1)
        feat_s = self.feat_mlp(feats).squeeze(-1)
        # combine match signals via small MLP on concatenated
        combo = torch.stack([mean_match, max_match, cos, feat_s], -1)  # B,16,4
        # expand with abs diffs of pools
        diff = (q_pool.unsqueeze(1) - c_pool).abs().mean(-1)
        prod = cos
        h = torch.cat(
            [
                q_pool.unsqueeze(1).expand(-1, 16, -1),
                c_pool,
                (q_pool.unsqueeze(1) * c_pool),
                feat_s.unsqueeze(-1),
            ],
            -1,
        )
        scores = self.out(self.drop(h)).squeeze(-1) + mean_match + 0.5 * max_match + feat_s
        return scores


def overlap_feats(query, raw):
    title, year = parse(raw)
    qt, ct = tokenize(query), tokenize(title)
    qs, cs = set(qt), set(ct)
    ql = query.lower()
    cd = [t for t in ct if len(t) >= 4 and t not in STOP]
    hits = [t for t in cd if t in ql]
    acr = {a.lower() for a in re.findall(r"\b[A-Z]{2,10}\b", title)}
    return [
        len(hits) / max(1, len(cd)),
        sum(len(t) for t in hits) / max(1.0, sum(len(t) for t in cd)),
        len(acr & qs) / max(1, len(acr)),
        float(year in query),
        float(len(hits) >= 2),
        math.log1p(len(hits)),
        float(any(" ".join(cd[i : i + 2]) in ql for i in range(max(0, len(cd) - 1)))),
        float(title.lower() in ql),
    ]


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

    counts = Counter()
    for q in tr["query"]:
        counts.update(tokenize(q))
    for _, row in tr.iterrows():
        for c in REF_COLS:
            counts.update(tokenize(str(row[c])))
    vocab = {"<pad>": 0, "<unk>": 1}
    for t, f in counts.most_common(25000):
        if f >= 2 and t not in vocab:
            vocab[t] = len(vocab)
    print("vocab", len(vocab), flush=True)

    def enc_text(text, limit):
        ids = [vocab.get(t, 1) for t in tokenize(text)[:limit]]
        return ids or [1]

    max_q, max_c = 180, 36
    model = SoftAlign(vocab, 96, max_q, max_c).to(device)
    # freeze emb initially? train all carefully
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    def batchify(df, indices):
        qs, cs, fs, labels, groups = [], [], [], [], []
        for i in indices:
            row = df.loc[i]
            qs.append(enc_text(row["query"], max_q))
            cids = []
            feats = []
            raws = [str(row[c]) for c in REF_COLS]
            for raw in raws:
                title, year = parse(raw)
                cids.append(enc_text(title + " " + year, max_c))
                feats.append(overlap_feats(row["query"], raw))
            cs.append(cids)
            fs.append(feats)
            labels.append(int(row["label"]))
            # groups
            m, g = {}, []
            for raw in raws:
                t, _ = parse(raw)
                if t not in m:
                    m[t] = len(m)
                g.append(m[t])
            groups.append(g)
        B = len(indices)
        q = torch.zeros(B, max_q, dtype=torch.long, device=device)
        c = torch.zeros(B, 16, max_c, dtype=torch.long, device=device)
        for i, ids in enumerate(qs):
            q[i, : len(ids)] = torch.tensor(ids, device=device)
        for i, cids in enumerate(cs):
            for j, ids in enumerate(cids):
                c[i, j, : len(ids)] = torch.tensor(ids, device=device)
        f = torch.tensor(fs, dtype=torch.float32, device=device)
        return q, c, f, labels, groups

    def eval_df(df):
        model.eval()
        title_ok = both = 0
        confs = []
        with torch.no_grad():
            for start in range(0, len(df), 16):
                idx = list(range(start, min(len(df), start + 16)))
                q, c, f, _, groups = batchify(df, idx)
                scores = model(q, c, f)
                for bi, i in enumerate(idx):
                    s = scores[bi]
                    g = groups[bi]
                    uniq = sorted(set(g))
                    gs = s.new_full((len(uniq),), -1e4)
                    gmap = torch.tensor(g, device=device)
                    for u, ug in enumerate(uniq):
                        gs[u] = torch.logsumexp(s[gmap == ug], 0)
                    p = F.softmax(gs, 0)
                    best_g = uniq[int(p.argmax())]
                    mask = [j for j, gg in enumerate(g) if gg == best_g]
                    best_j = mask[int(s[mask].argmax())]
                    card = json.loads(df.loc[i, "provenance_card"])
                    t, y = parse(df.loc[i, REF_COLS[best_j]])
                    title_ok += int(t == card["source_title"])
                    both += int(t == card["source_title"] and y == str(card["source_year"]))
                    confs.append(float(p.max() * 100))
        n = len(df)
        tr = title_ok / n
        br = both / n
        conf = np.asarray(confs) / 100
        ybin = np.zeros(n)  # approximate skip exact cal rebuild
        # quick cal
        base = br
        # rebuild ybin properly
        # skip heavy — use both rate proxy
        cal = 0.0
        score = 0.85 * tr + 0.05 * br + 0.10 * cal
        return {"title_rate": tr, "both_rate": br, "score": score}

    best = 0.0
    for epoch in range(12):
        model.train()
        idx = list(range(len(tr)))
        random.shuffle(idx)
        tot = 0.0
        steps = 0
        for start in range(0, len(idx), 12):
            batch = idx[start : start + 12]
            q, c, f, labels, groups = batchify(tr, batch)
            scores = model(q, c, f)
            loss = 0.0
            for bi, i in enumerate(batch):
                s = scores[bi]
                g = groups[bi]
                gold_title = json.loads(tr.loc[i, "provenance_card"])["source_title"]
                titles = [parse(tr.loc[i, col])[0] for col in REF_COLS]
                uniq = []
                gmap = []
                for t in titles:
                    if t not in uniq:
                        uniq.append(t)
                    gmap.append(uniq.index(t))
                gmap_t = torch.tensor(gmap, device=device)
                gs = s.new_full((len(uniq),), -1e4)
                for u in range(len(uniq)):
                    gs[u] = torch.logsumexp(s[gmap_t == u], 0)
                loss = loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([uniq.index(gold_title)], device=device))
                loss = loss + 0.25 * F.cross_entropy(s.unsqueeze(0), torch.tensor([labels[bi]], device=device))
            loss = loss / len(batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            steps += 1
        m = eval_df(va)
        best = max(best, m["score"])
        print(f"epoch {epoch+1} {m} best={best:.4f} loss={tot/max(1,steps):.4f} t={time.time()-t0:.0f}s", flush=True)
    print("DONE best", best, flush=True)


if __name__ == "__main__":
    main()
