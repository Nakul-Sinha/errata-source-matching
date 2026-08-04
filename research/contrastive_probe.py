"""Contrastive char dual-encoder + train-memory retrieval probe (offline)."""
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

REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
DATA = Path(r"G:\Datacurve\gpuchals\newone\dataset\public")


def parse_ref(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def build_query(row, max_chars=2000):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(300, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


def encode_chars(text, limit):
    ids = []
    for ch in str(text)[:limit]:
        o = ord(ch)
        ids.append(o - 30 if 32 <= o <= 126 else 1)
    return ids or [1]


class CharEnc(nn.Module):
    def __init__(self, d=192):
        super().__init__()
        self.emb = nn.Embedding(98, 48, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(48, 96, k, padding=k // 2) for k in (3, 5, 7)])
        self.proj = nn.Linear(96 * 3, d)
        self.d = d

    def forward(self, ids, mask):
        x = self.emb(ids).transpose(1, 2)
        xs = []
        for conv in self.convs:
            h = F.gelu(conv(x))
            h = h.masked_fill(~mask.unsqueeze(1), -1e4)
            xs.append(h.max(-1).values)
        h = self.proj(torch.cat(xs, -1))
        return F.normalize(h, dim=-1)


def pad_batch(texts, limit, device):
    seqs = [encode_chars(t, limit) for t in texts]
    L = limit
    ids = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        n = min(L, len(s))
        ids[i, :n] = torch.tensor(s[:n], device=device)
        mask[i, :n] = True
    return ids, mask


def main():
    t0 = time.time()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = pd.read_csv(DATA / "train.csv")

    def label_index(row):
        card = json.loads(row["provenance_card"])
        for i, col in enumerate(REF_COLS):
            t, y = parse_ref(row[col])
            if t == card["source_title"] and y == str(card["source_year"]):
                return i
        raise ValueError(row["id"])

    train["label"] = train.apply(label_index, axis=1)
    train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
    train["query"] = train.apply(lambda r: build_query(r), axis=1)
    train["gold_title"] = train["provenance_card"].apply(lambda s: json.loads(s)["source_title"])
    train["gold_year"] = train["provenance_card"].apply(lambda s: str(json.loads(s)["source_year"]))
    train["gold_str"] = train["gold_title"] + " year " + train["gold_year"]

    tr = train[train.src_year < 2011].reset_index(drop=True)
    va = train[train.src_year >= 2011].reset_index(drop=True)
    print("split", len(tr), len(va), "device", device, flush=True)

    enc = CharEnc(192).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-3, weight_decay=0.01)

    def contrastive_step(df, bs=48):
        idx = np.random.choice(len(df), size=min(bs, len(df)), replace=False)
        qs = [df.loc[i, "query"] for i in idx]
        ts = [df.loc[i, "gold_str"] for i in idx]
        # also hard in-slate negatives from same rows' distractors
        q_ids, q_m = pad_batch(qs, 768, device)
        t_ids, t_m = pad_batch(ts, 160, device)
        q = enc(q_ids, q_m)
        t = enc(t_ids, t_m)
        logits = q @ t.T * 15.0
        labels = torch.arange(len(idx), device=device)
        loss = F.cross_entropy(logits, labels)
        # slate loss on a subset
        slate_loss = 0.0
        n_slate = min(16, len(idx))
        for j in range(n_slate):
            i = int(idx[j])
            cands = [str(df.loc[i, c]) for c in REF_COLS]
            cands = [parse_ref(c)[0] + " year " + parse_ref(c)[1] for c in cands]
            c_ids, c_m = pad_batch(cands, 160, device)
            c = enc(c_ids, c_m)
            s = (q[j : j + 1] @ c.T).squeeze(0) * 15.0
            # group by title
            titles = [parse_ref(str(df.loc[i, c]))[0] for c in REF_COLS]
            gold_t = df.loc[i, "gold_title"]
            # title group logsumexp
            uniq = []
            gmap = []
            for tname in titles:
                if tname not in uniq:
                    uniq.append(tname)
                gmap.append(uniq.index(tname))
            gmap = torch.tensor(gmap, device=device)
            max_g = len(uniq)
            gs = s.new_full((max_g,), -1e4)
            for u in range(max_g):
                gs[u] = torch.logsumexp(s[gmap == u], 0)
            lab = uniq.index(gold_t)
            slate_loss = slate_loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([lab], device=device))
        slate_loss = slate_loss / n_slate
        return loss + 0.7 * slate_loss

    best = 0.0
    for epoch in range(25):
        enc.train()
        tot = 0.0
        steps = max(1, len(tr) // 48)
        for _ in range(steps):
            loss = contrastive_step(tr, 48)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        # eval
        enc.eval()
        with torch.no_grad():
            # encode all train gold titles for memory
            tr_titles = tr["gold_str"].tolist()
            # chunk
            mem = []
            for i in range(0, len(tr_titles), 128):
                ids, m = pad_batch(tr_titles[i : i + 128], 160, device)
                mem.append(enc(ids, m))
            mem = torch.cat(mem, 0)  # (Ntr, D)
            tr_gold_titles = tr["gold_title"].tolist()

            title_ok = 0
            both_ok = 0
            for _, row in va.iterrows():
                q = row["query"] if "query" in row else build_query(row)
                # rebuild query field
            # ensure query col
            if "query" not in va.columns:
                va = va.copy()
                va["query"] = va.apply(build_query, axis=1)

            for i, row in va.iterrows():
                q_ids, q_m = pad_batch([row["query"]], 768, device)
                qv = enc(q_ids, q_m)  # (1,D)
                cands_raw = [str(row[c]) for c in REF_COLS]
                cands = [parse_ref(c)[0] + " year " + parse_ref(c)[1] for c in cands_raw]
                c_ids, c_m = pad_batch(cands, 160, device)
                cv = enc(c_ids, c_m)
                # direct slate scores
                direct = (qv @ cv.T).squeeze(0)
                # memory: retrieve top train queries... need train query embeddings
            # build train query memory once per epoch
            qmem = []
            for i in range(0, len(tr), 64):
                qs = tr["query"].iloc[i : i + 64].tolist()
                ids, m = pad_batch(qs, 768, device)
                qmem.append(enc(ids, m))
            qmem = torch.cat(qmem, 0)

            for _, row in va.iterrows():
                q_ids, q_m = pad_batch([row["query"]], 768, device)
                qv = enc(q_ids, q_m)
                cands_raw = [str(row[c]) for c in REF_COLS]
                cands = [parse_ref(c)[0] + " year " + parse_ref(c)[1] for c in cands_raw]
                titles = [parse_ref(c)[0] for c in cands_raw]
                years = [parse_ref(c)[1] for c in cands_raw]
                c_ids, c_m = pad_batch(cands, 160, device)
                cv = enc(c_ids, c_m)
                direct = (qv @ cv.T).squeeze(0)
                # retrieve top-32 train queries
                sim = (qv @ qmem.T).squeeze(0)
                topk = sim.topk(min(32, len(tr))).indices
                # retrieved gold title embeddings
                retrieved = mem[topk]  # (K,D)
                # score each candidate by max/mean sim to retrieved golds + direct
                mem_score = (cv @ retrieved.T).mean(-1)
                scores = direct * 1.0 + 0.35 * mem_score
                # title group decode
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
                best_g = int(gs.argmax())
                mask = [j for j, g in enumerate(gmap) if g == best_g]
                best_j = mask[int(scores[mask].argmax())]
                card = json.loads(row["provenance_card"])
                title_ok += int(titles[best_j] == card["source_title"])
                both_ok += int(
                    titles[best_j] == card["source_title"] and years[best_j] == str(card["source_year"])
                )
            tr_rate = title_ok / len(va)
            both = both_ok / len(va)
            score = 0.85 * tr_rate + 0.05 * both
            best = max(best, score)
            print(
                f"epoch {epoch+1} title={tr_rate:.4f} both={both:.4f} score~={score:.4f} "
                f"best~={best:.4f} loss={tot/steps:.4f} t={time.time()-t0:.0f}s",
                flush=True,
            )

    print("DONE best~", best, flush=True)


if __name__ == "__main__":
    main()
