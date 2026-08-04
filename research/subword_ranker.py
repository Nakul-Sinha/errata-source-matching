"""From-scratch subword skipgram + ranking head (FastText-like, offline).

Trains subword embeddings on train.csv text only, then a listwise ranker.
No internet, no external weights, no TF-IDF.
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

REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d{2,4}")
DATA = Path(r"G:\ml\gpuchals\newone\dataset\public")


def parse_ref(s):
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


def subwords(token: str, min_n=3, max_n=6) -> list[str]:
    t = f"<{token}>"
    grams = [t]
    for n in range(min_n, max_n + 1):
        for i in range(0, max(0, len(t) - n + 1)):
            grams.append(t[i : i + n])
    return grams


def title_group_ids(raws):
    m = {}
    out = []
    for raw in raws:
        t, _ = parse_ref(raw)
        if t not in m:
            m[t] = len(m)
        out.append(m[t])
    return out


class SubwordEmb(nn.Module):
    def __init__(self, n_grams: int, dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(n_grams, dim)
        self.dim = dim

    def forward_tokens(self, token_gram_ids: list[list[int]], device) -> torch.Tensor:
        # mean of all gram vectors across all tokens
        if not token_gram_ids:
            return torch.zeros(self.dim, device=device)
        ids = [g for grams in token_gram_ids for g in grams]
        if not ids:
            return torch.zeros(self.dim, device=device)
        v = self.emb(torch.tensor(ids, device=device, dtype=torch.long))
        return F.normalize(v.mean(0), dim=0)


def main():
    t0 = time.time()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    train = pd.read_csv(DATA / "train.csv")

    def label_index(row):
        card = json.loads(row["provenance_card"])
        for i, c in enumerate(REF_COLS):
            t, y = parse_ref(row[c])
            if t == card["source_title"] and y == str(card["source_year"]):
                return i
        raise ValueError()

    train["label"] = train.apply(label_index, axis=1)
    train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
    train["query"] = [build_query(r) for _, r in train.iterrows()]
    tr = train[train.src_year < 2011].reset_index(drop=True)
    va = train[train.src_year >= 2011].reset_index(drop=True)
    print("split", len(tr), len(va), flush=True)

    # Build subword vocab from train
    gram_counts = Counter()
    corpus_tokens = []
    for _, row in tr.iterrows():
        toks = tokenize(row["query"])
        for c in REF_COLS:
            toks += tokenize(str(row[c]))
        corpus_tokens.append(toks)
        for tok in toks:
            gram_counts.update(subwords(tok))
    # keep frequent grams
    grams = ["<pad>", "<unk>"] + [g for g, f in gram_counts.most_common(80000) if f >= 3]
    g2i = {g: i for i, g in enumerate(grams)}
    print("grams", len(g2i), flush=True)

    def tok_grams(tok: str) -> list[int]:
        ids = [g2i[g] for g in subwords(tok) if g in g2i]
        return ids or [1]

    emb = SubwordEmb(len(g2i), 128).to(device)
    # Skipgram-ish: predict context token grams from center — simplified as
    # contrastive between window bags.
    opt = torch.optim.AdamW(emb.parameters(), lr=2e-3)

    def skipgram_loss(batch_tok_lists, window=5):
        # sample centers and positives from same sentence; negatives from batch
        loss = 0.0
        n = 0
        for toks in batch_tok_lists:
            if len(toks) < 3:
                continue
            for i in range(0, len(toks), 3):
                center = emb.forward_tokens([tok_grams(toks[i])], device)
                pos_idx = [j for j in range(max(0, i - window), min(len(toks), i + window + 1)) if j != i]
                if not pos_idx:
                    continue
                j = random.choice(pos_idx)
                pos = emb.forward_tokens([tok_grams(toks[j])], device)
                # negatives: random tokens from other sentences
                neg_loss = 0.0
                for _ in range(3):
                    other = random.choice(batch_tok_lists)
                    if not other:
                        continue
                    nt = random.choice(other)
                    neg = emb.forward_tokens([tok_grams(nt)], device)
                    neg_loss = neg_loss + F.logsigmoid(-(center * neg).sum())
                loss = loss + F.logsigmoid((center * pos).sum()) + neg_loss / 3
                n += 1
        return -loss / max(1, n)

    print("pretrain subword...", flush=True)
    for ep in range(3):
        random.shuffle(corpus_tokens)
        tot = 0.0
        steps = 0
        for i in range(0, len(corpus_tokens), 32):
            batch = corpus_tokens[i : i + 32]
            loss = skipgram_loss(batch)
            if isinstance(loss, float):
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            steps += 1
        print(f"skipgram ep {ep+1} loss={tot/max(1,steps):.4f}", flush=True)

    # Ranking head on [cos, feat overlaps...]
    def feats(query, raw):
        title, year = parse_ref(raw)
        qt, ct = tokenize(query), tokenize(title)
        qs, cs = set(qt), set(ct)
        ql = query.lower()
        hits = [t for t in ct if len(t) >= 4 and t in ql]
        acr = {a.lower() for a in re.findall(r"\b[A-Z]{2,10}\b", title)}
        return [
            len(hits) / max(1, len([t for t in ct if len(t) >= 4])),
            sum(len(t) for t in hits) / max(1.0, sum(len(t) for t in ct if len(t) >= 4)),
            len(acr & qs) / max(1, len(acr)),
            float(year in query),
            float(len(hits) >= 2),
            math.log1p(len(hits)),
        ]

    class RankHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(1 + 6, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, cos, f):
            return self.mlp(torch.cat([cos.unsqueeze(-1), f], -1)).squeeze(-1)

    head = RankHead().to(device)
    opt2 = torch.optim.AdamW(list(emb.parameters()) + list(head.parameters()), lr=1e-3)

    def embed_text(text: str) -> torch.Tensor:
        toks = tokenize(text)
        return emb.forward_tokens([tok_grams(t) for t in toks], device)

    def score_row(row):
        q = row["query"]
        qv = embed_text(q)
        scores = []
        fs = []
        for c in REF_COLS:
            raw = str(row[c])
            title, year = parse_ref(raw)
            cv = embed_text(title + " year " + year)
            scores.append((qv * cv).sum())
            fs.append(feats(q, raw))
        cos = torch.stack(scores)
        f = torch.tensor(fs, dtype=torch.float32, device=device)
        return head(cos * 10, f)

    def eval_df(df):
        title_ok = both_ok = 0
        confs = []
        preds = []
        with torch.no_grad():
            for _, row in df.iterrows():
                s = score_row(row)
                raws = [str(row[c]) for c in REF_COLS]
                groups = title_group_ids(raws)
                # group decode
                uniq = []
                gmap = []
                for g in groups:
                    pass
                titles = [parse_ref(r)[0] for r in raws]
                years = [parse_ref(r)[1] for r in raws]
                uniq_t = []
                gmap = []
                for t in titles:
                    if t not in uniq_t:
                        uniq_t.append(t)
                    gmap.append(uniq_t.index(t))
                gmap_t = torch.tensor(gmap, device=device)
                gs = s.new_full((len(uniq_t),), -1e4)
                for u in range(len(uniq_t)):
                    gs[u] = torch.logsumexp(s[gmap_t == u], 0)
                best_g = int(gs.argmax())
                mask = [j for j, g in enumerate(gmap) if g == best_g]
                best_j = mask[int(s[mask].argmax())]
                card = json.loads(row["provenance_card"])
                title_ok += int(titles[best_j] == card["source_title"])
                both_ok += int(
                    titles[best_j] == card["source_title"] and years[best_j] == str(card["source_year"])
                )
                p = float(F.softmax(gs, 0)[best_g] * 100)
                confs.append(p)
                preds.append(best_j)
        tr = title_ok / len(df)
        both = both_ok / len(df)
        # rough cal skipped
        score = 0.85 * tr + 0.05 * both
        return tr, both, score

    best = 0.0
    for epoch in range(12):
        emb.train()
        head.train()
        tot = 0.0
        idx = list(range(len(tr)))
        random.shuffle(idx)
        for bi in range(0, len(idx), 16):
            batch = idx[bi : bi + 16]
            loss = 0.0
            for i in batch:
                row = tr.loc[i]
                s = score_row(row)
                raws = [str(row[c]) for c in REF_COLS]
                groups = title_group_ids(raws)
                titles = [parse_ref(r)[0] for r in raws]
                gold = json.loads(row["provenance_card"])["source_title"]
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
                lab = uniq.index(gold)
                loss = loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([lab], device=device))
                loss = loss + 0.25 * F.cross_entropy(s.unsqueeze(0), torch.tensor([int(row["label"])], device=device))
            loss = loss / len(batch)
            opt2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(emb.parameters()) + list(head.parameters()), 1.0)
            opt2.step()
            tot += float(loss)
        emb.eval()
        head.eval()
        tr_rate, both, score = eval_df(va)
        best = max(best, score)
        print(
            f"epoch {epoch+1} title={tr_rate:.4f} both={both:.4f} score~={score:.4f} best~={best:.4f} "
            f"loss={tot/max(1,len(idx)//16):.4f} t={time.time()-t0:.0f}s",
            flush=True,
        )
    print("DONE best~", best, flush=True)


if __name__ == "__main__":
    main()
