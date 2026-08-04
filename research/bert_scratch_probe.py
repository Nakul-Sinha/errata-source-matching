"""From-scratch BERT (random BertConfig, no pretrained weights) + feature anchor.

Pattern adapted from Meridian Ashes: HF architecture classes only, weights trained
on train.csv. Feature MLP provides OOD-stable prior; BERT residual is gated.
"""
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
from tokenizers import Tokenizer
from tokenizers import models as tok_models
from tokenizers import pre_tokenizers as tok_pre
from tokenizers import trainers as tok_trainers
from transformers import BertConfig, BertForMaskedLM, BertModel

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


def tokenize_words(text):
    return [t.lower().replace("\u2019", "'") for t in TOKEN_RE.findall(str(text))]


def distinctive(tokens):
    return [t for t in tokens if len(t) >= 4 and t not in STOP]


def build_query(row, max_chars=1600):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(250, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


def feats(query, raw):
    title, year = parse(raw)
    cd = distinctive(tokenize_words(title))
    ql = query.lower()
    hits = [t for t in cd if t in ql]
    acr = {a.lower() for a in re.findall(r"\b[A-Z]{2,12}\b", title)}
    qs = set(tokenize_words(query))
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
        len(hits) / max(1, len(set(distinctive(tokenize_words(query))))),
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


def train_tokenizer(texts, vocab_size=8000):
    tok = Tokenizer(tok_models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = tok_pre.Whitespace()
    trainer = tok_trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode_pair(tok, q, c, max_len=256):
    q_ids = tok.encode(q).ids[:180]
    c_ids = tok.encode(c).ids[:60]
    ids = [tok.token_to_id("[CLS]")] + q_ids + [tok.token_to_id("[SEP]")] + c_ids + [tok.token_to_id("[SEP]")]
    ids = ids[:max_len]
    attn = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(tok.token_to_id("[PAD]"))
        attn.append(0)
    return ids, attn


class FeatMLP(nn.Module):
    def __init__(self, d=20):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RankHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.head = nn.Linear(hidden, 1)
        self.gate = nn.Parameter(torch.tensor(-2.0))  # start feat-heavy (~0.12)

    def forward(self, cls, feat_scores):
        ce = self.head(cls).squeeze(-1)
        g = torch.sigmoid(self.gate)
        return (1 - g) * feat_scores + g * ce, g


def decode(scores, groups):
    uniq = sorted(set(groups))
    gt = torch.tensor(groups, device=scores.device)
    gs = torch.stack([torch.logsumexp(scores[gt == u], 0) for u in uniq])
    p = F.softmax(gs, 0)
    bg = uniq[int(p.argmax())]
    mask = [j for j, g in enumerate(groups) if g == bg]
    return mask[int(scores[mask].argmax())], float(p.max() * 100)


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

    # features first
    def build_X(df):
        X = np.zeros((len(df), 16, 20), dtype=np.float32)
        G = np.zeros((len(df), 16), dtype=np.int64)
        for i, row in df.iterrows():
            raws = [str(row[c]) for c in REF_COLS]
            G[i] = title_groups(raws)
            for j, raw in enumerate(raws):
                X[i, j] = feats(row["query"], raw)
        return X, G

    Xtr, Gtr = build_X(tr)
    Xva, Gva = build_X(va)
    feat = FeatMLP().to(device)
    opt = torch.optim.AdamW(feat.parameters(), lr=2e-3, weight_decay=0.02)
    best_state, best_sc = None, -1
    for ep in range(6):
        feat.train()
        idx = np.random.permutation(len(tr))
        for start in range(0, len(tr), 64):
            bi = idx[start : start + 64]
            s = feat(torch.tensor(Xtr[bi], device=device))
            loss = 0.0
            for b, i in enumerate(bi):
                raws = [str(tr.loc[i, c]) for c in REF_COLS]
                titles = [parse(r)[0] for r in raws]
                gold = json.loads(tr.loc[i, "provenance_card"])["source_title"]
                uniq, gmap = [], []
                for t in titles:
                    if t not in uniq:
                        uniq.append(t)
                    gmap.append(uniq.index(t))
                gt = torch.tensor(gmap, device=device)
                gs = s[b].new_full((len(uniq),), -1e4)
                for u in range(len(uniq)):
                    gs[u] = torch.logsumexp(s[b][gt == u], 0)
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
        print(f"feat ep{ep+1} {m} best={best_sc:.4f}", flush=True)
    feat.load_state_dict(best_state)
    feat.eval()
    with torch.no_grad():
        feat_tr = feat(torch.tensor(Xtr, device=device)).detach()
        feat_va = feat(torch.tensor(Xva, device=device)).detach()

    # tokenizer + MLM
    texts = list(tr["query"])
    for _, row in tr.iterrows():
        for c in REF_COLS:
            t, y = parse(row[c])
            texts.append(f"{t} ({y})")
    print("training wordpiece...", flush=True)
    tok = train_tokenizer(texts, 8000)
    pad_id = tok.token_to_id("[PAD]")
    mask_id = tok.token_to_id("[MASK]")
    cls_id = tok.token_to_id("[CLS]")
    sep_id = tok.token_to_id("[SEP]")
    cfg = BertConfig(
        vocab_size=tok.get_vocab_size(),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=768,
        max_position_embeddings=288,
        pad_token_id=pad_id,
        type_vocab_size=2,
    )
    mlm = BertForMaskedLM(cfg).to(device)
    opt = torch.optim.AdamW(mlm.parameters(), lr=5e-4, weight_decay=0.01)
    print("mlm...", flush=True)
    for ep in range(5):
        random.shuffle(texts)
        tot = 0.0
        steps = 0
        for start in range(0, min(len(texts), 20000), 32):
            batch = texts[start : start + 32]
            ids_list = [tok.encode(t).ids[:240] for t in batch]
            L = max(len(x) for x in ids_list)
            input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros(len(batch), L, dtype=torch.long, device=device)
            for i, ids in enumerate(ids_list):
                input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
                attn[i, : len(ids)] = 1
            labels = input_ids.clone()
            prob = (torch.rand_like(input_ids, dtype=torch.float) < 0.15) & (attn.bool())
            prob[:, 0] = False
            input_ids = input_ids.clone()
            input_ids[prob] = mask_id
            labels = labels.masked_fill(~prob, -100)
            loss = mlm(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mlm.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            steps += 1
        print(f"mlm ep{ep+1} loss={tot/max(1,steps):.4f} t={time.time()-t0:.0f}s", flush=True)

    encoder = mlm.bert
    head = RankHead(cfg.hidden_size).to(device)
    opt2 = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": 1e-4},
            {"params": head.parameters(), "lr": 1e-3},
        ],
        weight_decay=0.01,
    )

    best = best_sc
    for epoch in range(6):
        encoder.train()
        head.train()
        idx = list(range(len(tr)))
        random.shuffle(idx)
        tot = 0.0
        steps = 0
        for start in range(0, len(idx), 2):
            batch = idx[start : start + 2]
            all_ids, all_attn = [], []
            f_scores = []
            groups, golds = [], []
            for i in batch:
                row = tr.loc[i]
                raws = [str(row[c]) for c in REF_COLS]
                for raw in raws:
                    t, y = parse(raw)
                    ids, attn = encode_pair(tok, row["query"], f"{t} ({y})")
                    all_ids.append(ids)
                    all_attn.append(attn)
                f_scores.append(feat_tr[i])
                groups.append(title_groups(raws))
                golds.append(json.loads(row["provenance_card"])["source_title"])
            input_ids = torch.tensor(all_ids, device=device)
            attn = torch.tensor(all_attn, device=device)
            cls = encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, 0]
            cls = cls.view(len(batch), 16, -1)
            fs = torch.stack(f_scores)
            scores, gate = head(cls, fs)
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
                # distill toward feat
                loss = loss + 0.2 * F.mse_loss(s, fs[bi])
            loss = loss / len(batch)
            opt2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
            opt2.step()
            tot += float(loss)
            steps += 1
        # eval
        encoder.eval()
        head.eval()
        preds, confs = [], []
        with torch.no_grad():
            for start in range(0, len(va), 2):
                batch = list(range(start, min(len(va), start + 2)))
                all_ids, all_attn = [], []
                for i in batch:
                    row = va.loc[i]
                    for c in REF_COLS:
                        t, y = parse(row[c])
                        ids, attn = encode_pair(tok, row["query"], f"{t} ({y})")
                        all_ids.append(ids)
                        all_attn.append(attn)
                input_ids = torch.tensor(all_ids, device=device)
                attn = torch.tensor(all_attn, device=device)
                cls = encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, 0]
                cls = cls.view(len(batch), 16, -1)
                fs = feat_va[batch]
                scores, gate = head(cls, fs)
                for bi in range(len(batch)):
                    p, c = decode(scores[bi], Gva[batch[bi]].tolist())
                    preds.append(p)
                    confs.append(c)
        m = metric(preds, confs, va)
        best = max(best, m["score"])
        print(
            f"rank ep{epoch+1} {m} best={best:.4f} gate={float(torch.sigmoid(head.gate)):.3f} "
            f"loss={tot/max(1,steps):.4f} t={time.time()-t0:.0f}s",
            flush=True,
        )
    print("DONE best", best, flush=True)


if __name__ == "__main__":
    main()
