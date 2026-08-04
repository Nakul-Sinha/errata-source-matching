"""Offline-only CE (local HF cache, no download) + lexical feature blend.

Uses transformers local_files_only=True. Research probe for ceiling.
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
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

DATA = Path(__file__).resolve().parents[1] / "dataset" / "public"
REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d{2,4}")
STOP = {
    "year", "protocol", "specification", "version", "internet", "network",
    "format", "message", "standard", "requirements", "framework", "profile",
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
}
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def parse(s):
    m = REF_RE.match(str(s).strip())
    return m.group(1), m.group(2)


def tokenize(text):
    return [t.lower().replace("\u2019", "'") for t in TOKEN_RE.findall(str(text))]


def build_query(row, max_chars=1600):
    note = str(row.get("submitter_note") or "")
    corr = str(row.get("proposed_correction") or "")
    excerpt = str(row.get("original_excerpt") or "")
    head = f"{note}\n{corr}\n"
    budget = max(300, max_chars - len(head))
    if len(excerpt) > budget:
        excerpt = excerpt[: budget // 2] + "\n" + excerpt[-budget // 2 :]
    return head + excerpt


def feats(query, raw):
    title, year = parse(raw)
    ql = query.lower()
    cd = [t for t in tokenize(title) if len(t) >= 4 and t not in STOP]
    hits = [t for t in cd if t in ql]
    acr = {a.lower() for a in re.findall(r"\b[A-Z]{2,10}\b", title)}
    qs = set(tokenize(query))
    phrase = float(any(" ".join(cd[i : i + 2]) in ql for i in range(max(0, len(cd) - 1))))
    return [
        len(hits) / max(1, len(cd)),
        sum(len(t) for t in hits) / max(1.0, sum(len(t) for t in cd)),
        len(acr & qs) / max(1, len(acr)),
        float(year in query),
        phrase,
        math.log1p(len(hits)),
        float(len(hits) >= 2),
        float(title.lower() in ql),
    ]


class CE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL, local_files_only=True)
        d = self.encoder.config.hidden_size
        self.head = nn.Linear(d, 1)
        self.feat = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 1))
        self.gate = nn.Parameter(torch.tensor(-1.0))  # start feat-heavy

    def forward(self, input_ids, attention_mask, feat):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        ce = self.head(cls).squeeze(-1)
        ff = self.feat(feat).squeeze(-1)
        g = torch.sigmoid(self.gate)
        return (1 - g) * ff * 5 + g * ce, g


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

    model = CE().to(device)
    # freeze bottom 4 layers
    for name, p in model.encoder.named_parameters():
        if any(f"layer.{i}." in name for i in range(4)):
            p.requires_grad = False
        if "embeddings" in name:
            p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)

    def encode_batch(queries, cands):
        pairs_q, pairs_c = [], []
        for q, cs in zip(queries, cands):
            for c in cs:
                pairs_q.append(q)
                pairs_c.append(c)
        enc = tok(
            pairs_q,
            pairs_c,
            padding=True,
            truncation=True,
            max_length=288,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    best = 0.0
    best_m = None
    for epoch in range(6):
        model.train()
        idx = list(range(len(tr)))
        random.shuffle(idx)
        tot = 0.0
        steps = 0
        for start in range(0, len(idx), 2):
            batch = idx[start : start + 2]
            queries = [tr.loc[i, "query"] for i in batch]
            cands, feat, groups, titles_gold, labs = [], [], [], [], []
            for i in batch:
                row = tr.loc[i]
                raws = [str(row[c]) for c in REF_COLS]
                cands.append([parse(r)[0] + " (" + parse(r)[1] + ")" for r in raws])
                feat.append([feats(row["query"], r) for r in raws])
                groups.append(title_groups(raws))
                titles_gold.append(json.loads(row["provenance_card"])["source_title"])
                labs.append(int(row["label"]))
            enc = encode_batch(queries, cands)
            f = torch.tensor(feat, dtype=torch.float32, device=device).view(-1, 8)
            scores, gate = model(enc["input_ids"], enc["attention_mask"], f)
            scores = scores.view(len(batch), 16)
            loss = 0.0
            for bi in range(len(batch)):
                s = scores[bi]
                gmap = groups[bi]
                uniq = sorted(set(gmap))
                gs = s.new_full((len(uniq),), -1e4)
                gt = torch.tensor(gmap, device=device)
                for u, ug in enumerate(uniq):
                    gs[u] = torch.logsumexp(s[gt == ug], 0)
                # map gold title to group
                raws = [str(tr.loc[batch[bi], c]) for c in REF_COLS]
                titles = [parse(r)[0] for r in raws]
                lab = uniq.index(
                    gmap[titles.index(titles_gold[bi])]
                    if False
                    else title_groups(raws)[titles.index(titles_gold[bi])]
                )
                # simpler:
                gold_g = title_groups(raws)[titles.index(titles_gold[bi])]
                lab = uniq.index(gold_g)
                loss = loss + F.cross_entropy(gs.unsqueeze(0), torch.tensor([lab], device=device))
                loss = loss + 0.2 * F.cross_entropy(s.unsqueeze(0), torch.tensor([labs[bi]], device=device))
            loss = loss / len(batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            steps += 1
        # eval
        model.eval()
        preds, confs = [], []
        gates = []
        with torch.no_grad():
            for start in range(0, len(va), 4):
                batch = list(range(start, min(len(va), start + 4)))
                queries = [va.loc[i, "query"] for i in batch]
                cands, feat, groups = [], [], []
                for i in batch:
                    row = va.loc[i]
                    raws = [str(row[c]) for c in REF_COLS]
                    cands.append([parse(r)[0] + " (" + parse(r)[1] + ")" for r in raws])
                    feat.append([feats(row["query"], r) for r in raws])
                    groups.append(title_groups(raws))
                enc = encode_batch(queries, cands)
                f = torch.tensor(feat, dtype=torch.float32, device=device).view(-1, 8)
                scores, gate = model(enc["input_ids"], enc["attention_mask"], f)
                scores = scores.view(len(batch), 16)
                gates.append(float(gate))
                for bi in range(len(batch)):
                    s = scores[bi]
                    gmap = groups[bi]
                    uniq = sorted(set(gmap))
                    gs = s.new_full((len(uniq),), -1e4)
                    gt = torch.tensor(gmap, device=device)
                    for u, ug in enumerate(uniq):
                        gs[u] = torch.logsumexp(s[gt == ug], 0)
                    p = F.softmax(gs, 0)
                    best_g = uniq[int(p.argmax())]
                    mask = [j for j, g in enumerate(gmap) if g == best_g]
                    preds.append(mask[int(s[mask].argmax())])
                    confs.append(float(p.max() * 100))
        m = metric(preds, confs, va)
        best = max(best, m["score"])
        if m["score"] >= best:
            best_m = m
        print(
            f"epoch {epoch+1} {m} best={best:.4f} gate={np.mean(gates):.3f} "
            f"loss={tot/max(1,steps):.4f} t={time.time()-t0:.0f}s",
            flush=True,
        )
    print("DONE", best_m, "best_score", best, flush=True)


if __name__ == "__main__":
    main()
