"""Technical Standard Errata Provenance Cards — offline from-scratch solver.

Compliant: no runtime internet, no pretrained weights, no TF-IDF.
Trains a rich pairwise-feature MLP with title-group listwise loss on train.csv.

Metric: 0.85*exact_title + 0.05*exact_year + 0.10*confidence_calibration
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d{2,4}")
STOP = {
    "year", "protocol", "specification", "version", "internet", "network",
    "format", "message", "standard", "requirements", "framework", "profile",
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
    "using", "based", "system", "data", "information", "control", "services",
}


@dataclass
class Config:
    seed: int = 42
    max_chars: int = 2200
    feat_dim: int = 20
    batch_size: int = 64
    epochs: int = 14
    lr: float = 2e-3
    weight_decay: float = 0.02
    val_year: int = 2011
    time_limit: int = 3200
    n_seeds: int = 5
    cand_loss_w: float = 0.15


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ref(s: str) -> tuple[str, str]:
    m = REF_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"Bad reference title: {s!r}")
    return m.group(1), m.group(2)


def tokenize(text: str) -> list[str]:
    return [t.lower().replace("\u2019", "'") for t in TOKEN_RE.findall(str(text))]


def distinctive(tokens: list[str]) -> list[str]:
    return [t for t in tokens if len(t) >= 4 and t not in STOP]


def build_query(row: pd.Series, max_chars: int) -> str:
    note = str(row.get("submitter_note") or "").strip()
    corr = str(row.get("proposed_correction") or "").strip()
    excerpt = str(row.get("original_excerpt") or "").strip()
    head = f"{note}\n{corr}\n"
    budget = max(300, max_chars - len(head))
    if len(excerpt) > budget:
        keep = budget // 2
        excerpt = excerpt[:keep] + "\n" + excerpt[-keep:]
    return head + excerpt


def label_index(row: pd.Series) -> int:
    card = json.loads(row["provenance_card"])
    for i, col in enumerate(REF_COLS):
        title, year = parse_ref(row[col])
        if title == card["source_title"] and year == str(card["source_year"]):
            return i
    raise ValueError(row["id"])


def title_group_ids(raw_cands: list[str]) -> list[int]:
    m: dict[str, int] = {}
    out = []
    for raw in raw_cands:
        t, _ = parse_ref(raw)
        if t not in m:
            m[t] = len(m)
        out.append(m[t])
    return out


def char_ngrams(s: str, n: int = 4) -> set[str]:
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def overlap_features(query: str, candidate_raw: str) -> list[float]:
    title, year = parse_ref(candidate_raw)
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
    qd = set(distinctive(q_toks))
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
        len(hits) / max(1, len(qd)),
        float(bool(hits)),
    ]


class FeatMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def title_group_scores(logits: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    bsz = logits.size(0)
    max_g = int(groups.max().item()) + 1
    scores = logits.new_full((bsz, max_g), -1e4)
    for b in range(bsz):
        g = groups[b]
        for gi in g.unique():
            mask = g == gi
            scores[b, int(gi.item())] = torch.logsumexp(logits[b, mask], dim=0)
    return scores


def decode_prediction(logits: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    bsz = logits.size(0)
    pred = torch.zeros(bsz, dtype=torch.long, device=logits.device)
    gs = title_group_scores(logits, groups)
    for b in range(bsz):
        best_g = int(gs[b].argmax().item())
        mask = groups[b] == best_g
        idx = torch.arange(logits.size(1), device=logits.device)[mask]
        pred[b] = idx[logits[b, mask].argmax()]
    return pred


def provenance_card_metric(pred_idx: np.ndarray, conf: np.ndarray, df: pd.DataFrame) -> dict:
    rows = df.reset_index(drop=True)
    title_ok = year_ok = both = 0
    ybin = []
    for i, row in rows.iterrows():
        card = json.loads(row["provenance_card"])
        t, y = parse_ref(row[REF_COLS[int(pred_idx[i])]])
        title_ok += int(t == card["source_title"])
        year_ok += int(y == str(card["source_year"]))
        ok = int(t == card["source_title"] and y == str(card["source_year"]))
        both += ok
        ybin.append(float(ok))
    n = len(rows)
    base = both / n
    c = np.asarray(conf, dtype=np.float64) / 100.0
    ybin = np.asarray(ybin, dtype=np.float64)
    brier = float(np.mean((c - ybin) ** 2))
    brier_base = float(base * (1 - base))
    cal = 0.0 if brier_base < 1e-12 else max(0.0, 1.0 - brier / brier_base)
    tr = title_ok / n
    br = both / n
    return {
        "title_rate": tr,
        "year_rate": year_ok / n,
        "both_rate": br,
        "cal": cal,
        "score": 0.85 * tr + 0.05 * br + 0.10 * cal,
        "label_acc": br,
    }


def fit_temperature(logits: np.ndarray, groups: np.ndarray, title_labels: np.ndarray) -> float:
    best_t, best_nll = 1.0, float("inf")
    z = torch.tensor(logits, dtype=torch.float32)
    g = torch.tensor(groups, dtype=torch.long)
    y = torch.tensor(title_labels, dtype=torch.long)
    for t in np.linspace(0.2, 5.0, 40):
        gs = title_group_scores(z / t, g)
        nll = float(F.cross_entropy(gs, y))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def build_matrices(df: pd.DataFrame, cfg: Config):
    rows = df.reset_index(drop=True)
    X = np.zeros((len(rows), 16, cfg.feat_dim), dtype=np.float32)
    groups = np.zeros((len(rows), 16), dtype=np.int64)
    labels = rows["label"].to_numpy() if "label" in rows.columns else np.full(len(rows), -1)
    title_labels = np.zeros(len(rows), dtype=np.int64)
    for i, row in rows.iterrows():
        q = row["query"]
        raws = [str(row[c]) for c in REF_COLS]
        groups[i] = title_group_ids(raws)
        for j, raw in enumerate(raws):
            X[i, j] = overlap_features(q, raw)
        if labels[i] >= 0:
            title_labels[i] = groups[i, int(labels[i])]
    return X, groups, labels, title_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_path")
    ap.add_argument("--mode", choices=["validate", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--time-limit", type=int, default=None)
    ap.add_argument("--val-year", type=int, default=2011)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.time_limit is not None:
        cfg.time_limit = args.time_limit
    if args.seeds is not None:
        cfg.n_seeds = args.seeds
    cfg.val_year = args.val_year

    t0 = time.time()
    deadline = t0 + cfg.time_limit
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train["label"] = train.apply(label_index, axis=1)
    train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))
    train["query"] = [build_query(r, cfg.max_chars) for _, r in train.iterrows()]
    test["query"] = [build_query(r, cfg.max_chars) for _, r in test.iterrows()]

    tr_all = train.reset_index(drop=True)
    va = train[train.src_year >= cfg.val_year].reset_index(drop=True)
    tr = train[train.src_year < cfg.val_year].reset_index(drop=True)
    if args.mode == "validate":
        infer_df = va
    else:
        infer_df = test

    print(
        f"device={device} mode={args.mode} train={len(tr)} val={len(va)} infer={len(infer_df)}",
        flush=True,
    )

    print("building features...", flush=True)
    Xtr, Gtr, Ytr, Yt_tr = build_matrices(tr, cfg)
    Xva, Gva, Yva, Yt_va = build_matrices(va, cfg)
    Xinf, Ginf, _, _ = build_matrices(infer_df, cfg)

    va_logits_all = []
    inf_logits_all = []
    best_overall = -1.0
    best_overall_m = None

    for seed in range(cfg.n_seeds):
        if time.time() > deadline - 120:
            break
        seed_everything(cfg.seed + seed * 17)
        model = FeatMLP(cfg.feat_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best_state = None
        best_sc = -1.0
        for epoch in range(cfg.epochs):
            model.train()
            idx = np.random.permutation(len(Xtr))
            for start in range(0, len(Xtr), cfg.batch_size):
                bi = idx[start : start + cfg.batch_size]
                scores = model(torch.tensor(Xtr[bi], device=device))
                g = torch.tensor(Gtr[bi], device=device)
                y = torch.tensor(Yt_tr[bi], device=device)
                loss = F.cross_entropy(title_group_scores(scores, g), y)
                loss = loss + cfg.cand_loss_w * F.cross_entropy(
                    scores, torch.tensor(Ytr[bi], device=device)
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                sv = model(torch.tensor(Xva, device=device)).cpu().numpy()
                pred = decode_prediction(torch.tensor(sv), torch.tensor(Gva)).numpy()
                gs = title_group_scores(torch.tensor(sv), torch.tensor(Gva))
                conf = F.softmax(gs, dim=-1).max(-1).values.numpy() * 100
                m = provenance_card_metric(pred, conf, va)
            if m["score"] > best_sc:
                best_sc = m["score"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 0 or (epoch + 1) % 2 == 0:
                print(f"seed{seed} ep{epoch+1} {m} best={best_sc:.4f}", flush=True)
            # early peak usually epoch 1; stop if clearly degrading
            if epoch >= 3 and m["score"] < best_sc - 0.03:
                break
            if time.time() > deadline - 90:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            va_logits_all.append(model(torch.tensor(Xva, device=device)).cpu().numpy())
            inf_logits_all.append(model(torch.tensor(Xinf, device=device)).cpu().numpy())
        if best_sc > best_overall:
            best_overall = best_sc
            best_overall_m = m
        print(f"seed{seed} done best={best_sc:.4f} t={time.time()-t0:.0f}s", flush=True)

    logits_va = np.mean(va_logits_all, axis=0)
    logits_inf = np.mean(inf_logits_all, axis=0)
    temp = fit_temperature(logits_va, Gva, Yt_va)
    pred = decode_prediction(torch.tensor(logits_va), torch.tensor(Gva)).numpy()
    gs = title_group_scores(torch.tensor(logits_va) / temp, torch.tensor(Gva))
    conf = F.softmax(gs, dim=-1).max(-1).values.numpy() * 100
    ens_m = provenance_card_metric(pred, conf, va)
    print(f"ENSEMBLE temp={temp:.3f} {ens_m} best_single={best_overall:.4f}", flush=True)

    # In full mode, optionally refit on all train with early-stopped epoch count
    if args.mode == "full" and time.time() < deadline - 180:
        print("refitting on all train...", flush=True)
        Xall, Gall, Yall, Yt_all = build_matrices(tr_all, cfg)
        refit_logits = []
        for seed in range(min(3, cfg.n_seeds)):
            seed_everything(cfg.seed + 100 + seed * 13)
            model = FeatMLP(cfg.feat_dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            # use 2 epochs — peak is early on time holdout
            for epoch in range(2):
                model.train()
                idx = np.random.permutation(len(Xall))
                for start in range(0, len(Xall), cfg.batch_size):
                    bi = idx[start : start + cfg.batch_size]
                    scores = model(torch.tensor(Xall[bi], device=device))
                    g = torch.tensor(Gall[bi], device=device)
                    y = torch.tensor(Yt_all[bi], device=device)
                    loss = F.cross_entropy(title_group_scores(scores, g), y)
                    loss = loss + cfg.cand_loss_w * F.cross_entropy(
                        scores, torch.tensor(Yall[bi], device=device)
                    )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                refit_logits.append(model(torch.tensor(Xinf, device=device)).cpu().numpy())
        logits_inf = np.mean(refit_logits, axis=0)

    pred = decode_prediction(torch.tensor(logits_inf), torch.tensor(Ginf)).numpy()
    gs = title_group_scores(torch.tensor(logits_inf) / temp, torch.tensor(Ginf))
    conf = np.clip(F.softmax(gs, dim=-1).max(-1).values.numpy() * 100, 0, 100)

    rows_out = []
    infer_rows = infer_df.reset_index(drop=True)
    for i, row in infer_rows.iterrows():
        t, y = parse_ref(row[REF_COLS[int(pred[i])]])
        rows_out.append(
            {
                "id": row["id"],
                "provenance_card": json.dumps(
                    {"source_title": t, "source_year": y},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "confidence": float(conf[i]),
            }
        )
    sub = pd.DataFrame(rows_out)
    sub.to_csv(out_path, index=False)
    print(
        f"wrote {out_path} n={len(sub)} val_score={ens_m['score']:.4f} t={time.time()-t0:.0f}s",
        flush=True,
    )
    if args.mode == "validate":
        print(f"VAL_METRIC {ens_m}", flush=True)


if __name__ == "__main__":
    main()
