"""Technical Standard Errata Provenance Cards — offline from-scratch solver.

Compliant constraints:
- No runtime internet / no downloaded pretrained weights
- No TF-IDF
- Real training inside this script on train.csv only

Model:
- Stateless pairwise overlap features (set/char overlaps; no corpus IDF)
- From-scratch character CNN bi-encoder for compositional title matching
- Title-group listwise loss (title is 85% of the official metric) plus a lighter
  exact-candidate term for year disambiguation
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d{2,4}|[A-Z]{2,}")
STOP = {
    "year",
    "protocol",
    "specification",
    "version",
    "internet",
    "network",
    "format",
    "message",
    "standard",
    "requirements",
    "framework",
    "profile",
    "the",
    "and",
    "for",
    "with",
    "from",
}


@dataclass
class Config:
    seed: int = 42
    max_chars: int = 1800
    query_chars: int = 768
    cand_chars: int = 160
    char_dim: int = 64
    cnn_channels: int = 128
    d_model: int = 160
    dropout: float = 0.12
    feat_dim: int = 16
    batch_size: int = 24
    epochs: int = 12
    lr: float = 2.5e-3
    weight_decay: float = 0.01
    warmup_fraction: float = 0.06
    grad_clip: float = 1.0
    val_year: int = 2011
    time_limit: int = 3000


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


def build_query(row: pd.Series, max_chars: int) -> str:
    note = str(row.get("submitter_note") or "").strip()
    corr = str(row.get("proposed_correction") or "").strip()
    excerpt = str(row.get("original_excerpt") or "").strip()
    head = f"{note}\n{corr}\n"
    budget = max(200, max_chars - len(head))
    if len(excerpt) > budget:
        keep = budget // 2
        excerpt = excerpt[:keep] + " " + excerpt[-keep:]
    return head + excerpt


def format_candidate(raw: str) -> str:
    title, year = parse_ref(raw)
    return f"{title} year {year}"


def label_index(row: pd.Series) -> int:
    card = json.loads(row["provenance_card"])
    for i, col in enumerate(REF_COLS):
        title, year = parse_ref(row[col])
        if title == card["source_title"] and year == str(card["source_year"]):
            return i
    raise ValueError(row["id"])


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def overlap_features(query: str, candidate: str) -> list[float]:
    """Stateless pairwise overlaps only — no corpus statistics / TF-IDF."""
    q = tokenize(query)
    c = tokenize(candidate)
    q_set, c_set = set(q), set(c)
    content_q = {t for t in q_set if len(t) >= 4 and t not in STOP}
    content_c = {t for t in c_set if len(t) >= 4 and t not in STOP}
    distinctive_c = {t for t in content_c if len(t) >= 5}
    raw_acr = {
        a.lower()
        for a in re.findall(r"\b[A-Z]{2,10}\b", candidate) + re.findall(r"\b[A-Z]{2,10}\b", query)
    }
    ql = re.sub(r"\s+", " ", query.lower())
    cl = re.sub(r"\s+", " ", candidate.lower())

    inter = content_q & content_c
    dist_inter = content_q & distinctive_c
    # substring hits: title token appears inside query text
    sub_hits = [t for t in distinctive_c if t in ql]
    acr_inter = raw_acr & q_set
    qg = {ql[i : i + 4] for i in range(max(0, len(ql) - 3))}
    cg = {cl[i : i + 4] for i in range(max(0, len(cl) - 3))}
    q5 = {ql[i : i + 5] for i in range(max(0, len(ql) - 4))}
    c5 = {cl[i : i + 5] for i in range(max(0, len(cl) - 4))}
    qb, cb = _ngrams(q, 2), _ngrams(c, 2)
    year_toks = [t for t in c if re.fullmatch(r"\d{4}", t)]
    year_in_q = float(any(y in q_set for y in year_toks))
    len_weighted = sum(len(t) for t in dist_inter) / max(1.0, sum(len(t) for t in distinctive_c))
    return [
        len(inter) / max(1, len(content_c)),
        len(inter) / max(1, len(content_q)),
        len(dist_inter) / max(1, len(distinctive_c)),
        len(sub_hits) / max(1, len(distinctive_c)),
        len_weighted,
        len(qg & cg) / max(1, len(cg)),
        len(q5 & c5) / max(1, len(c5)),
        len(qb & cb) / max(1, len(cb)),
        len(acr_inter) / max(1, len(raw_acr) + 1),
        year_in_q,
        float(bool(dist_inter)),
        float(len(dist_inter) >= 2),
        float(len(sub_hits) >= 2),
        float(len(acr_inter) >= 1),
        math.log1p(sum(len(t) for t in sub_hits)) / 5.0,
        math.log1p(len(c)) / 5.0,
    ]


def title_group_ids(raw_cands: list[str]) -> list[int]:
    title_to_g: dict[str, int] = {}
    groups: list[int] = []
    for raw in raw_cands:
        title, _ = parse_ref(raw)
        if title not in title_to_g:
            title_to_g[title] = len(title_to_g)
        groups.append(title_to_g[title])
    return groups


def encode_chars(text: str, limit: int) -> list[int]:
    # 0 pad, 1 unk, 2..97 printable-ish bytes
    ids = []
    for ch in str(text)[:limit]:
        o = ord(ch)
        if 32 <= o <= 126:
            ids.append(o - 30)  # 2..96
        else:
            ids.append(1)
    return ids or [1]


class RankDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: Config, labelled: bool):
        self.config = config
        self.ids = df["id"].tolist()
        self.labels = df["label"].tolist() if labelled else [-1] * len(df)
        self.queries, self.cands, self.feats = [], [], []
        self.raw_cands, self.groups, self.title_labels = [], [], []
        for _, row in df.iterrows():
            q = build_query(row, config.max_chars)
            raws = [str(row[c]) for c in REF_COLS]
            cands = [format_candidate(r) for r in raws]
            groups = title_group_ids(raws)
            self.queries.append(q)
            self.cands.append(cands)
            self.raw_cands.append(raws)
            self.feats.append([overlap_features(q, c) for c in cands])
            self.groups.append(groups)
            self.title_labels.append(groups[int(row["label"])] if labelled else -1)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> int:
        return idx


def collate_factory(dataset: RankDataset):
    cfg = dataset.config

    def collate(indices):
        q_ids, c_ids, feats, labels, title_labels, groups, row_ids = [], [], [], [], [], [], []
        for idx in indices:
            i = int(idx)
            q_ids.append(torch.tensor(encode_chars(dataset.queries[i], cfg.query_chars), dtype=torch.long))
            for c in dataset.cands[i]:
                c_ids.append(torch.tensor(encode_chars(c, cfg.cand_chars), dtype=torch.long))
            feats.append(dataset.feats[i])
            labels.append(dataset.labels[i])
            title_labels.append(dataset.title_labels[i])
            groups.append(dataset.groups[i])
            row_ids.append(dataset.ids[i])

        def pad(seqs, limit):
            out = torch.zeros((len(seqs), limit), dtype=torch.long)
            mask = torch.zeros((len(seqs), limit), dtype=torch.bool)
            for i, s in enumerate(seqs):
                n = min(limit, s.size(0))
                out[i, :n] = s[:n]
                mask[i, :n] = True
            return out, mask

        q_out, q_m = pad(q_ids, cfg.query_chars)
        c_out, c_m = pad(c_ids, cfg.cand_chars)
        bsz = len(indices)
        return {
            "q_ids": q_out,
            "q_mask": q_m,
            "c_ids": c_out.view(bsz, 16, -1),
            "c_mask": c_m.view(bsz, 16, -1),
            "feats": torch.tensor(feats, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "title_labels": torch.tensor(title_labels, dtype=torch.long),
            "groups": torch.tensor(groups, dtype=torch.long),
            "row_ids": row_ids,
        }

    return collate


class CharEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.emb = nn.Embedding(98, config.char_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(config.char_dim, config.cnn_channels, k, padding=k // 2)
                for k in (3, 5, 7)
            ]
        )
        self.proj = nn.Linear(config.cnn_channels * 3, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ids (B, L)
        x = self.emb(ids).transpose(1, 2)  # (B, C, L)
        feats = []
        for conv in self.convs:
            h = F.gelu(conv(x))
            h = h.masked_fill(~mask.unsqueeze(1), -1e4)
            feats.append(h.max(dim=-1).values)
        h = torch.cat(feats, dim=-1)
        h = self.dropout(self.proj(h))
        return F.normalize(h, dim=-1)


class Ranker(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = CharEncoder(config)
        self.feat_mlp = nn.Sequential(
            nn.Linear(config.feat_dim, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(config.d_model * 3 + config.feat_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.neural_gate = nn.Parameter(torch.tensor(-2.0))
        # Initialize feat MLP near a strong linear overlap scorer.
        with torch.no_grad():
            self.feat_mlp[0].weight.zero_()
            self.feat_mlp[0].bias.zero_()
            self.feat_mlp[3].weight.zero_()
            self.feat_mlp[3].bias.zero_()
            # Map selected feature channels through identity-ish first layer.
            for src, w in (
                (0, 2.0),
                (2, 4.0),
                (3, 5.0),
                (4, 5.0),
                (5, 2.0),
                (6, 2.5),
                (8, 2.0),
                (10, 1.5),
                (11, 2.0),
                (12, 2.5),
                (14, 2.0),
            ):
                self.feat_mlp[0].weight[0, src] = w
            self.feat_mlp[3].weight[0, 0] = 1.0

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q = self.encoder(batch["q_ids"], batch["q_mask"])
        bsz = q.size(0)
        c = self.encoder(
            batch["c_ids"].view(bsz * 16, -1), batch["c_mask"].view(bsz * 16, -1)
        ).view(bsz, 16, -1)
        q_exp = q.unsqueeze(1).expand(-1, 16, -1)
        cos = (q_exp * c).sum(-1) * self.scale
        feats = batch["feats"]
        feat_scores = self.feat_mlp(feats).squeeze(-1)
        pair = torch.cat([q_exp, c, q_exp * c, feats], dim=-1)
        neural = self.pair_mlp(pair).squeeze(-1) + cos
        gate = torch.sigmoid(self.neural_gate)
        return feat_scores + gate * neural


def title_group_scores(cand_logits: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    bsz = cand_logits.size(0)
    max_g = int(groups.max().item()) + 1
    scores = cand_logits.new_full((bsz, max_g), -1e4)
    for b in range(bsz):
        g = groups[b]
        for gi in g.unique():
            mask = g == gi
            scores[b, int(gi.item())] = torch.logsumexp(cand_logits[b, mask], dim=0)
    return scores


def decode_prediction(cand_logits: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    bsz = cand_logits.size(0)
    pred = torch.zeros(bsz, dtype=torch.long, device=cand_logits.device)
    group_scores = title_group_scores(cand_logits, groups)
    for b in range(bsz):
        best_g = int(group_scores[b].argmax().item())
        mask = groups[b] == best_g
        idx = torch.arange(16, device=cand_logits.device)[mask]
        pred[b] = idx[cand_logits[b, mask].argmax()]
    return pred


def ranking_loss(
    logits: torch.Tensor, labels: torch.Tensor, title_labels: torch.Tensor, groups: torch.Tensor
) -> torch.Tensor:
    group_scores = title_group_scores(logits, groups)
    return F.cross_entropy(group_scores, title_labels) + 0.25 * F.cross_entropy(logits, labels)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    y = torch.tensor(labels, dtype=torch.long)
    z = torch.tensor(logits, dtype=torch.float32)
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.2, 5.0, 50):
        nll = float(F.cross_entropy(z / t, y))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def metric_from_preds(pred: np.ndarray, conf: np.ndarray, df: pd.DataFrame) -> dict:
    title_ok = np.zeros(len(df), dtype=np.int32)
    year_ok = np.zeros(len(df), dtype=np.int32)
    both_ok = np.zeros(len(df), dtype=np.int32)
    for i, (_, row) in enumerate(df.iterrows()):
        card = json.loads(row["provenance_card"])
        title, year = parse_ref(row[REF_COLS[int(pred[i])]])
        title_ok[i] = int(title == card["source_title"])
        year_ok[i] = int(year == str(card["source_year"]))
        both_ok[i] = int(title_ok[i] and year_ok[i])
    y = both_ok.astype(np.float64)
    p = np.clip(conf / 100.0, 0.0, 1.0)
    brier = float(((p - y) ** 2).mean())
    base = float(y.mean())
    ref = base * (1 - base) if 0 < base < 1 else (1 / 16) * (15 / 16)
    cal = float(np.clip(1 - brier / ref, 0, 1))
    title_rate = float(title_ok.mean())
    year_rate = float(year_ok.mean())
    return {
        "title_rate": title_rate,
        "year_rate": year_rate,
        "both_rate": float(y.mean()),
        "cal": cal,
        "score": 0.85 * title_rate + 0.05 * year_rate + 0.10 * cal,
        "label_acc": float((pred == df["label"].to_numpy()).mean()) if "label" in df else float(y.mean()),
    }


@torch.no_grad()
def predict(model: Ranker, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all, pred_all, conf_all, ids_all = [], [], [], []
    for batch in loader:
        batch_t = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
            if k != "row_ids"
        }
        logits = model(batch_t)
        groups = batch_t["groups"]
        pred = decode_prediction(logits, groups)
        g_scores = title_group_scores(logits, groups)
        g_prob = F.softmax(g_scores, dim=-1)
        conf = [
            float(g_prob[b, int(groups[b, pred[b]].item())].item() * 100.0)
            for b in range(logits.size(0))
        ]
        logits_all.append(logits.float().cpu().numpy())
        pred_all.append(pred.cpu().numpy())
        conf_all.append(np.asarray(conf, dtype=np.float64))
        ids_all.extend(batch["row_ids"])
    return (
        np.concatenate(logits_all, 0),
        np.concatenate(pred_all, 0),
        np.concatenate(conf_all, 0),
        ids_all,
    )


def resolve_paths(public: str | None, out: str | None) -> tuple[Path, Path]:
    if public and out:
        return Path(public), Path(out)
    public_p = Path("./dataset/public")
    out_p = Path("./working/submission.csv")
    return public_p, out_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir", nargs="?", default=None)
    ap.add_argument("submission_out", nargs="?", default=None)
    ap.add_argument("--mode", choices=["full", "validate"], default="full")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--time-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config(seed=args.seed)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.time_limit is not None:
        cfg.time_limit = args.time_limit

    t0 = time.time()
    seed_everything(cfg.seed)
    data_dir, sub_path = resolve_paths(args.public_dir, args.submission_out)
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} mode={args.mode} data={data_dir}", flush=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train["label"] = train.apply(label_index, axis=1)
    train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))

    if args.mode == "validate":
        tr_df = train[train["src_year"] < cfg.val_year].reset_index(drop=True)
        va_df = train[train["src_year"] >= cfg.val_year].reset_index(drop=True)
    else:
        va_df = train[train["src_year"] >= cfg.val_year].reset_index(drop=True)
        va_ids = set(va_df["id"])
        tr_df = train[~train["id"].isin(va_ids)].reset_index(drop=True)
        if len(va_df) < 80:
            va_df = train.sample(n=min(500, len(train)), random_state=cfg.seed).reset_index(drop=True)
            va_ids = set(va_df["id"])
            tr_df = train[~train["id"].isin(va_ids)].reset_index(drop=True)

    print(f"split train={len(tr_df)} val={len(va_df)} test={len(test)}", flush=True)

    tr_ds = RankDataset(tr_df, cfg, labelled=True)
    va_ds = RankDataset(va_df, cfg, labelled=True)
    tr_loader = DataLoader(
        tr_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_factory(tr_ds)
    )
    va_loader = DataLoader(
        va_ds, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=collate_factory(va_ds)
    )

    model = Ranker(cfg).to(device)
    # Freeze char encoder briefly; learn feature/pair heads first.
    freeze_epochs = min(4, max(2, cfg.epochs // 3))
    for p in model.encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, len(tr_loader) * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_fraction)
    global_step = 0
    encoder_unfrozen = False

    def lr_at(step: int) -> float:
        if step < warmup:
            return cfg.lr * float(step + 1) / float(max(1, warmup))
        prog = (step - warmup) / float(max(1, total_steps - warmup))
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    best = {"score": -1.0, "temperature": 1.0}
    best_state = None
    stop = False

    # Baseline before training
    with torch.no_grad():
        _, pred0, conf0, _ = predict(model, va_loader, device)
        print("init_val", metric_from_preds(pred0, conf0, va_df), flush=True)

    for epoch in range(cfg.epochs):
        if stop:
            break
        if (not encoder_unfrozen) and epoch >= freeze_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.4, weight_decay=cfg.weight_decay)
            encoder_unfrozen = True
            print(f"unfroze char encoder at epoch {epoch+1}", flush=True)

        model.train()
        running = 0.0
        for step, batch in enumerate(tr_loader):
            if time.time() - t0 > cfg.time_limit:
                print("time limit during training", flush=True)
                stop = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(global_step) * (0.5 if encoder_unfrozen else 1.0)
            batch_t = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
                if k != "row_ids"
            }
            logits = model(batch_t)
            loss = ranking_loss(
                logits, batch_t["labels"], batch_t["title_labels"], batch_t["groups"]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.item())
            global_step += 1
            if (step + 1) % 40 == 0:
                print(
                    f"epoch {epoch+1} step {step+1}/{len(tr_loader)} "
                    f"loss={running/(step+1):.4f} gate={float(torch.sigmoid(model.neural_gate)):.3f} "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )

        va_logits, _, _, _ = predict(model, va_loader, device)
        temp = fit_temperature(va_logits, va_df["label"].to_numpy())
        with torch.no_grad():
            z = torch.tensor(va_logits, device=device) / temp
            groups = torch.tensor(va_ds.groups, device=device)
            pred_t = decode_prediction(z, groups).cpu().numpy()
            g_scores = title_group_scores(z, groups)
            g_prob = F.softmax(g_scores, dim=-1).cpu().numpy()
            conf = np.array(
                [g_prob[i, va_ds.groups[i][pred_t[i]]] * 100.0 for i in range(len(pred_t))]
            )
        metrics = metric_from_preds(pred_t, conf, va_df)
        metrics["temperature"] = temp
        metrics["gate"] = float(torch.sigmoid(model.neural_gate).item())
        print(f"epoch {epoch+1} val {metrics}", flush=True)
        if metrics["score"] > best["score"]:
            best = {**metrics, "epoch": epoch + 1}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.mode == "full" and time.time() - t0 < cfg.time_limit - 120:
        print("refine on all train", flush=True)
        all_ds = RankDataset(train, cfg, labelled=True)
        all_loader = DataLoader(
            all_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_factory(all_ds)
        )
        for p in model.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.2, weight_decay=cfg.weight_decay)
        model.train()
        for step, batch in enumerate(all_loader):
            if time.time() - t0 > cfg.time_limit - 60:
                break
            batch_t = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
                if k != "row_ids"
            }
            loss = ranking_loss(
                model(batch_t), batch_t["labels"], batch_t["title_labels"], batch_t["groups"]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        va_logits, va_pred, va_conf, _ = predict(model, va_loader, device)
        print("post-refine val", metric_from_preds(va_pred, va_conf, va_df), flush=True)
        best["temperature"] = fit_temperature(va_logits, va_df["label"].to_numpy())

    temp = float(best.get("temperature", 1.0))
    te_ds = RankDataset(test, cfg, labelled=False)
    te_loader = DataLoader(
        te_ds, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=collate_factory(te_ds)
    )
    te_logits, _, _, te_ids = predict(model, te_loader, device)
    with torch.no_grad():
        z = torch.tensor(te_logits) / temp
        groups = torch.tensor(te_ds.groups)
        pred = decode_prediction(z, groups).numpy()
        g_scores = title_group_scores(z, groups)
        g_prob = F.softmax(g_scores, dim=-1).numpy()
        conf = np.clip(
            np.array([g_prob[i, te_ds.groups[i][pred[i]]] * 100.0 for i in range(len(pred))]),
            1.0,
            99.0,
        )

    id_to_i = {rid: i for i, rid in enumerate(te_ids)}
    rows = []
    for _, row in test.iterrows():
        i = id_to_i[row["id"]]
        title, year = parse_ref(row[REF_COLS[int(pred[i])]])
        rows.append(
            {
                "id": row["id"],
                "provenance_card": json.dumps(
                    {"source_title": title, "source_year": year}, ensure_ascii=False
                ),
                "confidence": float(conf[i]),
            }
        )
    sub = pd.DataFrame(rows)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    assert list(sub.columns) == list(sample.columns)
    assert set(sub["id"]) == set(sample["id"])
    assert sub["id"].is_unique
    for card in sub["provenance_card"]:
        obj = json.loads(card)
        assert set(obj) >= {"source_title", "source_year"}
    assert np.isfinite(sub["confidence"]).all()
    assert ((sub["confidence"] >= 0) & (sub["confidence"] <= 100)).all()
    sub.to_csv(sub_path, index=False)
    print(f"wrote {sub_path} rows={len(sub)} best_val={best} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
