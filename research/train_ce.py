"""Research cross-encoder for errata provenance ranking (local / Kaggle)."""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

REF_COLS = [f"reference_title_{i:02d}" for i in range(1, 17)]
REF_RE = re.compile(r"^(.*) \((\d{4})\)$")


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


def build_query(row: pd.Series, max_chars: int = 1800) -> str:
    note = str(row.get("submitter_note") or "").strip()
    corr = str(row.get("proposed_correction") or "").strip()
    excerpt = str(row.get("original_excerpt") or "").strip()
    # Prefer correction/note (short, high-signal) then a head+tail excerpt window.
    head = f"submitter note: {note}\nproposed correction: {corr}\noriginal excerpt: "
    budget = max(200, max_chars - len(head))
    if len(excerpt) > budget:
        keep = budget // 2
        excerpt = excerpt[:keep] + " ... " + excerpt[-keep:]
    return head + excerpt


def format_candidate(raw: str) -> str:
    """Make year an explicit field so version collisions are easier to learn."""
    title, year = parse_ref(raw)
    return f"{title} | year {year}"


def label_index(row: pd.Series) -> int:
    card = json.loads(row["provenance_card"])
    for i, col in enumerate(REF_COLS):
        title, year = parse_ref(row[col])
        if title == card["source_title"] and year == card["source_year"]:
            return i
    raise ValueError(f"Label not in candidates for {row['id']}")


def provenance_score(title_ok: np.ndarray, year_ok: np.ndarray, conf: np.ndarray) -> dict:
    title_rate = float(title_ok.mean())
    year_rate = float(year_ok.mean())
    y = (title_ok & year_ok).astype(np.float64)
    p = np.clip(conf / 100.0, 0.0, 1.0)
    brier = float(((p - y) ** 2).mean())
    base = float(y.mean())
    ref = base * (1.0 - base) if 0.0 < base < 1.0 else (1.0 / 16.0) * (15.0 / 16.0)
    cal = float(np.clip(1.0 - brier / ref, 0.0, 1.0))
    score = 0.85 * title_rate + 0.05 * year_rate + 0.10 * cal
    return {
        "title_rate": title_rate,
        "year_rate": year_rate,
        "cal": cal,
        "score": score,
        "base": base,
    }


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: list[int] | None, max_chars: int):
        self.queries = [build_query(r, max_chars) for _, r in df.iterrows()]
        self.cands = [
            [format_candidate(str(r[c])) for c in REF_COLS] for _, r in df.iterrows()
        ]
        self.raw_cands = [[str(r[c]) for c in REF_COLS] for _, r in df.iterrows()]
        self.labels = labels if labels is not None else [-1] * len(df)
        self.ids = df["id"].tolist()

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        return self.queries[idx], self.cands[idx], self.labels[idx], self.ids[idx]


class CrossEncoder(nn.Module):
    """Keep a pretrained sequence-classification ranking head when available."""

    def __init__(self, model_name: str):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        # Prefer the published 1-logit ranking head (e.g. ms-marco cross-encoders).
        num_labels = getattr(cfg, "num_labels", 1) or 1
        if num_labels < 1:
            num_labels = 1
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        if logits.ndim == 2 and logits.size(-1) == 1:
            return logits.squeeze(-1)
        if logits.ndim == 2 and logits.size(-1) == 2:
            # Some CE checkpoints use 2-class heads; use positive-class logit.
            return logits[:, 1]
        return logits.squeeze(-1)


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        queries, cand_lists, labels, ids = zip(*batch)
        flat_q, flat_c = [], []
        for q, cands in zip(queries, cand_lists):
            for c in cands:
                flat_q.append(q)
                flat_c.append(c)
        enc = tokenizer(
            flat_q,
            flat_c,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        bsz = len(queries)
        for k in enc:
            enc[k] = enc[k].view(bsz, 16, -1)
        return enc, torch.tensor(labels, dtype=torch.long), list(ids)

    return collate


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    all_logits, all_ids = [], []
    for enc, _, ids in loader:
        bsz = enc["input_ids"].size(0)
        flat = {k: v.view(bsz * 16, -1).to(device) for k, v in enc.items()}
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            logits = model(flat["input_ids"], flat["attention_mask"]).view(bsz, 16)
        all_logits.append(logits.float().cpu().numpy())
        all_ids.extend(ids)
    return np.concatenate(all_logits, 0), all_ids


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Maximize NLL calibration of softmax(logits/T)."""
    y = torch.tensor(labels, dtype=torch.long)
    z = torch.tensor(logits, dtype=torch.float32)
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.3, 5.0, 48):
        nll = float(torch.nn.functional.cross_entropy(z / t, y))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def evaluate(logits: np.ndarray, df: pd.DataFrame, temperature: float) -> dict:
    probs = torch.softmax(torch.tensor(logits) / temperature, dim=-1).numpy()
    pred = probs.argmax(1)
    conf = probs.max(1) * 100.0
    title_ok = np.zeros(len(df), dtype=np.int32)
    year_ok = np.zeros(len(df), dtype=np.int32)
    for i, (_, row) in enumerate(df.iterrows()):
        card = json.loads(row["provenance_card"])
        title, year = parse_ref(row[REF_COLS[pred[i]]])
        title_ok[i] = int(title == card["source_title"])
        year_ok[i] = int(year == card["source_year"])
    out = provenance_score(title_ok, year_ok, conf)
    out["temperature"] = temperature
    out["mean_conf"] = float(conf.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"G:\ml\gpuchals\newone\dataset\public")
    ap.add_argument("--out", default=r"G:\ml\gpuchals\newone\research\ce_out")
    ap.add_argument("--model", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--max-chars", type=int, default=1600)
    ap.add_argument("--val-year", type=int, default=2011)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["validate", "full"], default="validate")
    args = ap.parse_args()

    t0 = time.time()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    train["label"] = train.apply(label_index, axis=1)
    train["src_year"] = train["provenance_card"].apply(lambda s: int(json.loads(s)["source_year"]))

    if args.subset:
        train = train.sample(n=min(args.subset, len(train)), random_state=args.seed).reset_index(drop=True)
        test = test.head(min(200, len(test))).reset_index(drop=True)

    if args.mode == "validate":
        tr_df = train[train["src_year"] < args.val_year].reset_index(drop=True)
        va_df = train[train["src_year"] >= args.val_year].reset_index(drop=True)
    else:
        tr_df = train.reset_index(drop=True)
        va_df = train.sample(n=min(400, len(train)), random_state=args.seed).reset_index(drop=True)

    print(
        f"device={device} model={args.model} train={len(tr_df)} val={len(va_df)} "
        f"mode={args.mode}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    collate = make_collate(tok, args.max_length)
    model = CrossEncoder(args.model).to(device)

    tr_loader = DataLoader(
        PairDataset(tr_df, tr_df["label"].tolist(), args.max_chars),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    va_loader = DataLoader(
        PairDataset(va_df, va_df["label"].tolist(), args.max_chars),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(tr_loader) / args.grad_accum) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    best = {"score": -1.0}
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, (enc, labels, _) in enumerate(tr_loader):
            bsz = enc["input_ids"].size(0)
            flat = {k: v.view(bsz * 16, -1).to(device) for k, v in enc.items()}
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(flat["input_ids"], flat["attention_mask"]).view(bsz, 16)
                loss = loss_fn(logits, labels) / args.grad_accum
            scaler.scale(loss).backward()
            running += float(loss.item()) * args.grad_accum
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(tr_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()
                global_step += 1
            if (step + 1) % 50 == 0:
                print(
                    f"epoch {epoch+1} step {step+1}/{len(tr_loader)} "
                    f"loss={running/(step+1):.4f} elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )

        va_logits, _ = predict_logits(model, va_loader, device)
        temp = fit_temperature(va_logits, va_df["label"].to_numpy())
        metrics = evaluate(va_logits, va_df, temp)
        print(f"epoch {epoch+1} val {metrics}", flush=True)
        if metrics["score"] > best["score"]:
            best = {**metrics, "epoch": epoch + 1}
            torch.save(
                {"model": model.state_dict(), "temperature": temp, "args": vars(args)},
                out_dir / "best.pt",
            )
            np.save(out_dir / "val_logits.npy", va_logits)

    print("BEST", best, flush=True)

    if args.mode == "full":
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        temp = float(ckpt["temperature"])
        te_loader = DataLoader(
            PairDataset(test, None, args.max_chars),
            batch_size=max(1, args.batch_size),
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        te_logits, te_ids = predict_logits(model, te_loader, device)
        probs = torch.softmax(torch.tensor(te_logits) / temp, dim=-1).numpy()
        pred = probs.argmax(1)
        conf = probs.max(1) * 100.0
        rows = []
        for i, rid in enumerate(te_ids):
            title, year = parse_ref(test.loc[test["id"] == rid, REF_COLS[pred[i]]].iloc[0])
            # Prefer the row order from test
            pass
        # rebuild in test order
        id_to_i = {rid: i for i, rid in enumerate(te_ids)}
        for _, row in test.iterrows():
            i = id_to_i[row["id"]]
            title, year = parse_ref(row[REF_COLS[pred[i]]])
            card = json.dumps({"source_title": title, "source_year": year}, ensure_ascii=False)
            rows.append({"id": row["id"], "provenance_card": card, "confidence": float(conf[i])})
        sub = pd.DataFrame(rows)
        sub_path = out_dir / "submission.csv"
        sub.to_csv(sub_path, index=False)
        print("wrote", sub_path, sub.shape, flush=True)

    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
