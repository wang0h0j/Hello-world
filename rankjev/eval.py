"""Evaluate a saved rank head.

  python -m rankjev.score \
    --ckpt runs/kq-h16d4.pt \
    --data data/listwise/dev.jsonl \
    --device mps --limit 300 --show 0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .data import RankCollator, RankSample
from .lcs import lcs_ratio, orders_from_scores
from .metrics import eval_batch
from .model import HFRankModel
from .train import amp_dtype, pick_device


class RowDataset(Dataset):
    def __init__(self, path: Path, limit_per_source: int = 0, limit: int = 0):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if limit_per_source:
            kept, n = [], defaultdict(int)
            for row in rows:
                src = row.get("source", "?")
                if n[src] >= limit_per_source:
                    continue
                n[src] += 1
                kept.append(row)
            rows = kept
        if limit:
            rows = rows[:limit]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_rows(backend, tokenizer, max_len):
    inner = RankCollator(backend=backend, tokenizer=tokenizer, max_len=max_len)

    def _fn(rows: list[dict]):
        samples = [RankSample(r["stem"], list(r["options"]), [float(x) for x in r["gold"]]) for r in rows]
        batch = inner(samples)
        batch["sources"] = [r.get("source", "?") for r in rows]
        batch["ids"] = [r.get("id", "") for r in rows]
        batch["options"] = [r["options"] for r in rows]
        batch["stems"] = [r["stem"] for r in rows]
        return batch

    return _fn


def load_model(ckpt_path: Path, device: torch.device, model_override: str = ""):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    orig = model_override or ckpt.get("base") or ckpt.get("args", {}).get("model")
    finetuned = ckpt.get("backbone_dir")
    if not orig and not finetuned:
        raise ValueError("checkpoint has no base model path")
    dtype = amp_dtype(device)
    head_kind = (ckpt.get("args") or {}).get("head") or "linear"
    model = HFRankModel(
        finetuned or orig, freeze_backbone=True, dtype=dtype, head_kind=head_kind
    )
    if ckpt.get("model") and not finetuned:
        model.load_state_dict(ckpt["model"], strict=False)
    if ckpt.get("head"):
        model.head.load_state_dict(ckpt["head"])
    model.to(device).eval()
    print(f"eval backbone={finetuned or orig}", flush=True)
    return model, ckpt


def show_example(stem, options, gold, pred, src, rid):
    mask = [False] * len(gold)
    pred_o = orders_from_scores(pred, mask)
    gold_o = orders_from_scores(gold, mask)
    print(f"\n--- {src}  {rid}  LCS={lcs_ratio(pred_o, gold_o):.2f} ---")
    print(stem.replace("\n", " | ")[:200])
    print(f"{'gold':>8} {'pred':>8}  option")
    for g, p, opt in sorted(zip(gold, pred, options), key=lambda x: -x[0]):
        print(f"{g:8.2f} {p:8.2f}  {opt[:90]}")


def run(args):
    device = pick_device(args.device)
    print(f"device={device} ckpt={args.ckpt}", flush=True)
    model, ckpt = load_model(Path(args.ckpt), device, args.model)
    if ckpt.get("val"):
        print("train-time val:", {k: round(v, 3) for k, v in ckpt["val"].items()}, flush=True)

    from transformers import AutoTokenizer

    tok_src = args.model or ckpt.get("base") or ckpt.get("backbone_dir")
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    max_len = args.max_len or int((ckpt.get("args") or {}).get("max_len") or 512)
    ds = RowDataset(
        Path(args.data),
        limit_per_source=args.limit_per_source,
        limit=getattr(args, "limit", 0),
    )
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=collate_rows("hf", tok, max_len),
    )
    print(f"eval rows={len(ds)} max_len={max_len}", flush=True)

    grouped = defaultdict(
        lambda: {
            "top1": 0.0, "hit3": 0.0, "mrr": 0.0, "ndcg": 0.0,
            "spearman": 0.0, "lcs": 0.0, "confidence": 0.0,
            "n": 0, "lcs_n": 0,
        }
    )
    shown = defaultdict(int)
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            src = batch["sources"][0]
            gpu = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            scores = model(
                gpu["input_ids"], gpu["attention_mask"], gpu["option_pos"],
                answer_pos=gpu.get("answer_pos"),
            )
            ev = eval_batch(scores, gpu["gold"], gpu["padding_mask"])
            g = grouped[src]
            for k in ("top1", "hit3", "mrr", "ndcg", "spearman", "confidence"):
                g[k] += ev[k] * ev["n"]
            g["lcs"] += ev["lcs"] * ev["lcs_n"]
            g["lcs_n"] += ev["lcs_n"]
            g["n"] += ev["n"]
            if shown[src] < args.show:
                valid = ~gpu["padding_mask"][0]
                show_example(
                    batch["stems"][0], batch["options"][0],
                    gpu["gold"][0][valid].tolist(), scores[0][valid].tolist(),
                    src, batch["ids"][0],
                )
                shown[src] += 1
            if i == 1 or i % 20 == 0 or i == len(loader):
                print(f"eval-progress {i}/{len(loader)}", flush=True)

    print("\nby source")
    print(
        f"{'source':<22} {'n':>4} {'top1':>6} {'hit3':>6} {'mrr':>6} "
        f"{'ndcg':>6} {'spearman':>9} {'lcs*':>6} {'conf':>6}"
    )
    tot = {
        "top1": 0.0, "hit3": 0.0, "mrr": 0.0, "ndcg": 0.0,
        "spearman": 0.0, "lcs": 0.0, "confidence": 0.0, "n": 0,
        "lcs_n": 0,
    }
    for src in sorted(grouped):
        g = grouped[src]
        n = max(g["n"], 1)
        lcs_n = max(g["lcs_n"], 1)
        print(
            f"{src:<22} {g['n']:4.0f} {g['top1']/n:6.3f} {g['hit3']/n:6.3f} "
            f"{g['mrr']/n:6.3f} {g['ndcg']/n:6.3f} {g['spearman']/n:9.3f} "
            f"{g['lcs']/lcs_n:6.3f} {g['confidence']/n:6.3f}"
        )
        for k in tot:
            tot[k] += g[k]
    n = max(tot["n"], 1)
    lcs_n = max(tot["lcs_n"], 1)
    print(
        f"{'ALL':<22} {tot['n']:4.0f} {tot['top1']/n:6.3f} {tot['hit3']/n:6.3f} "
        f"{tot['mrr']/n:6.3f} {tot['ndcg']/n:6.3f} {tot['spearman']/n:9.3f} "
        f"{tot['lcs']/lcs_n:6.3f} {tot['confidence']/n:6.3f}"
    )
    print("* lcs is averaged only over strict full-order rows (no gold ties)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/kq-h16d4.pt")
    p.add_argument("--data", default="data/listwise/dev.jsonl")
    p.add_argument("--model", default="", help="override backbone path")
    p.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    p.add_argument(
        "--max-len", type=int, default=0,
        help="0 = use the checkpoint training max_len",
    )
    p.add_argument("--limit-per-source", type=int, default=0, help="0 = all")
    p.add_argument("--limit", type=int, default=0, help="total row limit; 0 = all")
    p.add_argument("--show", type=int, default=1, help="examples to print per source")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
