"""Evaluate a compact K/Q checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import eval as shared_eval
from .head import CompactKQHFRankModel
from .train import amp_dtype


def load_kq_model(ckpt_path: Path, device: torch.device, model_override: str = ""):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    orig = model_override or ckpt.get("base") or ckpt.get("args", {}).get("model")
    finetuned = ckpt.get("backbone_dir")
    if not orig and not finetuned:
        raise ValueError("checkpoint has no base model path")
    saved_args = ckpt.get("args") or {}
    relation_heads = int(saved_args.get("relation_heads") or 16)
    relation_head_dim = int(saved_args.get("relation_head_dim") or 4)
    model = CompactKQHFRankModel(
        finetuned or orig,
        freeze_backbone=True,
        dtype=amp_dtype(device),
        relation_heads=relation_heads,
        relation_head_dim=relation_head_dim,
    )
    if ckpt.get("head"):
        model.head.load_state_dict(ckpt["head"])
    model.to(device).eval()
    print(
        f"eval backbone={finetuned or orig} "
        f"relation={relation_heads}x{relation_head_dim}",
        flush=True,
    )
    return model, ckpt


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the compact K/Q head")
    p.add_argument("--ckpt", default="weights/qwen35-9b-kq-head.pt")
    p.add_argument("--data", default="data/listwise/dev.jsonl")
    p.add_argument("--model", default="", help="override backbone path")
    p.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    p.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="0 = use the checkpoint training max_len",
    )
    p.add_argument("--limit-per-source", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="total row limit; 0 = all")
    p.add_argument("--show", type=int, default=1)
    shared_eval.load_model = load_kq_model
    shared_eval.run(p.parse_args(argv))


if __name__ == "__main__":
    main()
