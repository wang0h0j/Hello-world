"""Frozen Qwen3.5-9B plus a grouped compact K/Q relation space.

    K_i = W_k LN(h_i)
    Q   = tanh(W_q LN(h_answer))
    z_i = sum_j <K_ij, Q_j> / sqrt(4×16)

The head has about 0.541M parameters at hidden size 4096.

  RANKJEV_MODEL=/path/to/Qwen3.5-9B python -m rankjev.fit
  python -m rankjev.fit --resume auto
"""

from __future__ import annotations

import os
import sys

MODEL = os.environ.get("RANKJEV_MODEL", "Qwen/Qwen3.5-9B")
SAVE = "runs/kq-h16d4.pt"
RELATION_HEADS = 16
RELATION_HEAD_DIM = 4

DEFAULTS = [
    "--device", "mps",
    "--model", MODEL,
    "--data", "data/listwise/train.jsonl",
    "--val-data", "data/listwise/dev.jsonl",
    "--batch", "1",
    "--steps", "22062",
    "--lr", "3e-5",
    "--warmup", "400",
    "--lr-schedule", "cosine",
    "--min-lr-ratio", "0.1",
    "--max-len", "768",
    "--log-every", "20",
    "--save-every", "2000",
    "--pairwise", "0",
    "--vocab-align", "0",
    "--ic-weight", "0",
    "--gap-weight", "0",
    "--topk", "10",
    "--topk-weight", "0",
    "--top1-weight", "0",
    "--order-only",
    "--shuffle-options",
    "--optim", "adamw",
    "--save-optim",
    "--save", SAVE,
]


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    from . import train
    from .head import CompactKQHFRankModel

    args = train.parse_args(DEFAULTS + argv)
    args.head = "compact-kq"
    args.head_rank = RELATION_HEADS * RELATION_HEAD_DIM
    args.relation_heads = RELATION_HEADS
    args.relation_head_dim = RELATION_HEAD_DIM

    def build_kq_model(
        name: str,
        freeze_backbone: bool,
        dtype,
        with_lm_head: bool = False,
        head_kind: str = "linear",
    ):
        del head_kind
        return CompactKQHFRankModel(
            name,
            freeze_backbone=freeze_backbone,
            dtype=dtype,
            with_lm_head=with_lm_head,
            relation_heads=RELATION_HEADS,
            relation_head_dim=RELATION_HEAD_DIM,
        )

    train.build_model = build_kq_model
    train.run(args)


if __name__ == "__main__":
    main()
