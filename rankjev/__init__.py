"""Jev-style listwise ranker: score head + Spearman loss, LCS proofread."""

import logging
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class _DropKernelFallback(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "falling back to its reference PyTorch implementation" not in record.getMessage()


def _silence_kernel_fallback() -> None:
    """Qwen3.5 on MPS has no FLA/causal_conv1d; the advisory is correct and ugly."""
    for name in (
        "transformers",
        "transformers.integrations.hub_kernels",
    ):
        log = logging.getLogger(name)
        if not any(isinstance(f, _DropKernelFallback) for f in log.filters):
            log.addFilter(_DropKernelFallback())


_silence_kernel_fallback()

from .lcs import lcs_length, lcs_ratio, lcs_proofread
from .losses import ic_rank_loss, pairwise_rank_loss, soft_spearman_loss
from .metrics import derived_confidence, eval_batch, hit_at_k, spearman_topk, topk_overlap
from .data import JsonlRankDataset, RankSample

__all__ = [
    "ic_rank_loss",
    "pairwise_rank_loss",
    "soft_spearman_loss",
    "lcs_length",
    "lcs_ratio",
    "lcs_proofread",
    "derived_confidence",
    "eval_batch",
    "hit_at_k",
    "spearman_topk",
    "topk_overlap",
]
