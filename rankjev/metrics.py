"""Eval metrics. Confidence is derived from the score distribution, never trained.

The readout is a ranking of all K options. top1 is only the first name;
hit@k / topk overlap score the prefix of that ranking.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .lcs import lcs_ratio, orders_from_scores
from .losses import average_ranks, pearson_ic

RANK_KEYS = ("top1", "hit3", "mrr", "ndcg", "spearman", "lcs", "confidence")
TOPK_EVAL = 10


def spearman_topk(pred: torch.Tensor, true: torch.Tensor, k: int = TOPK_EVAL) -> float:
    kk = min(int(k), int(pred.numel()))
    if kk < 2:
        return 0.0
    if kk == int(true.numel()):
        idx = torch.arange(true.numel(), device=true.device)
    else:
        threshold = torch.topk(true, k=kk, largest=True).values[-1]
        idx = torch.nonzero(true >= threshold, as_tuple=False).squeeze(-1)
    pred_rank = average_ranks(pred[idx])
    gold_rank = average_ranks(true[idx])
    return float(pearson_ic(pred_rank, gold_rank).item())


def hit_at_k(pred_order: Sequence[int], gold_order: Sequence[int], k: int) -> float:
    """Gold winner sits in the predicted top-k."""
    if not gold_order:
        return 1.0
    kk = min(k, len(pred_order))
    return float(gold_order[0] in pred_order[:kk])


def topk_overlap(pred_order: Sequence[int], gold_order: Sequence[int], k: int = 3) -> float:
    """|pred[:k] ∩ gold[:k]| / k. Prefix set agreement of the output ranking."""
    kk = min(k, len(pred_order), len(gold_order))
    if kk == 0:
        return 1.0
    return len(set(pred_order[:kk]) & set(gold_order[:kk])) / kk


def ndcg_score(pred: torch.Tensor, true: torch.Tensor) -> float:
    """NDCG over tie-aware ordinal relevance, independent of raw gold gaps."""
    relevance = average_ranks(true).float()
    pred_idx = torch.argsort(pred, descending=True)
    ideal_idx = torch.argsort(relevance, descending=True)
    discounts = torch.log2(
        torch.arange(2, relevance.numel() + 2, device=true.device, dtype=relevance.dtype)
    )
    gains = torch.pow(2.0, relevance) - 1.0
    dcg = (gains[pred_idx] / discounts).sum()
    ideal = (gains[ideal_idx] / discounts).sum()
    if float(ideal) <= 0.0:
        return 1.0
    return float((dcg / ideal).item())


def derived_confidence(scores: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """c = (p_max - 1/K) / (1 - 1/K), Noul-style. No extra head."""
    valid = ~padding_mask
    masked = scores.masked_fill(padding_mask, -1e9)
    k = valid.sum(dim=-1).clamp(min=2).float()
    p = torch.softmax(masked, dim=-1) * valid.float()
    p_max = p.max(dim=-1).values
    return (p_max - 1.0 / k) / (1.0 - 1.0 / k)


@torch.no_grad()
def eval_batch(
    scores: torch.Tensor,
    gold: torch.Tensor,
    padding_mask: torch.Tensor,
) -> dict[str, float]:
    valid = ~padding_mask
    acc = {k: 0.0 for k in ("top1", "hit3", "mrr", "ndcg", "spearman", "lcs")}
    n = 0
    lcs_n = 0
    for i in range(scores.size(0)):
        m = valid[i]
        if int(m.sum()) < 2:
            continue
        pred = scores[i][m]
        true = gold[i][m]
        mask_row = [False] * int(m.sum())
        pred_order = orders_from_scores(pred.tolist(), mask_row)
        gold_order = orders_from_scores(true.tolist(), mask_row)
        winners = set(torch.nonzero(true == true.max(), as_tuple=False).squeeze(-1).tolist())
        first_winner = min(j for j, idx in enumerate(pred_order) if idx in winners)
        acc["top1"] += float(pred_order[0] in winners)
        acc["hit3"] += float(first_winner < min(3, len(pred_order)))
        acc["mrr"] += 1.0 / (first_winner + 1)
        acc["ndcg"] += ndcg_score(pred, true)
        acc["spearman"] += float(
            pearson_ic(average_ranks(pred), average_ranks(true)).item()
        )
        if int(torch.unique(true).numel()) == int(true.numel()):
            acc["lcs"] += lcs_ratio(pred_order, gold_order)
            lcs_n += 1
        n += 1
    if n == 0:
        return {**{k: 0.0 for k in RANK_KEYS}, "n": 0, "lcs_n": 0}
    conf = derived_confidence(scores, padding_mask).mean().item()
    result = {k: acc[k] / n for k in ("top1", "hit3", "mrr", "ndcg", "spearman")}
    result["lcs"] = acc["lcs"] / max(lcs_n, 1)
    return result | {"confidence": conf, "n": n, "lcs_n": lcs_n}
