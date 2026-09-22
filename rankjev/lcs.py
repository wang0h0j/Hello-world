"""LCS is discrete alignment, not a neural head.

Use it to:
  1. score a predicted permutation against gold (metric)
  2. proofread a predicted order against a constraint sequence
"""

from __future__ import annotations

from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def lcs_length(a: Sequence[T], b: Sequence[T]) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[m]


def lcs_sequence(a: Sequence[T], b: Sequence[T]) -> list[T]:
    """Items of `a` that form an LCS with `b` (stable in `a`'s order)."""
    if not a or not b:
        return []
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    out: list[T] = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    out.reverse()
    return out


def lcs_ratio(pred_order: Sequence[T], gold_order: Sequence[T]) -> float:
    if not gold_order:
        return 1.0
    return lcs_length(pred_order, gold_order) / len(gold_order)


def lcs_proofread(pred_order: Sequence[T], constraint: Sequence[T]) -> list[T]:
    """Keep the longest subsequence of pred that respects constraint order.

    Leftovers are appended in the original predicted order. This is decode-time
    alignment, not a trainable readout.
    """
    kept = lcs_sequence(pred_order, constraint)
    kept_set = set(kept)
    rest = [x for x in pred_order if x not in kept_set]
    return list(kept) + rest


def orders_from_scores(scores: Iterable[float], mask: Iterable[bool] | None = None) -> list[int]:
    """Descending score order over valid indices. Higher score = better."""
    pairs = []
    for i, s in enumerate(scores):
        if mask is not None and mask[i]:
            continue
        pairs.append((s, i))
    pairs.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in pairs]
