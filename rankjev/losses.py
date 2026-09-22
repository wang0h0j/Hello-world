"""Differentiable ranking losses.

Primary: Pearson(pred, gold_ranks) == the Rank-IC proxy used in
`06_train_manifold_transformer.py`. Gold is already an order, so this
is the trainable Spearman stand-in (hard argsort is not differentiable).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _valid_rows(y_pred: torch.Tensor, y_true: torch.Tensor, padding_mask: torch.Tensor):
    """padding_mask: True = pad, False = valid (same as the trading script)."""
    valid = ~padding_mask
    batch = y_pred.size(0)
    for i in range(batch):
        m = valid[i]
        if int(m.sum()) < 2:
            continue
        yield y_pred[i][m], y_true[i][m]


def pearson_ic(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    zero = pred.sum() * 0.0
    pred_mu = pred - pred.mean()
    true_mu = true - true.mean()
    pred_std = torch.sqrt((pred_mu ** 2).sum() + 1e-8)
    true_std = torch.sqrt((true_mu ** 2).sum() + 1e-8)
    ic = ((pred_mu * true_mu).sum() / (pred_std * true_std)).clamp(-1.0, 1.0)
    bad = (
        ~torch.isfinite(pred).all()
        | ~torch.isfinite(true).all()
        | (pred_std < 1e-6)
        | (true_std < 1e-6)
        | ~torch.isfinite(ic)
    )
    return torch.where(bad, zero, ic)


def ic_rank_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """L = 1 - IC. Strictly an order loss; ignores absolute magnitude."""
    losses = [1.0 - pearson_ic(p, t) for p, t in _valid_rows(y_pred, y_true, padding_mask)]
    if not losses:
        return y_pred.sum() * 0.0
    return torch.stack(losses).mean()


def pairwise_one(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """RankNet / Bradley-Terry: gold i>j ⇒ softplus(-(z_i−z_j)).

    On Choice/Noul this is winner-vs-rest. Saturates; eats Δz.
    """
    if pred.numel() < 2:
        return pred.sum() * 0.0
    diff_true = true.unsqueeze(0) - true.unsqueeze(1)
    diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)
    pair_mask = diff_true > 0
    if not bool(pair_mask.any()):
        return pred.sum() * 0.0
    return F.softplus(-diff_pred[pair_mask]).mean()


def pairwise_rank_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Logistic pairwise: every gold i>j should have z_i > z_j.

    More stable than IC when K is small (3–8 options).
    """
    losses = [pairwise_one(p, t) for p, t in _valid_rows(y_pred, y_true, padding_mask)]
    if not losses:
        return y_pred.sum() * 0.0
    return torch.stack(losses).mean()


def soft_ranks(x: torch.Tensor, mask: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Differentiable ranks, higher value → larger rank number.

    rank_i ≈ sum_j sigmoid((x_i - x_j) / tau)  (includes diagonal 0.5)
    Invalid positions are pushed to -inf so they do not win comparisons.
    """
    valid = ~mask
    x_eff = x.masked_fill(mask, -1e9)
    diff = (x_eff.unsqueeze(-1) - x_eff.unsqueeze(-2)) / tau
    ranks = torch.sigmoid(diff).sum(dim=-1)
    return ranks * valid.float()


def soft_spearman_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    padding_mask: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """Pearson(soft_rank(pred), gold). Closer to textbook Spearman than raw IC."""
    pred_r = soft_ranks(y_pred, padding_mask, tau=tau)
    return ic_rank_loss(pred_r, y_true, padding_mask)


def vocab_align_loss(
    y_pred: torch.Tensor,
    y_vocab: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """1-IC between the trainable head and frozen lm_head letter ranking."""
    return ic_rank_loss(y_pred, y_vocab.detach(), padding_mask)


def _n_unique(true: torch.Tensor) -> int:
    return int(torch.unique(torch.round(true, decimals=5)).numel())


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Tie-aware ascending midranks: larger value receives a larger rank."""
    if values.numel() == 0:
        return values
    x = values.reshape(-1)
    equal_mask = torch.isclose(
        x.unsqueeze(1), x.unsqueeze(0), rtol=1e-5, atol=1e-6
    )
    less = (
        (x.unsqueeze(1) > x.unsqueeze(0)) & ~equal_mask
    ).sum(dim=1).to(x.dtype)
    equal = equal_mask.sum(dim=1).to(x.dtype)
    return less + 0.5 * (equal - 1.0)


def diff_huber_loss(pred: torch.Tensor, true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber(Δz − Δŷ). Gold is z-scored so gaps are O(1); |Δz| is learned."""
    y = (true - true.mean()) / (true.std(unbiased=False) + 1e-6)
    k = pred.numel()
    if k < 2:
        return pred.sum() * 0.0
    idx = torch.triu_indices(k, k, offset=1, device=pred.device)
    dz = pred[idx[0]] - pred[idx[1]]
    dy = y[idx[0]] - y[idx[1]]
    residual = dz - dy
    return F.huber_loss(residual, torch.zeros_like(residual), reduction="mean", delta=delta)


def listnet_loss(pred: torch.Tensor, true: torch.Tensor, tau_y: float = 0.5) -> torch.Tensor:
    """CE(softmax(z), softmax(y/τ)). Grad w.r.t. z is p − q."""
    q = torch.softmax(true / tau_y, dim=-1)
    log_p = torch.log_softmax(pred, dim=-1)
    return -(q * log_p).sum()


def gap_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Graded gold → Huber on pairwise gaps; winner-only / binary → ListNet."""
    if pred.numel() < 2:
        return pred.sum() * 0.0
    if _n_unique(true) >= 3:
        return diff_huber_loss(pred, true)
    return listnet_loss(pred, true)


def gap_rank_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    losses = [gap_loss(p, t) for p, t in _valid_rows(y_pred, y_true, padding_mask)]
    if not losses:
        return y_pred.sum() * 0.0
    return torch.stack(losses).mean()


def _gold_topk_idx(true: torch.Tensor, k: int) -> torch.Tensor:
    kk = min(int(k), int(true.numel()))
    if kk >= int(true.numel()):
        return torch.arange(true.numel(), device=true.device)
    threshold = torch.topk(true, k=max(kk, 1), largest=True).values[-1]
    # Keep the complete tie group at the cutoff instead of selecting tied
    # options according to incidental tensor position.
    return torch.nonzero(true >= threshold, as_tuple=False).squeeze(-1)


def topk_spearman_loss(pred: torch.Tensor, true: torch.Tensor, k: int = 10) -> torch.Tensor:
    """1-IC against tie-aware gold ranks on the gold top-k slice."""
    if pred.numel() < 2:
        return pred.sum() * 0.0
    idx = _gold_topk_idx(true, k)
    if idx.numel() < 2:
        return pred.sum() * 0.0
    return 1.0 - pearson_ic(pred[idx], average_ranks(true[idx]))


def top1_ce_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Differentiable top-1: −log softmax(z)[gold winner]."""
    if pred.numel() < 2:
        return pred.sum() * 0.0
    winner = true.argmax().unsqueeze(0)
    return F.cross_entropy(pred.unsqueeze(0), winner)


def topk_spearman_rank_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    padding_mask: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    losses = [topk_spearman_loss(p, t, k=k) for p, t in _valid_rows(y_pred, y_true, padding_mask)]
    if not losses:
        return y_pred.sum() * 0.0
    return torch.stack(losses).mean()


def top1_rank_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    losses = [top1_ce_loss(p, t) for p, t in _valid_rows(y_pred, y_true, padding_mask)]
    if not losses:
        return y_pred.sum() * 0.0
    return torch.stack(losses).mean()


def order_only_rank_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    padding_mask: torch.Tensor,
    k: int = 10,
) -> dict[str, torch.Tensor]:
    """Supervise order only.

    unique(gold) ≥ 3 → 1-IC on gold top-k (true ranking).
    unique(gold) ≤ 2 → RankNet winner-vs-rest (Choice/Noul). No CE.
    """
    zero = y_pred.sum() * 0.0
    routed: list[torch.Tensor] = []
    graded: list[torch.Tensor] = []
    choice: list[torch.Tensor] = []
    for pred, true in _valid_rows(y_pred, y_true, padding_mask):
        if _n_unique(true) >= 3:
            v = topk_spearman_loss(pred, true, k=k)
            graded.append(v)
        else:
            v = pairwise_one(pred, true)
            choice.append(v)
        routed.append(v)
    return {
        "total": torch.stack(routed).mean() if routed else zero,
        "sp10": torch.stack(graded).mean() if graded else zero,
        "pairwise": torch.stack(choice).mean() if choice else zero,
        "n_graded": len(graded),
        "n_choice": len(choice),
    }


def rank_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    padding_mask: torch.Tensor,
    pairwise_weight: float = 0.3,
    use_soft_spearman: bool = False,
    tau: float = 0.5,
    y_vocab: torch.Tensor | None = None,
    vocab_weight: float = 0.0,
    ic_weight: float = 1.0,
    gap_weight: float = 0.0,
    topk: int = 10,
    topk_weight: float = 0.0,
    top1_weight: float = 0.0,
    order_only: bool = False,
) -> dict[str, torch.Tensor]:
    zero = y_pred.sum() * 0.0
    if order_only:
        routed = order_only_rank_loss(y_pred, y_true, padding_mask, k=topk)
        return {
            "total": routed["total"],
            "gap": zero,
            "ic": zero,
            "pairwise": routed["pairwise"],
            "align": zero,
            "sp10": routed["sp10"],
            "top1ce": zero,
            "n_graded": routed["n_graded"],
            "n_choice": routed["n_choice"],
        }
    gap_l = gap_rank_loss(y_pred, y_true, padding_mask) if gap_weight else zero
    ic_l = (
        soft_spearman_loss(y_pred, y_true, padding_mask, tau=tau)
        if use_soft_spearman
        else ic_rank_loss(y_pred, y_true, padding_mask)
    )
    pair_l = pairwise_rank_loss(y_pred, y_true, padding_mask) if pairwise_weight else zero
    align_l = (
        vocab_align_loss(y_pred, y_vocab, padding_mask)
        if y_vocab is not None and vocab_weight
        else zero
    )
    sp10_l = topk_spearman_rank_loss(y_pred, y_true, padding_mask, k=topk) if topk_weight else zero
    top1_l = top1_rank_loss(y_pred, y_true, padding_mask) if top1_weight else zero
    total = (
        gap_weight * gap_l
        + ic_weight * ic_l
        + pairwise_weight * pair_l
        + vocab_weight * align_l
        + topk_weight * sp10_l
        + top1_weight * top1_l
    )
    return {
        "total": total,
        "gap": gap_l,
        "ic": ic_l,
        "pairwise": pair_l,
        "align": align_l,
        "sp10": sp10_l,
        "top1ce": top1_l,
        "n_graded": 0,
        "n_choice": 0,
    }
