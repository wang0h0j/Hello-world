"""Compact K/Q relation head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import HFRankModel, gather_option_states


class CompactKQRankHead(nn.Module):
    """Grouped compact K/Q relation space.

        K_i = W_k LN(h_i)              in R^(heads × head_dim)
        Q   = tanh(W_q LN(h_answer))   in R^(heads × head_dim)
        r_ij = <K_ij, Q_j> / sqrt(head_dim)
        z_i = exp(s) sum_j r_ij / sqrt(heads)

    K keeps its magnitude and is not normalized or squashed. Q is bounded.
    Grouping exposes individual relation channels while their sum remains an
    exact scalar potential.
    """

    def __init__(
        self,
        hidden_size: int,
        relation_heads: int = 16,
        relation_head_dim: int = 4,
    ):
        super().__init__()
        if relation_heads < 1 or relation_head_dim < 1:
            raise ValueError("relation_heads and relation_head_dim must be positive")
        self.relation_heads = int(relation_heads)
        self.relation_head_dim = int(relation_head_dim)
        self.relation_dim = self.relation_heads * self.relation_head_dim
        self.option_norm = nn.LayerNorm(hidden_size)
        self.state_norm = nn.LayerNorm(hidden_size)
        self.key_proj = nn.Linear(hidden_size, self.relation_dim, bias=False)
        self.query_proj = nn.Linear(hidden_size, self.relation_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
        self._last_diagnostics: dict[str, float] = {}

    def forward(
        self,
        hidden: torch.Tensor,
        option_pos: torch.Tensor,
        answer_pos=None,
    ) -> torch.Tensor:
        option_states = self.option_norm(gather_option_states(hidden, option_pos))
        if answer_pos is None:
            pos = hidden.new_full(
                (hidden.size(0),), hidden.size(1) - 1, dtype=torch.long
            )
        else:
            pos = answer_pos.clamp(min=0, max=hidden.size(1) - 1)
        bidx = torch.arange(hidden.size(0), device=hidden.device)
        state = self.state_norm(hidden[bidx, pos])

        keys = self.key_proj(option_states).unflatten(
            -1, (self.relation_heads, self.relation_head_dim)
        )
        query = torch.tanh(self.query_proj(state)).unflatten(
            -1, (self.relation_heads, self.relation_head_dim)
        )
        per_head = (
            (keys * query.unsqueeze(1)).sum(dim=-1)
            / math.sqrt(float(self.relation_head_dim))
        )
        relation = per_head.sum(dim=-1) / math.sqrt(float(self.relation_heads))
        scale = self.logit_scale.clamp(math.log(0.1), math.log(20.0)).exp()

        with torch.no_grad():
            self._last_diagnostics = {
                "relation_spread": float(
                    relation.std(dim=-1, unbiased=False).mean().item()
                ),
                "score_scale": float(scale.item()),
            }
        return scale * relation

    def training_diagnostics(self) -> dict[str, float]:
        out = dict(self._last_diagnostics)
        key_grad = self.key_proj.weight.grad
        query_grad = self.query_proj.weight.grad
        grad_sq = 0.0
        if key_grad is not None:
            grad_sq += float(key_grad.float().square().sum().item())
        if query_grad is not None:
            grad_sq += float(query_grad.float().square().sum().item())
        out["relation_grad"] = math.sqrt(grad_sq)
        return out


class CompactKQHFRankModel(HFRankModel):
    """Frozen backbone plus the compact K/Q head."""

    def __init__(
        self,
        model_name: str,
        freeze_backbone: bool = True,
        dtype: torch.dtype = torch.float16,
        with_lm_head: bool = False,
        relation_heads: int = 16,
        relation_head_dim: int = 4,
    ):
        super().__init__(
            model_name,
            freeze_backbone=freeze_backbone,
            dtype=dtype,
            with_lm_head=with_lm_head,
            head_kind="linear",
        )
        hidden_size = self.head.proj.in_features
        self.head = CompactKQRankHead(
            hidden_size,
            relation_heads=relation_heads,
            relation_head_dim=relation_head_dim,
        )
        self.head_kind = "compact-kq"
        self.head_rank = relation_heads * relation_head_dim
        self.relation_heads = relation_heads
        self.relation_head_dim = relation_head_dim
