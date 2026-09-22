"""Checks for the compact K/Q head."""

from __future__ import annotations

import torch

from .head import CompactKQRankHead


def main() -> None:
    head = CompactKQRankHead(4096, relation_heads=16, relation_head_dim=4)
    n = sum(p.numel() for p in head.parameters())
    assert n == 540_673, n

    small = CompactKQRankHead(64, relation_heads=4, relation_head_dim=4)
    hidden = torch.randn(1, 6, 64)
    option_pos = torch.tensor([[1, 3, 4]])
    scores = small(hidden, option_pos, answer_pos=torch.tensor([5]))
    assert scores.shape == (1, 3)
    assert torch.isfinite(scores).all()

    perm = torch.tensor([[4, 1, 3]])
    permuted = small(hidden, perm, answer_pos=torch.tensor([5]))
    assert torch.allclose(permuted, scores[:, [2, 0, 1]], atol=1e-6)

    scores.square().sum().backward()
    assert small.key_proj.weight.grad is not None
    assert small.query_proj.weight.grad is not None
    assert torch.isfinite(small.key_proj.weight.grad).all()
    assert torch.isfinite(small.query_proj.weight.grad).all()
    diag = small.training_diagnostics()
    assert 0.1 <= diag["score_scale"] <= 20.0
    assert diag["relation_spread"] >= 0.0
    assert diag["relation_grad"] >= 0.0
    assert all(torch.isfinite(torch.tensor(v)) for v in diag.values())
    print(
        f"ok  compact_kq_16x4_params={n:,}  scores={tuple(scores.shape)}"
    )


if __name__ == "__main__":
    main()
