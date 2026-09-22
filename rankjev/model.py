"""Prefill-only listwise score heads.

The current readout is a scalar bilinear potential over option and answer
states. LCS does not live here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def gather_option_states(hidden: torch.Tensor, option_pos: torch.Tensor) -> torch.Tensor:
    pos = option_pos.clamp(min=0)
    bidx = torch.arange(hidden.size(0), device=hidden.device).unsqueeze(1).expand_as(pos)
    return hidden[bidx, pos]


class ScoreHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, option_pos: torch.Tensor, answer_pos=None) -> torch.Tensor:
        states = gather_option_states(hidden, option_pos)
        return self.proj(states).squeeze(-1)


class BilinearExplainHead(nn.Module):
    """Bilinear latent pairing, then a 1-form that explains pairwise order.

    State s (Answer: token) is mapped with a full H×H matrix so every
    coordinate of the option state can interact with every coordinate of s:

        q = W s
        h'_k = LN(h_k + h_k ⊙ q)

    h' stays in R^H. The explanation layer is a linear differential:

        z_k = a · h'_k
        z_i - z_j = a · (h'_i - h'_j)

    Pairwise loss reads that first-order gap; it is a partial order, not a
    second MLP collapse.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.state_map = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.explain = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.state_map.weight)
        nn.init.zeros_(self.state_map.bias)

    def paired_states(self, hidden: torch.Tensor, option_pos: torch.Tensor, answer_pos=None) -> torch.Tensor:
        opts = gather_option_states(hidden, option_pos)
        if answer_pos is None:
            pos = hidden.new_full((hidden.size(0),), hidden.size(1) - 1, dtype=torch.long)
        else:
            pos = answer_pos.clamp(min=0, max=hidden.size(1) - 1)
        bidx = torch.arange(hidden.size(0), device=hidden.device)
        state = hidden[bidx, pos]
        query = self.state_map(state).unsqueeze(1)
        return self.norm(opts + opts * query)

    def forward(self, hidden: torch.Tensor, option_pos: torch.Tensor, answer_pos=None) -> torch.Tensor:
        paired = self.paired_states(hidden, option_pos, answer_pos)
        return self.explain(paired).squeeze(-1)


class BilinearPotentialHead(nn.Module):
    """Explicit scalar potential: linear readout + full bilinear interaction.

        u_k = LN(h_k), q = LN(h_answer)
        z_k = w·u_k + u_k^T A q / sqrt(H)

    A starts at zero, so optimization begins as a stable linear probe. Score
    differences are an exact, transitive potential rather than independently
    predicted pairwise preferences.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.option_norm = nn.LayerNorm(hidden_size)
        self.state_norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.state_map = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size**-0.5
        nn.init.zeros_(self.state_map.weight)

    def forward(self, hidden: torch.Tensor, option_pos: torch.Tensor, answer_pos=None) -> torch.Tensor:
        options = self.option_norm(gather_option_states(hidden, option_pos))
        if answer_pos is None:
            pos = hidden.new_full((hidden.size(0),), hidden.size(1) - 1, dtype=torch.long)
        else:
            pos = answer_pos.clamp(min=0, max=hidden.size(1) - 1)
        bidx = torch.arange(hidden.size(0), device=hidden.device)
        state = self.state_norm(hidden[bidx, pos])
        query = self.state_map(state)
        linear = self.linear(options).squeeze(-1)
        bilinear = (options * query.unsqueeze(1)).sum(dim=-1) * self.scale
        return linear + bilinear


def build_head(hidden_size: int, kind: str = "linear") -> nn.Module:
    if kind in {"bilinear-v2", "bilinear-potential"}:
        return BilinearPotentialHead(hidden_size)
    if kind in {"bilinear", "bilinear-explain"}:
        return BilinearExplainHead(hidden_size)
    if kind in {"linear", "pointer", ""}:
        return ScoreHead(hidden_size)
    raise ValueError(f"unknown head kind: {kind}")


class ToyRankModel(nn.Module):
    """Byte-level encoder so the loop can learn on a laptop without a 0.5B load."""

    vocab_size = 258  # 0-255 bytes + BOS + OPT

    def __init__(self, d_model: int = 192, nhead: int = 6, num_layers: int = 3, max_len: int = 512):
        super().__init__()
        self.emb = nn.Embedding(self.vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = build_head(d_model, "linear")
        # MPS lacks the nested-tensor fastpath used when a padding mask is present.
        try:
            torch.backends.mha.set_fastpath_enabled(False)
        except Exception:
            pass

    def forward(self, input_ids, attention_mask, option_pos, **_kw):
        t = input_ids.size(1)
        pos_ids = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.emb(input_ids) + self.pos(pos_ids)
        key_pad = attention_mask == 0
        h = self.enc(x, src_key_padding_mask=key_pad)
        return self.head(h, option_pos, answer_pos=_kw.get("answer_pos"))


def _hidden_size(config) -> int:
    if getattr(config, "hidden_size", None):
        return int(config.hidden_size)
    text = getattr(config, "text_config", None)
    if text is not None and getattr(text, "hidden_size", None):
        return int(text.hidden_size)
    raise ValueError("cannot read hidden_size from model config")


def _load_backbone(model_name: str, dtype: torch.dtype, with_lm_head: bool = False):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    from . import _silence_kernel_fallback

    _silence_kernel_fallback()
    kw = dict(trust_remote_code=True, torch_dtype=dtype)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # Qwen3 GQA (e.g. 40q / 8kv) aborts in mps.matmul; eager repeats KV first.
    n_q = int(getattr(config, "num_attention_heads", 1) or 1)
    n_kv = int(getattr(config, "num_key_value_heads", n_q) or n_q)
    if getattr(config, "model_type", "") == "qwen3" and n_q != n_kv:
        kw["attn_implementation"] = "eager"
    lm_weight = None
    if with_lm_head:
        backbone = AutoModelForCausalLM.from_pretrained(model_name, config=config, **kw)
        out = backbone.get_output_embeddings()
        if out is not None and getattr(out, "weight", None) is not None:
            lm_weight = out.weight
    else:
        try:
            backbone = AutoModel.from_pretrained(model_name, config=config, **kw)
        except Exception:
            backbone = AutoModelForCausalLM.from_pretrained(model_name, config=config, **kw)
    return backbone, _hidden_size(config), lm_weight


def _freeze_output_embeddings(backbone) -> None:
    getter = getattr(backbone, "get_output_embeddings", None)
    if getter is None:
        return
    out = getter()
    if out is None:
        return
    for p in out.parameters():
        p.requires_grad = False


def _enable_checkpointing(backbone) -> None:
    for obj in (backbone, getattr(backbone, "model", None)):
        if obj is None:
            continue
        if hasattr(obj, "gradient_checkpointing_enable"):
            try:
                obj.gradient_checkpointing_enable()
            except TypeError:
                obj.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        cfg = getattr(obj, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False


class HFRankModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_backbone: bool = True,
        dtype: torch.dtype = torch.float16,
        with_lm_head: bool = False,
        head_kind: str = "linear",
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.head_kind = head_kind
        self.backbone, hidden, lm_weight = _load_backbone(model_name, dtype, with_lm_head=with_lm_head)
        self.head = build_head(hidden, head_kind)
        self.lm_weight = lm_weight
        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
            _freeze_output_embeddings(self.backbone)
            _enable_checkpointing(self.backbone)

    def _encode(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        if getattr(out, "last_hidden_state", None) is not None:
            return out.last_hidden_state
        return out.hidden_states[-1]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def vocab_scores(self, hidden: torch.Tensor, answer_pos: torch.Tensor, letter_ids: torch.Tensor) -> torch.Tensor:
        """z_k = h_answer · W_lm[id_letter_k]. Frozen teacher, no decode."""
        if self.lm_weight is None:
            raise RuntimeError("HFRankModel was built without lm_head")
        pos = answer_pos.clamp(min=0, max=hidden.size(1) - 1)
        bidx = torch.arange(hidden.size(0), device=hidden.device)
        h = hidden[bidx, pos].float()
        emb = self.lm_weight.float()[letter_ids.clamp(min=0)]
        return (emb * h.unsqueeze(1)).sum(-1)

    def forward(
        self,
        input_ids,
        attention_mask,
        option_pos,
        answer_pos=None,
        letter_ids=None,
        return_vocab: bool = False,
    ):
        if any(p.requires_grad for p in self.backbone.parameters()):
            h = self._encode(input_ids, attention_mask)
        else:
            with torch.no_grad():
                h = self._encode(input_ids, attention_mask)
        scores = self.head(h.float(), option_pos, answer_pos=answer_pos)
        if return_vocab and self.lm_weight is not None and answer_pos is not None and letter_ids is not None:
            return scores, self.vocab_scores(h, answer_pos, letter_ids)
        return scores


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
