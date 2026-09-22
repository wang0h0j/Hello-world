"""Post-train a listwise ranker.

  python -m rankjev.train --smoke
  python -m rankjev.train --model Qwen/Qwen2.5-0.5B --steps 400
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader

from .data import JsonlRankDataset, RankCollator, SyntheticRankDataset
from .losses import rank_loss
from .metrics import eval_batch
from .model import HFRankModel, ToyRankModel, trainable_params


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def amp_dtype(device: torch.device, unfreeze: bool = False) -> torch.dtype:
    # Frozen-head MPS is fine in fp16. Full-rank fp16 overflows; prefer bf16.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "mps" and unfreeze:
        try:
            x = torch.zeros(2, device=device, dtype=torch.bfloat16)
            _ = (x + 1).sum()
            return torch.bfloat16
        except Exception:
            return torch.float16
    if device.type in ("mps", "cuda"):
        return torch.float16
    return torch.float32


def _make_scaler(device: torch.device, dtype: torch.dtype):
    # MPS GradScaler is unreliable and produces Inf grads on full-rank.
    if dtype != torch.float16 or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return None


def _finite_grads(model) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def build_optimizer(groups, name: str, weight_decay: float):
    """Full-rank 4B + fp32 AdamW needs ~29G just for moments. 24G cards cannot.

    adamw8bit ≈ 4–6G extra; adafactor ≈ 0.5G; adamw ≈ 29G.
    """
    key = (name or "adamw").lower()
    params = groups or []
    if key in {"adamw8bit", "8bit", "bnb"}:
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, weight_decay=weight_decay), "adamw8bit"
        except Exception as exc:
            print(f"AdamW8bit unavailable ({exc}); falling back to Adafactor", flush=True)
            key = "adafactor"
    if key == "adafactor":
        from transformers.optimization import Adafactor

        lr0 = params[0]["lr"] if params and isinstance(params[0], dict) else None
        return (
            Adafactor(
                params,
                lr=lr0,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=weight_decay,
            ),
            "adafactor",
        )
    return torch.optim.AdamW(params, weight_decay=weight_decay), "adamw"


def build_model(
    name: str,
    freeze_backbone: bool,
    dtype: torch.dtype,
    with_lm_head: bool = False,
    head_kind: str = "linear",
):
    if name == "toy":
        return ToyRankModel()
    return HFRankModel(
        name,
        freeze_backbone=freeze_backbone,
        dtype=dtype,
        with_lm_head=with_lm_head,
        head_kind=head_kind,
    )


def resolve_resume(spec: str, save: str) -> Path:
    if spec not in ("auto", "1", "true"):
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(f"resume ckpt not found: {path}")
        return path
    dest = Path(save)
    cands = [p for p in [dest, *dest.parent.glob(dest.stem + "-step*.pt")] if p.exists()]
    if not cands:
        raise FileNotFoundError(f"no checkpoint to resume under {dest.parent / dest.stem}")
    best, best_step = cands[0], -1
    for path in cands:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        st = int(blob.get("step") or 0)
        if st >= best_step:
            best, best_step = path, st
    return best


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = amp_dtype(device, unfreeze=args.unfreeze)
    print(f"device={device} dtype={dtype} model={args.model}")

    tokenizer = None
    backend = "toy"
    if args.model != "toy":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        backend = "hf"

    if args.data:
        train_ds = JsonlRankDataset(args.data)
        val_ds = JsonlRankDataset(args.val_data) if args.val_data else SyntheticRankDataset(n=args.n_val, seed=10_000)
    else:
        train_ds = SyntheticRankDataset(n=args.n_train, seed=0)
        val_ds = SyntheticRankDataset(n=args.n_val, seed=10_000)
    train_collate = RankCollator(
        backend=backend,
        tokenizer=tokenizer,
        max_len=args.max_len,
        shuffle_options=args.shuffle_options,
        seed=args.seed,
    )
    val_collate = RankCollator(
        backend=backend,
        tokenizer=tokenizer,
        max_len=args.max_len,
        shuffle_options=False,
        seed=args.seed,
    )
    loader_rng = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=train_collate,
        generator=loader_rng,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, collate_fn=val_collate
    )

    resume_ckpt = None
    resume_path = None
    if args.resume:
        resume_path = resolve_resume(args.resume, args.save)
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        step0 = int(resume_ckpt.get("step") or 0)
        print(f"resume {resume_path} step={step0}", flush=True)
        if step0 >= args.steps:
            print(f"already at step {step0} >= --steps {args.steps}; nothing to do", flush=True)
            return

    load_from = args.model
    if resume_ckpt and resume_ckpt.get("backbone_dir"):
        load_from = resume_ckpt["backbone_dir"]
        print(f"init backbone from {load_from}", flush=True)

    model = build_model(
        load_from,
        freeze_backbone=not args.unfreeze,
        dtype=dtype,
        with_lm_head=args.vocab_align > 0,
        head_kind=args.head,
    )
    model.to(device)
    if device.type == "mps":
        torch.mps.empty_cache()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
    n_train = trainable_params(model)
    print(
        f"trainable params: {n_train:,}  unfreeze={args.unfreeze} "
        f"head={args.head}",
        flush=True,
    )
    if args.unfreeze:
        print(
            "VRAM note: 4B bf16 weights+grads ≈ 16G; AdamW fp32 moments ≈ 29G "
            "(MPS 30G+ peak is mostly Adam). 24G needs --optim adamw8bit or adafactor.",
            flush=True,
        )

    head_lr = args.head_lr if args.head_lr else args.lr
    head_named = list(model.head.named_parameters()) if hasattr(model, "head") else []
    head_params = [p for _, p in head_named]
    head_ids = {id(p) for p in head_params}
    body_named = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad and id(p) not in head_ids
    ]
    groups = []

    def add_groups(named_params, lr):
        decay = [p for _, p in named_params if p.requires_grad and p.ndim >= 2]
        no_decay = [p for _, p in named_params if p.requires_grad and p.ndim < 2]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": 0.01})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

    add_groups(body_named, args.lr)
    add_groups(head_named, head_lr)
    opt, opt_name = build_optimizer(
        groups or [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}],
        args.optim,
        0.01,
    )
    if args.unfreeze and opt_name == "adamw":
        print("warning: full-rank AdamW will not fit 24G; pass --optim adamw8bit", flush=True)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = _make_scaler(device, dtype)
    print(
        f"optim={opt_name}  lr body={args.lr} head={head_lr} "
        f"warmup={args.warmup} schedule={args.lr_schedule} "
        f"scaler={scaler is not None} "
        f"order_only={args.order_only} topk={args.topk} "
        f"shuffle_options={args.shuffle_options} seed={args.seed}",
        flush=True,
    )

    def save_ckpt(path: Path, val=None, step=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "head": model.head.state_dict() if hasattr(model, "head") else model.state_dict(),
            "base": args.model,
            "freeze_backbone": not args.unfreeze,
            "args": vars(args),
            "step": step,
            "val": val,
        }
        if args.save_optim:
            payload["optim"] = opt.state_dict()
        if args.unfreeze and args.model != "toy" and hasattr(model, "backbone"):
            bdir = path.with_name(path.stem + "-backbone")
            model.backbone.save_pretrained(bdir)
            payload["backbone_dir"] = str(bdir.resolve())
        torch.save(payload, path)
        extra = f"  backbone={payload.get('backbone_dir', 'frozen '+args.model)}"
        print(f"saved {path}{extra}", flush=True)

    step = 0
    if resume_ckpt is not None:
        if hasattr(model, "head") and resume_ckpt.get("head"):
            model.head.load_state_dict(resume_ckpt["head"])
        elif resume_ckpt.get("model"):
            model.load_state_dict(resume_ckpt["model"], strict=False)
        if resume_ckpt.get("optim") and args.save_optim:
            try:
                opt.load_state_dict(resume_ckpt["optim"])
            except ValueError as exc:
                print(f"optimizer layout changed; starting fresh optimizer ({exc})", flush=True)
        step = int(resume_ckpt.get("step") or 0)
    model.train()
    nan_streak = 0
    window = {
        k: 0.0
        for k in (
            "loss", "sp10", "pair", "ic", "top1", "n_graded", "n_choice",
        )
    }
    window_n = 0
    while step < args.steps:
        for batch in train_loader:
            if args.warmup > 0 and step < args.warmup:
                lr_scale = (step + 1) / args.warmup
            elif args.lr_schedule == "cosine":
                span = max(args.steps - args.warmup, 1)
                progress = min(max((step - args.warmup) / span, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr_scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
            else:
                lr_scale = 1.0
            for group, base in zip(opt.param_groups, base_lrs):
                group["lr"] = base * lr_scale
            batch = {k: v.to(device) for k, v in batch.items()}
            if args.vocab_align > 0:
                scores, z_vocab = model(
                    batch["input_ids"], batch["attention_mask"], batch["option_pos"],
                    answer_pos=batch["answer_pos"], letter_ids=batch["letter_ids"],
                    return_vocab=True,
                )
            else:
                scores = model(
                    batch["input_ids"], batch["attention_mask"], batch["option_pos"],
                    answer_pos=batch.get("answer_pos"),
                )
                z_vocab = None
            if not torch.isfinite(scores).all():
                nan_streak += 1
                print(f"skip non-finite scores  streak={nan_streak}", flush=True)
                opt.zero_grad(set_to_none=True)
                if nan_streak >= 5:
                    raise RuntimeError("5 consecutive NaN forwards; restart, do not --resume this run")
                continue
            parts = rank_loss(
                scores, batch["gold"], batch["padding_mask"],
                pairwise_weight=args.pairwise,
                use_soft_spearman=args.soft_spearman,
                y_vocab=z_vocab,
                vocab_weight=args.vocab_align,
                ic_weight=args.ic_weight,
                gap_weight=args.gap_weight,
                topk=args.topk,
                topk_weight=args.topk_weight,
                top1_weight=args.top1_weight,
                order_only=args.order_only,
            )
            loss = parts["total"]
            if not torch.isfinite(loss):
                nan_streak += 1
                print(f"skip non-finite loss  streak={nan_streak}", flush=True)
                opt.zero_grad(set_to_none=True)
                if nan_streak >= 5:
                    raise RuntimeError("5 consecutive NaN losses; restart, do not --resume this run")
                continue
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            clip = 0.5 if args.unfreeze else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if not _finite_grads(model):
                nan_streak += 1
                print(f"skip non-finite grads  streak={nan_streak}", flush=True)
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                if nan_streak >= 5:
                    raise RuntimeError("5 consecutive NaN grads; restart, do not --resume this run")
                continue
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            nan_streak = 0
            step += 1
            with torch.no_grad():
                masked_scores = scores.detach().masked_fill(batch["padding_mask"], -1e9)
                masked_gold = batch["gold"].masked_fill(batch["padding_mask"], -1e9)
                batch_top1 = (
                    masked_scores.argmax(dim=-1) == masked_gold.argmax(dim=-1)
                ).float().mean().item()
            window["loss"] += float(loss.item())
            ng = int(parts.get("n_graded") or 0)
            nc = int(parts.get("n_choice") or 0)
            if ng:
                window["sp10"] += float(parts["sp10"].item()) * ng
                window["n_graded"] += ng
            if nc:
                window["pair"] += float(parts["pairwise"].item()) * nc
                window["n_choice"] += nc
            if not args.order_only:
                window["sp10"] += float(parts["sp10"].item())
                window["pair"] += float(parts["pairwise"].item())
            window["ic"] += float(parts["ic"].item())
            window["top1"] += batch_top1
            window_n += 1
            if step % args.log_every == 0:
                n = max(window_n, 1)
                fields = [
                    f"step {step:05d}",
                    f"loss={window['loss']/n:.4f}",
                    f"top1={window['top1']/n:.3f}",
                ]
                if args.order_only:
                    if window["n_graded"] > 0:
                        rho = 1.0 - window["sp10"] / window["n_graded"]
                        fields.append(f"rho={rho:.3f}")
                    if window["n_choice"] > 0:
                        pair = window["pair"] / window["n_choice"]
                        fields.append(f"pair={pair:.4f}")
                else:
                    fields.append(f"rho={1.0-window['ic']/n:.3f}")
                    if args.pairwise:
                        fields.append(f"pair={window['pair']/n:.4f}")
                diagnostic_fn = getattr(getattr(model, "head", None), "training_diagnostics", None)
                if callable(diagnostic_fn):
                    diag = diagnostic_fn()
                    if "ctx_ratio" in diag:
                        fields.extend(
                            (
                                f"ctx={diag['ctx_ratio']:.3f}",
                                f"cos={diag['direction_cos']:.3f}",
                                f"scale={diag['score_scale']:.3f}",
                                f"g_up={diag['up_grad']:.2e}",
                            )
                        )
                    elif "relation_spread" in diag:
                        fields.extend(
                            (
                                f"spread={diag['relation_spread']:.3f}",
                                f"scale={diag['score_scale']:.3f}",
                                f"g_kq={diag['relation_grad']:.2e}",
                            )
                        )
                print("  ".join(fields), flush=True)
                window = {k: 0.0 for k in window}
                window_n = 0
            if args.save_every and step % args.save_every == 0:
                mid = Path(args.save).with_name(Path(args.save).stem + f"-step{step}.pt")
                save_ckpt(mid, step=step)
            if step >= args.steps:
                break

    save_ckpt(Path(args.save), step=step)
    if args.skip_val:
        print("skip val", flush=True)
        return
    model.eval()
    acc = {
        k: 0.0
        for k in (
            "top1", "hit3", "mrr", "ndcg", "spearman", "lcs",
            "confidence", "n", "lcs_n",
        )
    }
    n_val = len(val_loader)
    print(f"validating {n_val} batches (this is slow on MPS; not stuck)", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(val_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            scores = model(
                batch["input_ids"], batch["attention_mask"], batch["option_pos"],
                answer_pos=batch.get("answer_pos"),
            )
            ev = eval_batch(scores, batch["gold"], batch["padding_mask"])
            w = ev["n"]
            for k in ("top1", "hit3", "mrr", "ndcg", "spearman", "confidence"):
                acc[k] += ev[k] * w
            acc["lcs"] += ev["lcs"] * ev["lcs_n"]
            acc["lcs_n"] += ev["lcs_n"]
            acc["n"] += w
            if i == 1 or i % 20 == 0 or i == n_val:
                print(f"val-progress {i}/{n_val}", flush=True)
    n = max(acc["n"], 1)
    val = {
        k: acc[k] / n
        for k in ("top1", "hit3", "mrr", "ndcg", "spearman", "confidence")
    }
    val["lcs"] = acc["lcs"] / max(acc["lcs_n"], 1)
    print(
        f"val  top1={val['top1']:.3f}  mrr={val['mrr']:.3f}  "
        f"ndcg={val['ndcg']:.3f}  spearman={val['spearman']:.3f}  "
        f"lcs={val['lcs']:.3f}",
        flush=True,
    )
    save_ckpt(Path(args.save), val=val, step=step)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jev-style rank post-training")
    p.add_argument("--model", default="toy", help="toy | HF model name")
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    p.add_argument("--data", default="", help="train.jsonl; empty = on-the-fly synthetic")
    p.add_argument("--val-data", default="", help="dev.jsonl")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--head-lr", type=float, default=0.0, help="score-head lr; 0 = same as --lr")
    p.add_argument("--warmup", type=int, default=0, help="linear lr warmup steps")
    p.add_argument(
        "--lr-schedule",
        default="constant",
        choices=("constant", "cosine"),
        help="schedule after warmup",
    )
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="final/base lr ratio for cosine schedule",
    )
    p.add_argument(
        "--save-optim",
        action="store_true",
        help="also dump Adam states (huge when --unfreeze)",
    )
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument(
        "--shuffle-options",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="reshuffle options with their gold labels on every training sample",
    )
    p.add_argument("--pairwise", type=float, default=0.3)
    p.add_argument(
        "--ic-weight",
        type=float,
        default=1.0,
        help="weight on legacy raw-score 1-IC (outside --order-only)",
    )
    p.add_argument(
        "--gap-weight",
        type=float,
        default=0.0,
        help="optional Huber(Δz−Δy) / ListNet term (outside --order-only)",
    )
    p.add_argument("--topk", type=int, default=10, help="gold top-k slice for Spearman")
    p.add_argument(
        "--topk-weight",
        type=float,
        default=0.0,
        help="weight on tie-aware rank-target IC (outside --order-only)",
    )
    p.add_argument(
        "--top1-weight",
        type=float,
        default=0.0,
        help="weight on CE of the gold winner. unused when --order-only",
    )
    p.add_argument(
        "--order-only",
        action="store_true",
        help="graded: 1-IC(top-k); Choice/Noul: RankNet winner-vs-rest. No CE",
    )
    p.add_argument("--soft-spearman", action="store_true")
    p.add_argument("--unfreeze", action="store_true", help="train HF backbone, not just the head")
    p.add_argument(
        "--head",
        default="linear",
        choices=("linear", "bilinear", "bilinear-potential"),
        help="linear pointer, gated bilinear, or explicit bilinear potential",
    )
    p.add_argument(
        "--optim",
        default="adamw",
        choices=("adamw", "adamw8bit", "adafactor"),
        help="adamw8bit or adafactor when a full-rank AdamW state will not fit",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save", default="runs/last.pt")
    p.add_argument("--save-every", type=int, default=0, help="also snapshot every N steps")
    p.add_argument("--smoke", action="store_true", help="tiny CPU/MPS sanity run")
    p.add_argument("--skip-val", action="store_true", help="save and exit without the full val sweep")
    p.add_argument(
        "--resume",
        default="",
        help="ckpt path, or 'auto' to pick the newest --save / --save-every snapshot",
    )
    p.add_argument(
        "--vocab-align",
        type=float,
        default=0.0,
        help="weight of 1-IC between the head and frozen lm_head letter ranks",
    )
    args = p.parse_args(argv)
    if args.smoke:
        args.model = "toy"
        args.steps = 40
        args.batch = 8
        args.n_train = 128
        args.n_val = 32
        args.log_every = 5
        args.save = "runs/smoke.pt"
    return args


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
