"""Rank a short list with the published Qwen3.5-9B K/Q head.

  python -m rankjev.use --demo
  python -m rankjev.use --json examples/use/answers.json
  python -m rankjev.use --stem "Rank the replies." --option "first" --option "second"

The checkpoint is trained for Qwen/Qwen3.5-9B. Another backbone will not
match these weights. Scores order the options; they are not calibrated
probabilities. Each call runs the full frozen backbone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import RankCollator, RankSample
from .metrics import derived_confidence
from .score import load_kq_model
from .train import pick_device

DEFAULT_CKPT = "weights/qwen35-9b-kq-head.pt"

DEMOS = [
    {
        "id": "answers",
        "stem": (
            "Rank the candidate answers to the user. "
            "Higher score = a better response.\n\n"
            "How do I sort a list of numbers in Python?"
        ),
        "options": [
            "Use sorted(numbers) for a new list, or numbers.sort() to sort in place.",
            "I am not sure, maybe try clicking the list.",
            "sorted(numbers) returns a new sorted list. numbers.sort() sorts the original list and returns None.",
            "Numbers cannot be sorted in Python.",
        ],
    },
    {
        "id": "stars",
        "stem": (
            "How many stars did this reviewer give?\n\n"
            "The food was cold, the order was wrong, and nobody came back to fix it."
        ),
        "options": [
            "1 star: terrible experience",
            "2 stars: poor",
            "3 stars: average",
            "4 stars: good",
            "5 stars: excellent",
        ],
    },
    {
        "id": "tools",
        "stem": (
            "Rank these read-only calls for the request. "
            "Higher score = should be tried first.\n\n"
            "Read the last 20 daily closes for stock 600036. Do not write anything."
        ),
        "options": [
            "server=sse name=quote_dayk effect=read\nargs: stockCode=600036 begin=-20 end=-1 period=day select=date,close",
            "server=sse name=quote_dayk effect=read\nargs: stockCode=600036 begin=-20 end=-1 period=week select=date,close",
            "server=sse name=quote_mink effect=read\nargs: stockCode=600036 begin=-20 end=-1",
            "server=db name=query_rows effect=query\nargs: table=quotes code=600036",
            "server=db name=insert_rows effect=append\nargs: table=quotes",
        ],
    },
]


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = [payload]
    cases = []
    for i, row in enumerate(payload):
        stem = row.get("stem")
        options = row.get("options")
        if not isinstance(stem, str) or not isinstance(options, list):
            raise ValueError(f"case {i} needs string stem and list options")
        cases.append({"id": row.get("id") or f"case-{i}", "stem": stem, "options": options})
    return cases


def rank_case(model, collate, device, case: dict) -> dict:
    options = [str(x) for x in case["options"]]
    if not 2 <= len(options) <= 16:
        raise ValueError(f"{case['id']}: need 2 to 16 options, got {len(options)}")
    if len(set(options)) != len(options):
        raise ValueError(f"{case['id']}: options must be unique")
    batch = collate([RankSample(case["stem"], options, [0.0] * len(options))])
    gpu = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    scores = model(
        gpu["input_ids"],
        gpu["attention_mask"],
        gpu["option_pos"],
        answer_pos=gpu.get("answer_pos"),
    )
    logits = scores[0, : len(options)].float()
    probs = torch.softmax(logits, dim=0)
    order = sorted(range(len(options)), key=lambda i: (-float(logits[i]), i))
    confidence = float(derived_confidence(scores, gpu["padding_mask"])[0].item())
    ranking = []
    for rank, index in enumerate(order, 1):
        ranking.append(
            {
                "rank": rank,
                "index": index,
                "score": round(float(logits[index]), 4),
                "p": round(float(probs[index]), 4),
                "option": options[index],
            }
        )
    return {
        "id": case["id"],
        "top": options[order[0]],
        "confidence": round(confidence, 4),
        "ranking": ranking,
    }


def print_result(result: dict) -> None:
    print(f"\n{result['id']}  top={result['ranking'][0]['index']}  conf={result['confidence']}")
    for row in result["ranking"]:
        text = row["option"].replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        print(f"  {row['rank']}. [{row['index']}] {row['score']:8.3f}  p={row['p']:.3f}  {text}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Rank options with the Qwen3.5-9B K/Q head")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--model", default="", help="local Qwen3.5-9B directory; default is the checkpoint Hub id")
    p.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    p.add_argument("--max-len", type=int, default=0, help="0 = 768")
    p.add_argument("--json", dest="json_path", default="", help="one case or a list of cases")
    p.add_argument("--demo", action="store_true", help="rank the three built-in examples")
    p.add_argument("--stem", default="")
    p.add_argument("--option", action="append", default=[])
    args = p.parse_args(argv)

    cases = []
    if args.demo:
        cases.extend(DEMOS)
    if args.json_path:
        cases.extend(load_cases(Path(args.json_path)))
    if args.stem or args.option:
        cases.append({"id": "cli", "stem": args.stem, "options": args.option})
    if not cases:
        p.error("pass --demo, --json, or --stem with at least two --option")

    device = pick_device(args.device)
    model, ckpt = load_kq_model(Path(args.ckpt), device, args.model)
    max_len = args.max_len or int((ckpt.get("args") or {}).get("max_len") or 768)
    from transformers import AutoTokenizer

    tok_src = args.model or ckpt.get("base") or ckpt.get("args", {}).get("model")
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate = RankCollator(backend="hf", tokenizer=tokenizer, max_len=max_len)
    print(f"device={device} max_len={max_len} cases={len(cases)}", flush=True)

    results = []
    with torch.no_grad():
        for case in cases:
            result = rank_case(model, collate, device, case)
            results.append(result)
            print_result(result)
    if len(results) > 1:
        print("\n" + json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
