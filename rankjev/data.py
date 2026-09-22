"""Listwise ranking samples.

Two synthetic tasks with a *full* gold order (not just one correct label):
  - order_restore: shuffled steps, recover chronological rank
  - numeric_mcq: distractors ranked by |error|
"""

from __future__ import annotations

import random
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


STEM_ORDER = "请按正确时间/逻辑顺序给下列选项打分，分数越高越应该排在前面。"
STEM_MCQ = "请给下列答案打分，越接近正确答案分数越高。"

SCRIPTS = [
    ["准备食材", "热锅", "倒油", "下菜", "翻炒", "调味", "出锅装盘"],
    ["打开编辑器", "写第一个测试", "实现函数", "跑通测试", "提交代码"],
    ["采集原始行情", "清洗缺失值", "构建特征", "切分训练验证", "拟合模型", "样本外回测"],
    ["提出假设", "设计实验", "收集数据", "分析结果", "写下结论"],
    ["用户打开页面", "填写表单", "点击提交", "服务端校验", "写入数据库", "返回成功"],
]


@dataclass
class RankSample:
    stem: str
    options: list[str]
    gold: list[float]  # higher = better


def _order_restore(rng: random.Random, min_k: int = 4, max_k: int = 7) -> RankSample:
    script = list(rng.choice(SCRIPTS))
    k = rng.randint(min_k, min(max_k, len(script)))
    steps = script[:k]
    indexed = list(enumerate(steps))  # (orig_i, text)
    rng.shuffle(indexed)
    options = [f"第{i + 1}步候选：{text}" for i, text in indexed]
    # earlier original step → higher gold
    gold = [float(k - 1 - orig_i) for orig_i, _ in indexed]
    return RankSample(STEM_ORDER, options, gold)


def _numeric_mcq(rng: random.Random, k: int = 5) -> RankSample:
    a, b = rng.randint(2, 40), rng.randint(2, 40)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        truth = a + b
        expr = f"{a} + {b}"
    elif op == "-":
        truth = a - b
        expr = f"{a} - {b}"
    else:
        truth = a * b
        expr = f"{a} × {b}"

    pool = {truth, truth + 1, truth - 1, truth + 2, a, b, abs(a - b), a + b + 1}
    if op != "*":
        pool.add(a * b)
    cands = [truth]
    for x in pool:
        if x != truth and x not in cands:
            cands.append(x)
        if len(cands) >= k:
            break
    while len(cands) < k:
        cands.append(truth + len(cands) * (1 if rng.random() < 0.5 else -1))

    indexed = list(cands[:k])
    rng.shuffle(indexed)
    options = [str(x) for x in indexed]
    # closer to truth is better; exact match uniquely highest
    gold = []
    for x in indexed:
        err = abs(x - truth)
        gold.append(100.0 if err == 0 else float(-err))
    return RankSample(f"{STEM_MCQ}\n{expr} = ?", options, gold)


class JsonlRankDataset(Dataset):
    """Materialized {stem, options, gold} jsonl."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> RankSample:
        row = self.rows[idx]
        return RankSample(row["stem"], list(row["options"]), [float(x) for x in row["gold"]])


class SyntheticRankDataset(Dataset):
    def __init__(self, n: int = 512, seed: int = 0, min_k: int = 4, max_k: int = 7):
        self.n = n
        self.seed = seed
        self.min_k = min_k
        self.max_k = max_k

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> RankSample:
        rng = random.Random(self.seed + idx)
        if rng.random() < 0.5:
            return _order_restore(rng, self.min_k, self.max_k)
        return _numeric_mcq(rng, k=rng.randint(self.min_k, self.max_k))


def encode_listwise_bytes(stem: str, options: list[str], max_len: int = 512):
    """Byte-level listwise encoding. Option position = last byte of that option."""
    bos = [256]
    opt_tok = [257]
    pieces = list(stem.encode("utf-8"))
    positions = []
    for opt in options:
        pieces.append(257)
        body = list(opt.encode("utf-8"))
        pieces.extend(body)
        positions.append(len(bos) + len(pieces) - 1)
    ids = bos + pieces
    if len(ids) > max_len:
        ids = ids[:max_len]
        positions = [min(p, max_len - 1) for p in positions]
    return ids, positions


def _letter_id(tokenizer, letter: str) -> int:
    pieces = tokenizer.encode(letter, add_special_tokens=False)
    return int(pieces[0]) if pieces else 0


def encode_listwise_hf(tokenizer, stem: str, options: list[str], max_len: int = 1024):
    """Returns ids, option_end_pos, letter_ids, answer_pos.

    Options are listed first so the model has seen them. `Answer:` is the
    pre-decode position whose lm_head logits over A/B/C... are the vocab ranking.
    """
    prefix = stem + "\n"
    ids = tokenizer.encode(prefix, add_special_tokens=True)
    positions = []
    letter_ids = []
    for i, opt in enumerate(options):
        label = chr(ord("A") + i)
        letter_ids.append(_letter_id(tokenizer, label))
        chunk = tokenizer.encode(f"{label}. {opt}\n", add_special_tokens=False)
        ids.extend(chunk)
        positions.append(len(ids) - 1)
    tail = tokenizer.encode("Answer:", add_special_tokens=False)
    ids.extend(tail if tail else tokenizer.encode(":", add_special_tokens=False))
    answer_pos = len(ids) - 1
    if len(ids) > max_len:
        ids = ids[:max_len]
        positions = [min(p, max_len - 1) for p in positions]
        answer_pos = min(answer_pos, max_len - 1)
    return ids, positions, letter_ids, answer_pos


class RankCollator:
    def __init__(
        self,
        backend: str = "toy",
        tokenizer=None,
        max_len: int = 512,
        shuffle_options: bool = False,
        seed: int = 20260921,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.shuffle_options = shuffle_options
        self.rng = random.Random(seed)

    def __call__(self, batch: list[RankSample]) -> dict[str, torch.Tensor]:
        enc_ids, enc_pos, enc_letters, enc_ans, golds = [], [], [], [], []
        for s in batch:
            options = list(s.options)
            gold = list(s.gold)
            if self.shuffle_options and len(options) > 1:
                order = list(range(len(options)))
                self.rng.shuffle(order)
                options = [options[i] for i in order]
                gold = [gold[i] for i in order]
            if self.backend == "hf":
                ids, pos, letters, ans = encode_listwise_hf(
                    self.tokenizer, s.stem, options, self.max_len
                )
            else:
                ids, pos = encode_listwise_bytes(s.stem, options, self.max_len)
                letters, ans = [0] * len(pos), max(len(ids) - 1, 0)
            enc_ids.append(ids)
            enc_pos.append(pos)
            enc_letters.append(letters)
            enc_ans.append(ans)
            golds.append(gold)

        max_t = max(len(x) for x in enc_ids)
        max_k = max(len(x) for x in enc_pos)
        B = len(batch)
        input_ids = torch.zeros(B, max_t, dtype=torch.long)
        attn = torch.zeros(B, max_t, dtype=torch.long)
        option_pos = torch.full((B, max_k), -1, dtype=torch.long)
        letter_ids = torch.zeros(B, max_k, dtype=torch.long)
        answer_pos = torch.zeros(B, dtype=torch.long)
        gold = torch.zeros(B, max_k, dtype=torch.float32)
        pad_mask = torch.ones(B, max_k, dtype=torch.bool)

        for i, (ids, pos, letters, ans, g) in enumerate(
            zip(enc_ids, enc_pos, enc_letters, enc_ans, golds)
        ):
            t = len(ids)
            k = len(pos)
            input_ids[i, :t] = torch.tensor(ids)
            attn[i, :t] = 1
            option_pos[i, :k] = torch.tensor(pos)
            letter_ids[i, :k] = torch.tensor(letters)
            answer_pos[i] = ans
            gold[i, :k] = torch.tensor(g, dtype=torch.float32)
            pad_mask[i, :k] = False

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "option_pos": option_pos,
            "letter_ids": letter_ids,
            "answer_pos": answer_pos,
            "gold": gold,
            "padding_mask": pad_mask,
        }
