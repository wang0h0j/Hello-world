# rank-kq

冻结 Qwen3.5-9B，只训练一个 `16×4` 的紧致 K/Q 排序头。底座不更新，不用 LoRA。

```text
K_i = W_k LN(h_i)
Q   = tanh(W_q LN(h_answer))
z_i = scale · Σ_j <K_ij, Q_j> / sqrt(4 × 16)
```

选项进 K，整道题的状态进 Q。交叉得到一个标量势，分数高低就是排序。隐藏维 4096 时，头大约 54 万参数。

损失只监督序：金标至少三档时，用前十名的 `1-IC`；只有赢家和输家时用 RankNet。LCS 只做评测，不进梯度。

## 运行

在本目录下执行：

```bash
pip install -r requirements.txt

export RANKJEV_MODEL=/path/to/Qwen3.5-9B

python -m rankjev.check
python -m rankjev.fit
python -m rankjev.score \
  --ckpt runs/kq-h16d4.pt \
  --data data/listwise/dev.jsonl \
  --limit 300 --show 0
```

`RANKJEV_MODEL` 也可以是 Hub 名 `Qwen/Qwen3.5-9B`。

训练读 `data/listwise/train.jsonl` 和 `data/listwise/dev.jsonl`。jsonl 不入库。编制规则见 [DATA.md](DATA.md)。检查点写到 `runs/`，同样不入库。

## 目录

| 文件 | 作用 |
|---|---|
| `rankjev/head.py` | K/Q 头 |
| `rankjev/fit.py` | 训练入口 |
| `rankjev/score.py` | 验证入口 |
| `rankjev/check.py` | 头的形状、置换和梯度检查 |
| `rankjev/train.py` `losses.py` `metrics.py` `data.py` `eval.py` `model.py` `lcs.py` | 训练循环和评测 |
| `data/listwise/manifest.json` | 语料规模，不含正文 |
