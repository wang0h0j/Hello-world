# rank-kq

列表排序的紧致读出。冻结 Qwen3.5-9B，只训练一个分组 K/Q 头。底座参数不更新。

每个选项映射到 K，题目状态映射到 Q。二者在 16 个通道、每通道 4 维的空间里做内积，再合成一个标量势。标量的高低就是该选项在本题中的排序。隐藏维为 4096 时，可训练参数为 540,673。

```text
K_i = W_k LayerNorm(h_i)
Q   = tanh(W_q LayerNorm(h_state))
z_i = scale · Σ_j ⟨K_ij, Q_j⟩ / √(4 · 16)
```

K 保留幅度。Q 经过 `tanh`，取值有界。

## 监督

训练只比较顺序，不拟合原始分数的绝对大小。

| 金标 | 损失 |
|---|---|
| 至少三个不同档位 | 金标前 10 名上的 Pearson `1 − IC`，金标先换成并列中位名次 |
| 只有两个档位 | RankNet：赢家相对每个输家的 logistic 损失 |

最长公共子序列只用于评测，不进入梯度。置信度由分数分布导出，没有单独的监督项。

选项在每个训练样本上连同金标一起重排，避免模型记住位置。

## 使用

```bash
pip install -r requirements.txt
export RANKJEV_MODEL=/path/to/Qwen3.5-9B   # 或 Qwen/Qwen3.5-9B

python -m rankjev.check
python -m rankjev.fit
python -m rankjev.score \
  --ckpt runs/kq-h16d4.pt \
  --data data/listwise/dev.jsonl \
  --limit 300 \
  --show 0
```

默认数据路径：

- `data/listwise/train.jsonl`
- `data/listwise/dev.jsonl`

样本正文不包含在本仓库中。字段、来源和切分见 [DATA.md](DATA.md)，规模见 `data/listwise/manifest.json`。检查点写入 `runs/`，同样不入库。

## 布局

| 路径 | 职责 |
|---|---|
| `rankjev/head.py` | K/Q 读出 |
| `rankjev/fit.py` | 训练入口 |
| `rankjev/score.py` | 验证入口 |
| `rankjev/check.py` | 参数量、置换等变性和梯度检查 |
| `rankjev/train.py` | 优化、学习率、日志和检查点 |
| `rankjev/losses.py` | 排序损失 |
| `rankjev/metrics.py` | top-1、Spearman、NDCG、LCS |
| `rankjev/data.py` | jsonl 读取和列表编码 |
| `rankjev/model.py` | 冻结底座的前向 |
| `rankjev/lcs.py` | 评测用的序列对齐 |
