# rank-kq

在冻结的预训练模型上训练一个列表排序头。底座权重固定，只更新头的参数。底座通过 `RANKJEV_MODEL` 指定，可以是本机目录或 Hugging Face 模型名。头的宽度跟随底座的隐藏维。隐藏维为 4096 时，头的参数量是 540,673。

头为每个选项和当前题目各算一组 $16 \times 4$ 的向量，分组做内积后相加，得到该选项的分数。分数高的选项排在前面。

$$
\begin{aligned}
K_i &= W_k \operatorname{LayerNorm}(h_i) \\
Q &= \tanh(W_q \operatorname{LayerNorm}(h_{\mathrm{state}})) \\
z_i &= s \cdot \frac{\sum_j K_{ij}^{\top} Q_j}{\sqrt{4 \times 16}}
\end{aligned}
$$

$W_k$ 与 $W_q$ 是两个独立的线性层。$h_i$ 是第 $i$ 个选项末尾的隐藏状态，$h_{\mathrm{state}}$ 是 `Answer:` 位置的隐藏状态。$\tanh$ 只作用在题目一侧。$s$ 是一个可学习的正标量。

## 损失

目标只包含顺序。

| 条件 | 损失 |
|---|---|
| `gold` 中不同数值不少于 3 个 | 取目标分数最高的前 10 项，计算预测分数与目标名次的 Pearson 相关 $\rho$，损失为 $1-\rho$。并列目标使用中位名次 |
| `gold` 中不同数值不超过 2 个 | 对每一对更高分与更低分计算 $\operatorname{softplus}(-(z_{\mathrm{high}}-z_{\mathrm{low}}))$ |

最长公共子序列只在验证时计算。置信度由当次分数的 $\operatorname{softmax}$ 算出，不单独作为损失。

每个训练样本会把 `options` 和 `gold` 用同一个随机置换打乱。

## 命令

```bash
pip install -r requirements.txt
export RANKJEV_MODEL=/path/to/model   # 或 Hugging Face 模型名

python -m rankjev.check
python -m rankjev.fit
python -m rankjev.score \
  --ckpt runs/kq-h16d4.pt \
  --data data/listwise/dev.jsonl \
  --limit 300 \
  --show 0
```

默认读取：

- `data/listwise/train.jsonl`
- `data/listwise/dev.jsonl`

样本正文不在仓库内。格式与构建规则见 [DATA.md](DATA.md)，条数见 `data/listwise/manifest.json`。权重写入 `runs/`，不纳入版本库。

## 文件

| 路径 | 内容 |
|---|---|
| `rankjev/head.py` | 排序头 |
| `rankjev/fit.py` | 训练入口 |
| `rankjev/score.py` | 验证入口 |
| `rankjev/check.py` | 参数量、选项置换和梯度检查 |
| `rankjev/train.py` | 优化器、学习率、日志、检查点 |
| `rankjev/losses.py` | 损失函数 |
| `rankjev/metrics.py` | top-1、Spearman、NDCG、LCS |
| `rankjev/data.py` | jsonl 与批量编码 |
| `rankjev/model.py` | 冻结模型的前向 |
| `rankjev/lcs.py` | 验证用的序列对齐 |
