# rank-kq

[English](README.en.md)

## Qwen3.5-9B 权重

`weights/qwen35-9b-kq-head.pt` 是用冻结的 Qwen3.5-9B 训练一轮后的排序头。文件约 2.1 MB，只有头的参数、步数和 dev 结果，不含底座，也不含优化器状态。

| 项 | 值 |
|---|---|
| 底座 | `Qwen/Qwen3.5-9B`，隐藏维 4096 |
| 头 | 16 组，每组 4 维，540,673 个参数 |
| 训练步数 | 22,062，batch 1，一轮 |
| 最大长度 | 768 |
| dev 1,767 条 | top1 0.715，MRR 0.846，NDCG 0.917，Spearman 0.634，LCS 0.820 |

这组权重只配 Qwen3.5-9B。换成别的底座，隐藏维或分词对不上。仓库里的训练代码可以用于其他模型，但那是重新训练，不是加载这个文件。

底座大约 18 GB，需要自行准备。本机目录和 Hub 名都可以：

```bash
export RANKJEV_MODEL=/path/to/Qwen3.5-9B
# 或
export RANKJEV_MODEL=Qwen/Qwen3.5-9B
```

Apple Silicon 用 `mps` ，NVIDIA 用 `cuda` 。下面命令在仓库根目录执行。

### 安装

```bash
pip install -r requirements.txt
```

需要 `torch>=2.2` 和 `transformers>=5.7` ，否则加载不了 Qwen3.5。

### 给一组选项排序

输入是一个 JSON 对象，或由这种对象组成的数组：

```json
{
  "id": "answers",
  "stem": "Rank the candidate answers to the user. Higher score = a better response.\n\nHow do I sort a list of numbers in Python?",
  "options": [
    "Use sorted(numbers) for a new list, or numbers.sort() to sort in place.",
    "Numbers cannot be sorted in Python."
  ]
}
```

`stem` 是题干， `options` 是 2 到 16 条互不重复的文本。分数高的排在前面。

```bash
python -m rankjev.use --demo --model "$RANKJEV_MODEL"
python -m rankjev.use --json examples/use/answers.json --model "$RANKJEV_MODEL"
python -m rankjev.use --json examples/use/stars.json --model "$RANKJEV_MODEL"
python -m rankjev.use --json examples/use/tools.json --model "$RANKJEV_MODEL"
```

命令行直接传：

```bash
python -m rankjev.use \
  --model "$RANKJEV_MODEL" \
  --stem "Rank the candidate answers to the user. Higher score = a better response." \
  --option "sorted(numbers) returns a new list." \
  --option "Numbers cannot be sorted in Python."
```

输出每行是名次、原下标、分数 `score` 、 softmax 份额 `p` 和选项文本。 `p` 是这组分数的 softmax 。 `conf` 由最大概率算出。

在已缓存权重的 Apple MPS 上，加载底座大约 3 秒。之后单次排序大约 0.5 秒；选项更长时大约 1 秒。

三个例子分别是候选回答、星级和只读调用。星级的相邻档可能对调。工具例子不在训练集里。

### 在 dev 上复现验证

样本正文不在仓库里。有 `data/listwise/dev.jsonl` 时：

```bash
python -m rankjev.score \
  --ckpt weights/qwen35-9b-kq-head.pt \
  --model "$RANKJEV_MODEL" \
  --data data/listwise/dev.jsonl \
  --device mps \
  --show 0
```

只看前 300 条时加上 `--limit 300` 。不加 limit 就是训练结束时那 1,767 条。

### 重新训练

```bash
python -m rankjev.fit --model "$RANKJEV_MODEL"
```

这会从随机初始化的头开始，不读取 `weights/qwen35-9b-kq-head.pt` 。数据格式见 [DATA.md](DATA.md)。

## 排序头

在冻结的预训练模型上训练一个列表排序头。底座权重固定，只更新头的参数。头的宽度跟随底座的隐藏维。隐藏维为 4096 时，头的参数量是 540,673。

头为每个选项和当前题目各算一组 $16 \times 4$ 的向量，分组做内积后相加，得到该选项的分数。分数高的选项排在前面。

$$
\begin{aligned}
K_i &= W_k \mathrm{LN}(h_i) \\
Q &= \tanh(W_q \mathrm{LN}(h_s)) \\
z_i &= s \cdot \frac{\sum_j K_{ij} \cdot Q_j}{\sqrt{4 \times 16}}
\end{aligned}
$$

$W_k$ 与 $W_q$ 是两个独立的线性层。 $h_i$ 是第 $i$ 个选项末尾的隐藏状态， $h_s$ 是 `Answer:` 位置的隐藏状态。 $\tanh$ 只作用在题目一侧。 $s$ 是一个可学习的正标量。 $\mathrm{LN}$ 是 LayerNorm。

## 损失

目标只包含顺序。

| 条件 | 损失 |
|---|---|
| `gold` 中不同数值不少于 3 个 | 取目标分数最高的前 10 项，计算预测分数与目标名次的 Pearson 相关 $\rho$ ，损失为 $1-\rho$ 。并列目标使用中位名次 |
| `gold` 中不同数值不超过 2 个 | 对每一对更高分与更低分计算 $\mathrm{softplus}(-(z_{\mathrm{high}}-z_{\mathrm{low}}))$ |

最长公共子序列只在验证时计算。置信度由当次分数的 $\mathrm{softmax}$ 算出，不单独作为损失。

每个训练样本会把 `options` 和 `gold` 用同一个随机置换打乱。

## 文件

| 路径 | 内容 |
|---|---|
| `weights/qwen35-9b-kq-head.pt` | Qwen3.5-9B 的排序头 |
| `rankjev/use.py` | 排序调用 |
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
