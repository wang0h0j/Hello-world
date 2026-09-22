# rank-kq

[中文](README.md)

## Qwen3.5-9B weights

`weights/qwen35-9b-kq-head.pt` is a ranking head trained for one epoch on a frozen Qwen3.5-9B. The file is about 2.1 MB. It contains the head parameters, the step count, and the dev metrics. It does not contain the backbone or the optimizer state.

| Item | Value |
|---|---|
| Backbone | `Qwen/Qwen3.5-9B` , hidden size 4096 |
| Head | 16 groups of 4 , 540,673 parameters |
| Steps | 22,062 , batch size 1 , one epoch |
| Max length | 768 |
| Dev , 1,767 rows | top1 0.715 , MRR 0.846 , NDCG 0.917 , Spearman 0.634 , LCS 0.820 |

These weights match Qwen3.5-9B only. Another backbone has a different hidden size or tokenizer, so the checkpoint does not apply. The training code can be used with another model, but that is a new run, not a load of this file.

The backbone is about 18 GB and is not included. A local directory or a Hub id both work:

```bash
export RANKJEV_MODEL=/path/to/Qwen3.5-9B
# or
export RANKJEV_MODEL=Qwen/Qwen3.5-9B
```

Use `mps` on Apple Silicon and `cuda` on NVIDIA. Run the commands below from the repository root.

### Install

```bash
pip install -r requirements.txt
```

`torch>=2.2` and `transformers>=5.7` are required to load Qwen3.5.

### Rank a list

The input is one JSON object, or an array of such objects:

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

`stem` is the question. `options` is 2 to 16 distinct strings. A higher score ranks earlier.

```bash
python -m rankjev.use --demo --model "$RANKJEV_MODEL"
python -m rankjev.use --json examples/use/answers.json --model "$RANKJEV_MODEL"
python -m rankjev.use --json examples/use/stars.json --model "$RANKJEV_MODEL"
python -m rankjev.use --json examples/use/tools.json --model "$RANKJEV_MODEL"
```

Or pass the list on the command line:

```bash
python -m rankjev.use \
  --model "$RANKJEV_MODEL" \
  --stem "Rank the candidate answers to the user. Higher score = a better response." \
  --option "sorted(numbers) returns a new list." \
  --option "Numbers cannot be sorted in Python."
```

Each output line is the rank, the original index, the score, the softmax share `p` , and the option text. `p` is only the relative weight of this call. It is not a calibrated probability. `conf` is derived from the maximum probability and is not trained on its own.

On Apple MPS with the backbone already cached, loading takes about 3 seconds. One ranking call then takes about 0.5 seconds, or about 1 second when the options are longer. Almost all of that time is the 9B forward pass.

The three examples match the uses that currently hold up: ranking candidate answers, ordering star ratings, and ordering a short list of read-only calls. Adjacent star levels can swap. The tool example is not in the training set, so treat it as a probe. Do not use these weights for entailment decisions such as MNLI.

### Reproduce the dev numbers

The row text is not in the repository. When `data/listwise/dev.jsonl` is present:

```bash
python -m rankjev.score \
  --ckpt weights/qwen35-9b-kq-head.pt \
  --model "$RANKJEV_MODEL" \
  --data data/listwise/dev.jsonl \
  --device mps \
  --show 0
```

Add `--limit 300` to score only the first 300 rows. Without it, the run is the same 1,767-row dev set used at the end of training.

### Train again

```bash
python -m rankjev.fit --model "$RANKJEV_MODEL"
```

This starts from a randomly initialized head and does not load `weights/qwen35-9b-kq-head.pt` . The data format is in [DATA.md](DATA.md).

## Ranking head

The head is trained on a frozen pretrained model. Backbone weights stay fixed. The head width follows the backbone hidden size. At hidden size 4096 the head has 540,673 parameters.

Each option and the question are mapped to a group of $16 \times 4$ vectors. A grouped inner product is summed into one score per option. Higher scores rank first.

$$
\begin{aligned}
K_i &= W_k \mathrm{LN}(h_i) \\
Q &= \tanh(W_q \mathrm{LN}(h_s)) \\
z_i &= s \cdot \frac{\sum_j K_{ij} \cdot Q_j}{\sqrt{4 \times 16}}
\end{aligned}
$$

$W_k$ and $W_q$ are separate linear maps. $h_i$ is the hidden state at the end of option $i$ . $h_s$ is the hidden state at the `Answer:` position. $\tanh$ is applied only on the question side. $s$ is a learned positive scalar. $\mathrm{LN}$ is LayerNorm.

## Loss

Only order is supervised.

| Condition | Loss |
|---|---|
| `gold` has at least 3 distinct values | Pearson correlation $\rho$ between predicted scores and target ranks on the top 10 target scores. The loss is $1-\rho$ . Tied targets use midranks |
| `gold` has at most 2 distinct values | For each higher/lower pair , $\mathrm{softplus}(-(z_{\mathrm{high}}-z_{\mathrm{low}}))$ |

Longest common subsequence is computed only at evaluation. Confidence is derived from a $\mathrm{softmax}$ over the scores and is not a separate loss.

Each training sample applies one random permutation to `options` and `gold` together.

## Files

| Path | Contents |
|---|---|
| `weights/qwen35-9b-kq-head.pt` | Ranking head for Qwen3.5-9B |
| `rankjev/use.py` | Ranking entry point |
| `rankjev/head.py` | Ranking head |
| `rankjev/fit.py` | Training entry point |
| `rankjev/score.py` | Evaluation entry point |
| `rankjev/check.py` | Parameter count, permutation, and gradient checks |
| `rankjev/train.py` | Optimizer, schedule, logs, checkpoints |
| `rankjev/losses.py` | Losses |
| `rankjev/metrics.py` | top-1 , Spearman , NDCG , LCS |
| `rankjev/data.py` | JSONL loading and list encoding |
| `rankjev/model.py` | Frozen-backbone forward |
| `rankjev/lcs.py` | Sequence alignment used in evaluation |
