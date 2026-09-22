# 数据规格

`data/listwise/train.jsonl` 与 `data/listwise/dev.jsonl` 的构建规则如下。仓库只提交 `manifest.json`，不提交样本正文。

按本规格生成的结果：训练 22,062 条，验证 1,767 条。编码长度用 Qwen3.5 词表计算，上限 768。4B 与 9B 使用同一词表。

目标分数只来自原始数据中的标注、可执行规则或既定顺序。

## 字段

```json
{
  "id": "ultrafeedback/015509/flan_v2_flan2021/a9bd58eddf",
  "source": "ultrafeedback",
  "stem": "Rank the candidate answers to the user. Higher score = a better response.",
  "options": ["...", "..."],
  "gold": [1.5, 4.25, 4.25, 5.0],
  "k": 4,
  "qtype": "rank",
  "n_unique": 3,
  "gap": 0.75,
  "span": 3.5
}
```

| 字段 | 定义 |
|---|---|
| `id` | 全局唯一 |
| `source` | 来源名称，切分和统计按此字段分组 |
| `stem` | 题干 |
| `options` | 选项文本。同一条内不得重复 |
| `gold` | 与 `options` 等长的目标分数，越大越靠前 |
| `k` | `len(options)` |
| `qtype` | `rank`：多档回答。`score`：有序评分。`choice`：多个选项中一个正确。`noul`：是非两选项 |
| `n_unique` | `gold` 中不同数值的个数，比较前四舍五入到 5 位小数 |
| `gap` | 将不同数值排序后，相邻差值的最小值 |
| `span` | `max(gold) - min(gold)` |

落盘前用 `id` 的哈希对 `options` 和 `gold` 做同一次置换。训练加载时再随机置换一次。

接受一条记录的条件：

1. `k >= 2`
2. `len(options) == len(gold)`
3. `options` 内无重复文本
4. `n_unique >= 2`
5. 编码长度 `<= 768`

编码顺序为：题干、`A.` 到选项末尾、固定后缀 `Answer:`。`Answer:` 的位置必须落在全部选项位置之后。

## 标签规则

### UltraFeedback

输入：`openbmb/UltraFeedback`。

一条回答的分数是 `instruction_following`、`honesty`、`truthfulness`、`helpfulness` 中已有评分的平均值。已有评分少于 3 个则丢弃该回答。不读取 `overall_score`。

同一题内，按分数从高到低保留回答。词集合 Jaccard `>= 0.9` 的后出现回答删除。剩余回答须满足 `k >= 3`、`n_unique >= 3`、`gap >= 0.5`、`span >= 1.5`。

超过 768 token 时，在非唯一最高分且非唯一最低分的回答中删除最长的一条，然后重新检查上述四项。无法同时满足时丢弃该题。

`id` 格式：`ultrafeedback/{index:06d}/{subset}/{sha1(prompt)[:10]}`。`index` 是数据集行号。

### HelpSteer2

输入：`nvidia/HelpSteer2` 的 `train` 与 `validation`。

按 `prompt` 聚合。分数使用人工字段 `helpfulness`。Jaccard 去重阈值同上。要求 `k >= 2` 且 `span >= 1`。`k > 4` 时保留 `helpfulness` 最高的 4 条。

超过 768 token 的记录丢弃，不删回答。

`id` 格式：`helpsteer2/{split}/{sha1(prompt)[:16]}`。

### 定额来源

下表来源已有选项和目标分数。按 `source` 去重并套用长度上限后，用固定随机种子打乱，再截断到上限。`n_unique <= 2` 且 `k > 6` 的记录不收。

| source | 上限 | 目标分数 |
|---|---:|---|
| numeric | 1000 | 数值越接近标准答案越高 |
| date | 600 | 时间或步骤越早越高 |
| yelp/score | 900 | 星级 |
| amazon/score | 900 | 星级 |
| sst5/score | 900 | 情感等级 |
| legacy_policy/score | 448 | 规则题的等级分 |
| mnli/choice | 900 | 正确关系为 1，其余为 0 |
| agnews/choice | 600 | 正确主题为 1，其余为 0 |
| compositional/choice | 700 | 规则结果匹配为 1，其余为 0 |
| boolq/noul | 600 | 与段落一致的选项为 1 |
| trec/choice | 500 | 正确问题类型为 1，其余为 0 |
| legacy_policy/choice | 224 | 正确决策为 1，其余为 0 |
| legacy_policy/noul | 200 | 规则允许为 1，否则为 0 |

不接收 `banking77`、`dbpedia14`，以及 `yelp`、`imdb`、`agnews` 的是非题。

### 规则生成

生成器产出的每条记录使用独立输入，生成后同样套用长度上限。

| source | 输入 | 目标分数 |
|---|---|---|
| order_restore | 5–8 个有固定先后的步骤。原序列长于 8 时只取前 7 步和后 7 步 | 原位置越靠前越高 |
| web_answer | 一个查询和四段文本 | 直接答案 3，相关文本 1.5，其余 0 |
| web_relevance | 一个查询和四条结果 | 4、2.5、1、0 |
| web_noul | 查询加一段摘要，选项为是/否 | 与事实一致的选项为 1 |
| web_intent | 一个查询和五个意图 | 正确意图为 1，其余为 0 |
| policy_rule | 退款期限、响应时限、年龄、密码、优惠券的结构化条件 | 条件成立为 1，否则为 0 |
| nli_claim | 证据、声明，以及支持/不足/矛盾 | 正确关系为 1，其余为 0 |

## 训练 / 验证划分

去重键为题干加上排序后的选项文本。按 `source` 分组后，组内按 `sha1(id)` 排序，取前 `n_val` 条为验证，其余为训练。

| 该来源条数 `n` | `n_val` |
|---|---|
| `< 10` | 0 |
| `10–19` | 2 |
| `>= 20` | `max(8, round(0.1 * n))`，且验证后训练至少保留 8 条 |

额外上限：`ultrafeedback` 的 `n_val <= 600`，`helpsteer2` 的 `n_val <= 250`。

划分后两个集合分别用种子 `20260921` 打乱。

验收计数：

- `id` 无重复
- 无 `len(options) != len(gold)` 或选项内部重复
- 训练集中 `n_unique >= 3` 为 14,600，`n_unique <= 2` 为 7,462
- 验证集中 `source == ultrafeedback` 的比例为 0.3396
