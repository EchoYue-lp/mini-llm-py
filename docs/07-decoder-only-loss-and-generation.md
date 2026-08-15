# 07 Decoder-Only、Next-Token Loss 与生成

Decoder-Only 模型只使用 causal self-attention，GPT 类模型属于这一架构。

## 学习目标

读完后应能：

1. 从 token sequence 构造 input 与 target。
2. 解释 cross entropy、perplexity 和有效 token 平均。
3. 比较 Greedy、Temperature、Top-K、Top-P 与 Beam Search。
4. 说明训练并行与自回归生成串行的原因。

## 本章符号与 Shape

| 符号 | 含义 | 典型 Tensor |
| --- | --- | --- |
| `B` | batch size | input ids `[B,T]` |
| `T` | 当前上下文长度 | hidden `[B,T,D]` |
| `D` | model dimension | 每个 token 的 hidden state |
| `V` | vocabulary size | logits `[B,T,V]` |
| `N` | 未被 ignore 的有效 target 数 | loss reduction 的分母 |

生成时通常只读取 `logits[:, -1, :]`，shape 是 `[B,V]`；训练时则同时监督多个位置，并将
`[B,T,V]` 与 `[B,T]` 展平为 `[B*T,V]` 和 `[B*T]`，但对应顺序必须保持一致。

## 训练数据

连续 token 被切成固定长度序列：

```text
[t0, t1, t2, t3, t4]
```

训练时：

```text
input:  [t0, t1, t2, t3]
target: [t1, t2, t3, t4]
```

模型输出 `[B, T, V]` logits，交叉熵优化正确下一个 token 的概率。

## Causal Loss

```text
loss = -log softmax(logits)[target]
```

PAD target 通过 `ignore_index` 排除。若使用梯度累积，loss 应按累积步数正确缩放，
学习率调度器按 optimizer update 次数前进。

## 最小语言模型

```bash
python -m labs.lab05_tiny_language_model --steps 100
```

实验使用可预测的 token 规律，目的是确认 loss 下降和生成闭环，而不是学习自然语言。
它会逐元素断言 `label=(input+1) mod V`，并打印随机均匀模型的交叉熵基线 `log(V)`。

## 完整生成流程

```bash
python -m scripts.download_datasets --generation
python -m scripts.preprocess --generation
python -m scripts.train_decoder
python -m scripts.generate
```

## 自回归生成

```text
prompt
  -> 预测 next token
  -> 拼接到输入
  -> 继续预测
```

### Greedy

每步选择概率最高 token。确定、快速，但容易进入局部最优。

### Top-K

只保留最高的 K 个 token，再归一化采样。

### Top-P

按概率降序，保留累计概率首次达到 `p` 的最小集合。必须包含第一个使累计概率越过
阈值的 token。

### Beam Search

维护多个累计分数最高的候选。适合翻译，开放式生成通常更适合采样。

## 长度与复杂度

标准 self-attention：

```text
时间复杂度约 O(T^2 D)
attention memory 约 O(T^2)
```

逐 token 生成若每次重算历史，会重复计算旧 token 的 K/V。下一篇文档介绍 KV Cache。

## 常见错误

1. 预处理时没有保留 EOS 边界。
2. 直接把 GPT-2 token 0 当 PAD。
3. 超过位置编码最大长度。
4. Top-P 候选集合少保留一个越界 token。
5. 训练时使用 padding mask，推理时却遗漏。

## 对照源码

- `models/transformer_models.py::DecoderOnlyModel`
- `scripts/train_decoder.py`
- `utils/generation_utils.py`
- `tests/test_preprocessing.py`
- `tests/test_sequence_length_handling.py`

## 一个 Loss 例子

假设词表只有 4 个 token，某位置 logits：

```text
[2.0, 0.5, -1.0, 0.0]
```

若正确 token 是第 0 个，softmax 后正确概率较高，loss 较小；若正确 token 是第 2 个，
loss 会明显更大。

一个 sequence 的 loss 是有效 target token loss 的平均。PAD 和 prompt mask 决定哪些
位置计入平均。

## Perplexity

```text
perplexity = exp(average_loss)
```

若 average loss 为 2：

```text
perplexity ~= 7.39
```

可以粗略理解为模型在每一步面对约 7.39 个同等可能选项，但该解释只在同 tokenizer、
同数据分布下比较才有意义。

## Temperature

采样前缩放 logits：

```text
scaled_logits = logits / temperature
```

| Temperature | 效果 |
| --- | --- |
| 小于 1 | 分布更尖锐，更保守 |
| 等于 1 | 原始分布 |
| 大于 1 | 分布更平坦，更随机 |

Temperature 不能修复训练不足，只改变采样行为。

## Top-K 的步骤

1. 取 logits 最大的 K 个位置。
2. 其余位置设为 `-inf`。
3. 对剩余 logits 做 softmax。
4. 从该分布采样。

K=1 等价于 Greedy。

## Top-P 的步骤

1. Softmax 得到概率。
2. 按概率降序。
3. 计算累计概率。
4. 保留首次达到阈值的最小集合。
5. 重新归一化并采样。

Top-P 候选数量会随当前分布变化：模型很确定时集合小，不确定时集合大。

## EOS 与停止条件

生成循环通常在以下情况停止：

- 生成 EOS。
- 达到 `max_new_tokens`。
- 命中特定 stop sequence。

若 BOS 与 EOS 共享 id，必须防止初始 token 被误判为已经完成。

## Train、Eval 与 No-Grad

推理前：

```python
model.eval()
with torch.no_grad():
    ...
```

`eval()` 关闭 dropout，`no_grad()` 避免保存反向传播图。两者作用不同，都应使用。

## 为什么生成越来越慢

没有 KV Cache 时，第 `t` 步会重新处理长度为 `t` 的整个序列。上下文越长，
每一步计算越多。KV Cache 让历史 K/V 复用，但当前 Q 仍需读取全部历史 cache。

## 生成质量问题的来源

- 数据过少或分布不匹配。
- Tokenizer 对目标语言效率低。
- 模型容量不足。
- 训练不足或过拟合。
- Sampling 参数不合适。
- Prompt 超出训练经验。
- 重复模式被模型放大。

不能只调整 temperature 来解决所有问题。

## Causal LM 的概率目标

Decoder-Only 学习：

$$
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

对一个 token 的稳定交叉熵：

$$
L_t=\log\sum_v e^{z_{t,v}}-z_{t,y_t}
$$

对有效 token 平均：

$$
L=\frac{\sum_t m_tL_t}{\sum_t m_t}
$$

其中 `m_t` 表示该 target 是否参与 loss。PAD、prompt masking 或特殊任务 mask 都通过这个
权重决定训练目标。只构造 causal mask 不会自动排除 PAD loss。

## 文档切块与 EOS

语言模型预处理常把长 token 流切成固定窗口。需要明确边界策略：

- 文档间插入 EOS：模型能学习停止与边界。
- 直接拼接无 EOS：模型可能把两篇文档当连续文本。
- 每篇独立 padding：边界明确，但 padding 比例可能更高。
- 滑动窗口重叠：增加样本与上下文覆盖，也会重复计算 token loss。

本项目 `preprocess.py` 在非空文本行后插入 EOS，再切分 token 流。若最后一个 chunk 太短会
被过滤；若 chunk 截断发生在文档中间，下一个 chunk 仍可继续该文档，但训练样本之间不会
共享运行时 hidden state。

## Perplexity 的严格解释边界

Perplexity 是平均 NLL 的指数：

$$
PPL=\exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log P(y_t)\right)
$$

它是 tokenization-dependent 指标。更细粒度 tokenizer 会改变 token 数和每 token 难度，
因此不同词表、不同 normalization、不同是否计入 EOS/PAD 的 PPL 不能直接横比。

Loss 很大时 `exp(loss)` 可能溢出，日志可先保留 NLL，或只在合理范围内展示 PPL。

## Temperature 的概率比变化

对两个 token `i,j`：

$$
\frac{p_i(\tau)}{p_j(\tau)}
=\exp\left(\frac{z_i-z_j}{\tau}\right)
$$

因此 temperature 直接缩放 log-odds。`tau` 越小，原有 logit 差异被放大；`tau` 越大，差异
被压平。`temperature=0` 没有数学定义，Greedy 应作为单独分支处理，而不是除以零。

## Top-K 与 Top-P 都改变了原分布

设保留集合为 `S`，过滤后的采样概率：

$$
p'(i)=\frac{p(i)}{\sum_{j\in S}p(j)},\quad i\in S
$$

Top-K 固定集合大小，Top-P 固定累计质量阈值。Top-P 必须保留第一个使累计概率越过 `p`
的 token，否则实际保留质量可能小于阈值。本项目 `top_p_candidates()` 通过把 remove mask
右移一位实现这一点。

过滤、temperature 和 Softmax 的推荐顺序：

```text
raw logits
-> divide by temperature
-> top-k/top-p filtering
-> softmax/renormalize
-> multinomial sample
```

项目部分实现先 Softmax 再选择 Top-K，数学上对排序集合等价，但生产实现常在 logits 上
过滤以减少不必要计算并统一数值处理。

## Beam Search 不是“更聪明的 Greedy”

Beam Search 近似寻找高序列概率候选，每步保留有限前缀。它仍可能：

- 早期剪掉后来更优的路径。
- 偏向短序列。
- 多个 beam 高度相似。
- 在开放式生成中产生保守、重复文本。

Beam width 增大不保证任务质量单调上升。翻译等条件生成常受益于搜索，开放式文本通常更
关注分布多样性，因此常用采样。

## Batch 生成的 Finished 状态

单样本代码可以在 EOS 时 `break`。Batch 生成中各样本完成时间不同，需要维护：

```text
finished: [B] boolean
```

已完成样本后续应写 PAD/EOS 占位，但不能让其继续改变累计分数或 cache 语义。只有
`finished.all()` 才能结束整个循环。Beam Search 还需要每个 beam 独立 finished 状态。

Stop sequence 可能跨 token 边界，不能只比较最后一个 token id。字符串 stop 还需考虑
decode normalization 和部分匹配。

## Context Window 与位置边界

生成时总长度：

```text
prompt_tokens + generated_tokens
```

必须不超过模型位置上限。常见策略：

- 拒绝过长 prompt。
- 从左侧截断旧上下文。
- 保留 system/关键前缀，截断中间历史。
- 使用经过长上下文训练和正确 scaling 的模型。

简单截断可能删除 BOS、系统指令或配对消息边界。位置上限不是只约束 `max_new_tokens`。

## 推荐的最小生成循环

```python
model.eval()
tokens = prompt_ids

with torch.no_grad():
    for _ in range(max_new_tokens):
        if tokens.size(1) >= model_max_length:
            break
        mask = create_causal_mask(tokens.size(1), tokens.device)
        logits, _ = model(tokens, mask=mask)
        next_logits = logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
        if (next_token == eos_token_id).all():
            break
```

`scripts/generate.py::greedy_generate` 与 `utils/generation_utils.py` 都会在循环前检查
`prompt_tokens + generated_tokens`，使用 inference mode，并在结束后恢复模型原先的
train/eval 状态。这里的 `max_len` 兼容旧 API，语义是“新生成 token 数”，不是总长度。

## 随机性与可复现

采样结果由以下因素共同决定：

- 随机 seed 与 RNG device。
- Temperature、Top-K、Top-P。
- 模型是否 `eval()`。
- 浮点 dtype、device 和非确定性 kernel。
- Batch 中其他样本是否改变 RNG 消耗顺序。

比较采样策略时应固定 seed，并报告完整解码参数。固定 seed 不意味着跨硬件和框架版本逐
token 完全一致。

## 本章调试不变量

1. Input 与 target 只 shift 一次，logits/labels token 数一致。
2. Loss 分母是有效 target token 数，不包含 PAD。
3. 每步只使用最后位置 logits 选择 next token。
4. Temperature 必须大于 0；Greedy 单独处理。
5. Top-K/P 过滤后概率重新归一化且至少保留一个 token。
6. Prompt + output 不超过上下文长度。
7. 推理同时使用 `eval()` 与 no-grad/inference mode。
8. Batch/Beam 的 finished 状态、token、score 与 cache 同步。

## 动手练习

1. 对同一 prompt 比较 Greedy、Top-K 和 Top-P。
2. 固定随机种子，改变 temperature。
3. 将 `max_len` 设置得很小，观察边界错误。
4. 打印每一步候选 token 和概率。
5. 对比使用和不使用 `torch.no_grad()` 的内存。

## 自测

1. 为什么 Decoder-Only 训练可以并行，生成却必须串行？
2. Temperature、Top-K、Top-P 分别改变什么？
3. Perplexity 能否跨 tokenizer 直接比较？
4. `model.eval()` 与 `torch.no_grad()` 有什么区别？
