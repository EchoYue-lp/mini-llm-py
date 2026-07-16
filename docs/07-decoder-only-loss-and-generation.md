# 07 Decoder-Only、Next-Token Loss 与生成

Decoder-Only 模型只使用 causal self-attention，GPT 类模型属于这一架构。

## 学习目标

读完后应能：

1. 从 token sequence 构造 input 与 target。
2. 解释 cross entropy、perplexity 和有效 token 平均。
3. 比较 Greedy、Temperature、Top-K、Top-P 与 Beam Search。
4. 说明训练并行与自回归生成串行的原因。

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
