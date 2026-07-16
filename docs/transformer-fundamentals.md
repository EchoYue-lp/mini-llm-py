# Transformer 基础：从 token 到 logits

本文对应项目中的 PyTorch 从零实现。目标不是只记住模块名称，而是能回答三个问题：

1. 每一步输入输出的 shape 是什么？
2. mask、残差和 LayerNorm 分别解决什么问题？
3. 模型为何能通过 next-token loss 学会生成或翻译？

## 1. 统一符号

| 符号 | 含义 |
| --- | --- |
| `B` | batch size |
| `S` | source 或当前序列长度 |
| `T` | target 序列长度 |
| `D` | `d_model`，隐藏维度 |
| `H` | attention head 数 |
| `Dh` | 每个 head 的维度，`D / H` |
| `V` | 词表大小 |
| `Dff` | FFN 中间维度 |

Decoder-Only 的主干 shape：

```text
token ids [B, T]
  -> embedding [B, T, D]
  -> N x Transformer block [B, T, D]
  -> final norm [B, T, D]
  -> output projection [B, T, V]
```

最后的 `[B, T, V]` 是 logits。每个位置都有一个长度为 `V` 的分数向量，用来预测
下一个 token。

## 2. Tokenizer、Embedding 和位置

### 2.1 Tokenizer

Tokenizer 把文本映射成整数 id。模型不直接接收字符串：

```text
"I love AI" -> [40, 1842, 9552]
```

特殊 token 通常包括：

- `PAD`：补齐 batch，不能贡献 loss。
- `BOS`：目标序列开始。
- `EOS`：序列结束。
- `UNK`：无法表示的内容。

本项目翻译使用 SentencePiece，生成使用 GPT-2 tokenizer。GPT-2 原始词表没有 PAD，
因此项目加载时会增加独立的 `<|pad|>`。不能把 id `0` 直接当 PAD，因为它是 GPT-2
词表中的真实 token。

### 2.2 Token Embedding

Embedding 是可学习查表矩阵：

```text
E: [V, D]
ids: [B, T]
E[ids] -> [B, T, D]
```

项目按原始 Transformer 做法乘以 `sqrt(D)`，避免初始化早期 token embedding 的尺度
相对固定位置编码过小。

### 2.3 位置编码

纯注意力对 token 顺序没有天然认识。经典正弦位置编码为：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/D))
PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
```

然后直接相加：

```text
x = token_embedding * sqrt(D) + positional_encoding
```

固定正弦编码没有可训练参数。现代 LLM 更常使用 RoPE，后文会解释。

## 3. Scaled Dot-Product Attention

注意力的核心公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(Dh)) V
```

以单个 head 为例：

```text
Q: [B, Tq, Dh]
K: [B, Tk, Dh]
V: [B, Tk, Dh]
QK^T: [B, Tq, Tk]
softmax 后乘 V: [B, Tq, Dh]
```

每个 query 位置会得到对所有 key 位置的权重，再用这些权重加权汇总 value。

### 3.1 为什么除以 `sqrt(Dh)`

当 `Dh` 增大时，点积方差也会增大。过大的 score 会把 softmax 推到接近 one-hot 的
饱和区，使梯度变小。缩放让不同 head_dim 下的数值范围更稳定。

### 3.2 Q、K、V 的直觉

- Query：当前位置想找什么。
- Key：每个位置提供什么匹配标签。
- Value：匹配后真正汇总的内容。

Q/K/V 都来自可学习线性投影，因此这种解释不是人工规定的字段，而是训练后形成的功能。

运行：

```bash
python labs/lab01_attention_basics.py
```

重点观察 attention weight 每一行和为 1，以及因果 mask 上三角为 0。

## 4. 三类 Mask

### 4.1 Causal Mask

自回归模型在位置 `t` 只能看 `0..t`，不能偷看未来标签：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

实现中将禁止位置的 score 设为 `-inf`，softmax 后对应概率变成 0。

### 4.2 Padding Mask

动态 batch 中短序列会补 PAD。所有 query 都不应把 PAD 当成有效 key。项目中的
padding mask shape 是 `[B, 1, 1, T]`，可广播到所有 head 和 query。

Padding query 本身可以继续计算，但其输出对应的 target 必须被 loss 的
`ignore_index=pad_token_id` 忽略。

### 4.3 Cross-Attention Mask

Encoder-Decoder 中，decoder query 读取 encoder 的 source memory。cross mask 只需屏蔽
source 里的 PAD，不需要因果约束，因为完整源句在翻译时已经可见。

## 5. Multi-Head Attention

先把 `D` 拆成 `H` 个子空间：

```text
[B, T, D]
  -> [B, T, H, Dh]
  -> [B, H, T, Dh]
```

每个 head 独立计算注意力，再合并：

```text
[B, H, T, Dh]
  -> [B, T, H, Dh]
  -> [B, T, D]
  -> output projection
```

多个 head 允许模型同时学习不同关系，例如局部搭配、长距离依赖、句法角色或位置模式。
这些语义不是预先固定的，也不能保证每个 head 都有清晰的人类解释。

约束 `D % H == 0`，否则无法等分。运行：

```bash
python labs/lab02_multi_head_attention.py
```

## 6. FFN：每个 token 独立变换

经典 Position-wise FFN：

```text
FFN(x) = W2 ReLU(W1 x + b1) + b2
```

shape：

```text
[B, T, D] -> [B, T, Dff] -> [B, T, D]
```

注意力负责 token 之间的信息混合；FFN 在每个 token 位置独立进行非线性特征变换。
所有位置共享同一组 FFN 参数。

## 7. 残差、LayerNorm 和 Pre-LN

项目采用 Pre-LN：

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Encoder-Decoder 的 decoder block 中间还多一个 cross-attention：

```text
x = x + SelfAttention(LN(x))
x = x + CrossAttention(LN(x), encoder_memory)
x = x + FFN(LN(x))
```

残差提供从深层直接回到浅层的恒等路径。Pre-LN 把归一化放在子层之前，通常比原论文
Post-LN 更容易训练深网络。由于最后一次残差之后没有归一化，整个 stack 末尾还需要
final LayerNorm。

运行：

```bash
python labs/lab03_pre_ln_block.py
```

项目为了复用代码，让 Decoder-Only 也使用通用 `DecoderLayer`。因此其中的
cross-attention 参数在 Decoder-Only 模式不会参与计算。这不影响数值正确性，但会让参数
统计略显冗余；学习时应明确 GPT 类 block 实际只有 self-attention 和 FFN。

## 8. 三种 Transformer 架构

### 8.1 Encoder-Only

双向 self-attention，每个 token 可看到整段输入。典型用途是分类、抽取和表征学习。
本项目没有单独实现该架构。

### 8.2 Decoder-Only

只使用 causal self-attention：

```text
prompt -> 预测下一个 token -> 拼回输入 -> 继续预测
```

GPT 类大语言模型属于这一类。项目对应 `DecoderOnlyModel`。

### 8.3 Encoder-Decoder

Encoder 双向读取 source；Decoder 先做因果 self-attention，再通过 cross-attention 读取
source memory。T5 和经典机器翻译 Transformer 属于这一类。项目对应
`EncoderDecoderModel`。

运行最小 copy task：

```bash
python labs/lab04_tiny_copy_task.py --steps 400
```

Copy task 没有语言歧义，适合先验证 teacher forcing、mask、shift 和模型连线是否正确。

## 9. Next-Token Loss

给定 token：

```text
[BOS, 我, 喜欢, AI, EOS]
```

训练时错一位对齐：

```text
inputs:  [BOS, 我,   喜欢, AI]
labels:  [我,  喜欢, AI,   EOS]
```

模型输出 `[B, T, V]` logits，交叉熵鼓励正确 label 的概率变大：

```text
loss = -log softmax(logits)[correct_token]
```

Decoder-Only 和 Encoder-Decoder decoder 都使用这个目标。区别在于后者还读取 encoder
memory。

Teacher forcing 指训练时 decoder 输入真实历史 token，而不是模型自己上一步的预测。
它能并行训练，但也造成训练和逐 token 推理之间的输入分布差异。

运行最小语言模型：

```bash
python labs/lab05_tiny_language_model.py --steps 100
```

## 10. 训练循环中容易错的地方

### 10.1 Padding 必须同时作用于 attention 和 loss

- attention mask：PAD 不能作为 key 被读取。
- `ignore_index`：PAD target 不能贡献交叉熵。

二者职责不同，缺一不可。

### 10.2 梯度累积

如果累积 `K` 个 micro-batch：

```text
loss = loss / K
backward K 次
optimizer.step 1 次
```

学习率调度器的 total steps 应按 optimizer update 次数计算，而不是 micro-batch 次数。
最后不足 `K` 的一组还要按实际数量缩放。

### 10.3 AMP 和梯度裁剪

CUDA AMP 可降低显存和提高吞吐。使用 GradScaler 时，应先 `unscale_`，再裁剪梯度，
最后执行 optimizer step。

### 10.4 Train、Validation、Test

- Train：计算梯度。
- Validation：选 checkpoint 和超参数。
- Test：所有选择完成后做最终评测。

只看训练 loss 不能证明生成质量或泛化能力。

## 11. 自回归解码

### Greedy

每步取概率最高 token。速度快、确定性强，但可能过早走入局部最优。

### Top-K

只保留概率最高的 K 个 token，再归一化采样。

### Top-P / Nucleus

按概率降序，保留累计概率首次达到或超过 `p` 的最小集合。实现时必须包含“第一个使累计
概率越过阈值”的 token。

### Beam Search

维护多个累计 log-prob 最高的候选。翻译常用，但需要长度惩罚，避免短序列天然占优。
开放式创作通常更适合采样而不是 beam search。

## 12. 复杂度

标准 self-attention 的 score 矩阵是 `[T, T]`：

```text
时间复杂度约 O(T^2 D)
attention memory 约 O(T^2)
```

长上下文的主要瓶颈来自二次方 attention；自回归推理还会反复计算历史 K/V。KV Cache、
FlashAttention 和稀疏注意力分别从重复计算、内存访问和连接结构上缓解问题。

## 13. 对照源码

| 概念 | 文件 |
| --- | --- |
| Scaled attention / MHA / FFN / PE | `models/layers.py` |
| Pre-LN Encoder/Decoder block | `models/decoder_encoder_layer.py` |
| Decoder-Only / Encoder-Decoder | `models/transformer_models.py` |
| Causal / Padding mask | `utils/mask_utils.py` |
| LM 训练 | `scripts/train_decoder.py` |
| 翻译训练 | `scripts/train_encoder_decoder.py` |
| Greedy / Top-K / Top-P / Beam | `utils/generation_utils.py`、`utils/translation_utils.py` |

读源码时始终在纸上写出 shape。只要 shape、可见范围和 loss 对齐三件事明确，大多数
Transformer bug 都能较快定位。
