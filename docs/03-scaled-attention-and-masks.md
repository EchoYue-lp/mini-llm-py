# 03 Scaled Attention 与 Mask

## 学习目标

读完后应能：

1. 手算一个小型 Q/K/V Attention。
2. 推导 `[B,H,Tq,Tk]` score 的来源。
3. 解释 causal、padding 和 cross-attention mask 的区别。
4. 定位 mask 广播、布尔语义和 NaN 问题。

## 核心公式

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(Dh)) V
```

单个 head 的 shape：

```text
Q: [B, Tq, Dh]
K: [B, Tk, Dh]
V: [B, Tk, Dh]
QK^T: [B, Tq, Tk]
输出: [B, Tq, Dh]
```

每个 query 位置得到一组对 key 的权重，再用权重汇总 value。

## 为什么除以 sqrt(Dh)

`Dh` 增大时，点积方差随之变大。过大的 score 会让 softmax 接近 one-hot，
梯度进入饱和区。缩放让不同 head dimension 下的数值范围更稳定。

## Q、K、V 的直觉

- Query：当前位置想寻找什么。
- Key：每个位置提供什么匹配标签。
- Value：匹配后真正汇总的内容。

Q/K/V 都来自可训练投影，这些功能不是人工固定的字段。

## Causal Mask

位置 `t` 只能看到 `0..t`：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

禁止位置的 score 设为 `-inf`，softmax 后概率为 0。

## Padding Mask

动态 batch 中短序列需要补 PAD。Padding mask 常用 shape：

```text
[B, 1, 1, T]
```

它会广播到所有 head 和 query，阻止任何位置读取 PAD key。

Padding query 可以继续计算，但对应 target 必须通过
`ignore_index=pad_token_id` 从 loss 中排除。

## Cross-Attention Mask

Encoder-Decoder 中，decoder query 读取完整 source memory。Cross mask 只屏蔽 source
中的 PAD，不需要 causal 约束。

## Mask 合并

Decoder self-attention 同时需要 causal 与 padding：

```text
combined_mask = causal_mask AND padding_mask
```

需要确认布尔语义：项目中 `True` 表示可见，`False` 表示屏蔽。

## 实验

```bash
python -m labs.lab01_attention_basics
```

观察：

- attention weight 每一行之和是否为 1。
- 上三角权重是否为 0。
- mask 广播后的 shape 是否正确。

## 对照源码

- `models/layers.py::MultiHeadAttention`
- `utils/mask_utils.py`
- `tests/test_padding_mask.py`

## 一个可以手算的 Attention

假设两个 token，每个向量二维：

```text
Q = [[1, 0],
     [0, 1]]

K = [[1, 0],
     [0, 1]]

V = [[10, 0],
     [0, 20]]
```

先算：

```text
QK^T = [[1, 0],
        [0, 1]]
```

`Dh=2`，缩放后：

```text
scores = QK^T / sqrt(2)
       ~= [[0.707, 0],
           [0, 0.707]]
```

第一行 softmax 约为：

```text
[0.670, 0.330]
```

第一位置输出：

```text
0.670 * [10,0] + 0.330 * [0,20]
= [6.70, 6.60]
```

注意力输出不是“选中一个 token”，而是对 Value 的加权和。

## Mask 在 Softmax 之前应用

正确顺序：

```text
scores = QK^T / sqrt(Dh)
scores = masked_fill(scores, -inf)
weights = softmax(scores)
```

若在 softmax 之后把概率改成 0，剩余概率之和不再是 1，除非重新归一化。

## Mask 的广播

Attention score：

```text
[B, H, Tq, Tk]
```

常见 mask：

| Mask | Shape | 广播含义 |
| --- | --- | --- |
| Causal | `[1,1,T,T]` | 所有 batch/head 共用 |
| Padding | `[B,1,1,Tk]` | 每个样本不同，所有 head/query 共用 |
| Combined | `[B,1,T,T]` | 样本级 causal + padding |

广播可以减少内存，但需要确保每一维语义正确。

## Fully Masked Row 与 NaN

若某一整行全部被设为 `-inf`：

```text
softmax([-inf, -inf, ...])
```

可能产生 NaN。工程中应避免构造完全不可见的有效 query，或对这种情况做显式处理。

## Padding Query 为什么还能计算

Padding mask 通常只屏蔽 key：

```text
query at PAD position -> 仍可读取非 PAD key
```

这些输出最终通过 target loss 的 `ignore_index` 丢弃。这样实现广播简单，也不会影响
有效 token 的结果。

## Attention 权重不等于完整解释

权重显示某层某个 head 在一次前向中的 Value 混合比例，但不能直接当作模型决策的完整
因果解释，因为：

- 后面还有多个层和 FFN。
- Value 本身是投影结果。
- 残差路径绕过 Attention。
- 不同 head 会再次混合。

## 常见错误

1. 将 `True` / `False` 的可见语义写反。
2. Mask 在 softmax 后应用。
3. Padding mask 错误屏蔽 query 维。
4. Cross-attention 使用 target padding mask。
5. Mask 与 score 不在同一 device。
6. 创建完全 masked 的有效行导致 NaN。

## 动手练习

1. 手算上面的两 token attention。
2. 给第一行增加 causal mask，重新计算输出。
3. 构造 batch 中一个短序列，打印广播后的 mask。
4. 将 scale 去掉，增大 `Dh`，观察 softmax 是否更尖锐。

## 自测

1. 为什么 mask 必须在 softmax 前应用？
2. Padding mask 为什么常是 `[B,1,1,T]`？
3. Q/K 决定什么，V 决定什么？
4. 为什么除以 `sqrt(Dh)` 而不是 `sqrt(D)`？
