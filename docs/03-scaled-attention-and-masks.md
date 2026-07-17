# 03 Scaled Attention 与 Mask

## 学习目标

读完后应能：

1. 手算一个小型 Q/K/V Attention。
2. 推导 `[B,H,Tq,Tk]` score 的来源。
3. 解释 causal、padding 和 cross-attention mask 的区别。
4. 定位 mask 广播、布尔语义和 NaN 问题。

## 本章符号与 Shape

| 符号 | 含义 |
| --- | --- |
| `B` | batch size |
| `H` | attention head 数；单头示例中可省略 |
| `Tq` | query 序列长度，决定输出有多少行 |
| `Tk` | key/value 序列长度，决定每行分布有多少列 |
| `Dh` | query/key 的 head dimension |
| `Dv` | value 的特征宽度，标准 self-attention 中通常等于 `Dh` |

完整多头合同是：

```text
Q       [B,H,Tq,Dh]
K       [B,H,Tk,Dh]
V       [B,H,Tk,Dv]
scores  [B,H,Tq,Tk]
output  [B,H,Tq,Dv]
```

因此 Q/K 最后一维必须相等，K/V 的 `Tk` 必须相等；`Tq` 与 `Tk` 可以不同，这正是
cross-attention 和增量解码中常见的情况。

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
- `Dh=64` 时 raw score 标准差是否接近 8，缩放后是否接近 1。
- 未缩放与缩放后的 Attention entropy 哪个更低。

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

## Attention 的索引级定义

对单个 batch/head，score、weight 和输出分别为：

$$
s_{ij}=\frac{1}{\sqrt{D_h}}\sum_d q_{i,d}k_{j,d}
$$

$$
a_{ij}=\frac{\exp(s_{ij})}{\sum_{j'}\exp(s_{ij'})}
$$

$$
o_{i,d}=\sum_j a_{ij}v_{j,d}
$$

这个定义揭示两次不同的求和：

1. `QK^T` 沿 feature 轴 `Dh` 求和，产生 token-to-token score。
2. `weights @ V` 沿 key 位置 `Tk` 求和，把 Value 混合回 feature 表示。

因此 Attention 不是简单“比较相似度”，而是先用 Q/K 生成数据依赖的混合矩阵，再用该矩阵
重组 V。

## Attention 矩阵的概率性质

Softmax 后、Dropout 前，每个有效 query 行满足：

```text
weights[i,j] >= 0
sum_j weights[i,j] = 1
```

所以输出每一维都是 Value 行的凸组合。它会落在 Value 点集的凸包中，但后续 output
projection、残差和 FFN 会继续改变表示。

若训练时对 attention weights 使用 Dropout：

```python
dropped = dropout(weights)
output = dropped @ value
```

PyTorch Dropout 会把保留项除以 `1-p` 以保持期望不变，因此单次 forward 中
`dropped.sum(-1)` 通常不再等于 1。测试“权重行和为 1”时应检查 Dropout 前权重，或在
`model.eval()` 下检查。

本项目 `ScaledDotProductAttention` 返回原始 `attn` 用于观察，但 output 使用 dropout 后
权重，这两个张量语义不同。

## Boolean Mask 与 Additive Mask

两种常见 API：

```python
# Boolean: True 表示可见
scores = scores.masked_fill(~visible, float("-inf"))

# Additive: 允许位置加 0，禁止位置加 -inf
scores = scores + additive_mask
```

它们数学等价，但布尔语义经常因框架而异。有的 API 中 `True` 表示“屏蔽”，本项目则表示
“可见”。接入 PyTorch fused attention 或第三方实现时，必须重新核对文档，不能复用变量名
就假设语义相同。

Mask 应能广播到：

```text
scores [B,H,Tq,Tk]
```

每一维必须明确：

| 轴 | Mask 是否通常变化 |
| --- | --- |
| B | Padding 随样本变化 |
| H | 通常所有 head 共用 |
| Tq | Causal 可见范围随 query 变化 |
| Tk | 表示具体被允许读取的 key |

## 数值稳定的 Masked Softmax

普通行可直接：

```python
masked_scores = scores.masked_fill(~visible, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)
```

若业务允许 fully masked row，需要先规定其语义。例如约定输出全零：

```python
masked_scores = scores.masked_fill(~visible, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)
weights = torch.nan_to_num(weights, nan=0.0)
output = weights @ value
```

但这只是定义了退化行的输出，不会修复错误 mask。更严格的调试代码应先检测：

```python
fully_masked = ~visible.any(dim=-1)
if fully_masked.any():
    raise ValueError("valid query has no visible key")
```

使用有限负数如 `torch.finfo(dtype).min` 可以避免某些 `-inf` 路径，但全屏蔽行会变成近似
均匀或实现相关结果，并不自动具有正确语义。

## 增量解码中的矩形 Causal Mask

训练时常见方阵 `[T,T]`。有 KV Cache 时，新 query 长度可能为 1，而 key 包含全部历史：

```text
Q: [B,H,1,Dh]
K: [B,H,Tcache+1,Dh]
score: [B,H,1,Tcache+1]
```

此时当前 token 可以读取所有 cache key，不应机械创建一个 `[1,1]` 下三角 mask。若一次
decode 多个新 token，还需要考虑 query 的绝对起始位置：

```text
visible(i,j) = key_position(j) <= query_position(i)
```

这也是缓存实现中“shape 没错但输出不等价”的高频来源。

## 计算量与内存

单个 head：

```text
QK^T:      O(Tq * Tk * Dh)
weights V: O(Tq * Tk * Dh)
score memory: O(Tq * Tk)
```

多头总量近似把 `H * Dh` 合回 `D`：

```text
compute O(Tq * Tk * D)
attention matrix memory O(H * Tq * Tk)
```

标准实现显式保存 score/weights，长序列时二次项成为主要内存瓶颈。FlashAttention 优化的是
这部分 IO 和中间矩阵物化方式，不会改变 Attention 的数学结果。

## 可运行的广播检查

```python
import math
import torch

B, H, T, Dh = 2, 3, 4, 8
q = torch.randn(B, H, T, Dh)
k = torch.randn(B, H, T, Dh)
v = torch.randn(B, H, T, Dh)

causal = torch.tril(torch.ones(T, T, dtype=torch.bool))[None, None]
tokens = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0]])
padding = (tokens != 0)[:, None, None, :]
visible = causal & padding

scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)
weights = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
weights = torch.nan_to_num(weights, nan=0.0)
output = weights @ v

assert visible.shape == (B, 1, T, T)
assert output.shape == (B, H, T, Dh)
assert torch.count_nonzero(weights.masked_select(~visible)) == 0
```

## 本章调试不变量

1. `scores.shape == [B,H,Tq,Tk]`，Softmax 始终沿 `Tk`。
2. Mask 的最后一维长度等于 key length，而不是 query length 的偶然同值。
3. 有效 query 至少存在一个可见 key。
4. Mask、Q/K/V 位于同一 device，布尔或 additive dtype 符合 API。
5. Dropout 前有效行和为 1；Dropout 后不要强制该断言。
6. Cached 与 full forward 对同一 token 的输出应在浮点误差内一致。

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
