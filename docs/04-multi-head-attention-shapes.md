# 04 Multi-Head Attention 的 Shape

Multi-Head Attention 将隐藏维拆到多个子空间中并行计算。

## 学习目标

读完后应能：

1. 在任意 `B,T,D,H` 下推导所有中间 shape。
2. 解释 split、transpose、merge 和 output projection。
3. 估算 Multi-Head Attention 参数量。
4. 区分 self-attention 与 cross-attention 的长度维度。

## 拆分

```text
[B, T, D]
  -> [B, T, H, Dh]
  -> [B, H, T, Dh]
```

约束：

```text
D = H * Dh
D % H == 0
```

每个 head 独立计算 Scaled Dot-Product Attention。

## 合并

```text
[B, H, T, Dh]
  -> [B, T, H, Dh]
  -> [B, T, D]
  -> output projection
```

合并前必须先把 `T` 和 `H` 调回正确顺序，并在必要时调用 `contiguous()`，
否则 reshape 可能读取错误的内存布局。

## 多头的意义

多个 head 可以学习不同关系，例如：

- 局部搭配。
- 长距离依赖。
- 位置模式。
- 句法或指代关系。

这些语义不是预先指定的，也不能保证每个 head 都有清晰的人类解释。

## Self-Attention 与 Cross-Attention

Self-Attention：

```text
Q, K, V 都来自同一序列
```

Cross-Attention：

```text
Q 来自 decoder
K, V 来自 encoder memory
```

二者数学形式相同，数据来源不同。

## 实验

```bash
python -m labs.lab02_multi_head_attention
```

建议修改 `d_model` 和 `num_heads`，验证合法与非法组合。

## 对照源码

- `labs/lab02_multi_head_attention.py`
- `models/layers.py::MultiHeadAttention`

## 一个完整 Shape 推导

设：

```text
B = 2
T = 5
D = 12
H = 3
Dh = D / H = 4
```

输入：

```text
x: [2, 5, 12]
```

Q/K/V 投影后仍为：

```text
q, k, v: [2, 5, 12]
```

拆 head：

```text
[2, 5, 12]
-> [2, 5, 3, 4]
-> [2, 3, 5, 4]
```

Score：

```text
q @ k.transpose(-2, -1)
[2, 3, 5, 4] @ [2, 3, 4, 5]
-> [2, 3, 5, 5]
```

乘 V：

```text
[2, 3, 5, 5] @ [2, 3, 5, 4]
-> [2, 3, 5, 4]
```

合并：

```text
[2, 3, 5, 4]
-> [2, 5, 3, 4]
-> [2, 5, 12]
```

最终 output projection 不改变最后 shape。

## Q/K/V 投影为什么需要三组参数

同一个 hidden state 需要同时扮演三种角色：

- 作为 query 发出检索需求。
- 作为 key 提供匹配特征。
- 作为 value 提供被汇总的信息。

三组独立线性层让模型学习不同表示。若直接令 Q=K=V=x，仍可计算注意力，但表达能力受限。

## 参数量估算

忽略 bias：

```text
Wq: D * D
Wk: D * D
Wv: D * D
Wo: D * D
总计约 4D^2
```

增加 head 数通常不会改变这四个投影的总参数量，因为 `D` 没变，只是重新分组。

## Head 数不是越多越好

固定 `D` 时：

```text
Dh = D / H
```

Head 越多，每个 head 越窄。过窄可能限制单个 head 的表达；过少又减少并行子空间。
Head 数是结构超参数，不是单向增加就更强。

## Cross-Attention 的 Shape

设 decoder 长度 `T`、source 长度 `S`：

```text
Q: [B, H, T, Dh]
K: [B, H, S, Dh]
V: [B, H, S, Dh]
score: [B, H, T, S]
```

Score 最后两维不再是方阵，因为 query 和 key 来自不同序列。

## 实现步骤

```python
q = q_proj(query)
k = k_proj(key)
v = v_proj(value)

q = split_heads(q)
k = split_heads(k)
v = split_heads(v)

scores = q @ k.transpose(-2, -1)
weights = softmax(apply_mask(scores))
context = weights @ v

output = out_proj(merge_heads(context))
```

阅读实现时逐行标注 shape，能快速发现 transpose 或 reshape 错误。

## 常见错误

1. `D % H != 0`。
2. 拆 head 后忘记交换 `T` 与 `H`。
3. 合并前未恢复连续内存。
4. Score 转置错成 query 的最后两维。
5. Cross-attention 假设 `T == S`。
6. Mask 使用了错误的 key length。

## 动手练习

1. 用 `B=2,T=5,D=12,H=3` 打印每一步 shape。
2. 改成 `H=4`，重新推导 `Dh`。
3. 尝试 `H=5`，解释为什么失败。
4. 让 source length 与 target length 不同，运行 cross-attention。

## 自测

1. 为什么 head 数变化时投影参数量通常不变？
2. Attention score 的最后两维分别代表什么？
3. Cross-attention 为什么可能得到 `[T,S]` 而不是 `[T,T]`？
4. `transpose` 和 `reshape` 各自解决什么问题？
