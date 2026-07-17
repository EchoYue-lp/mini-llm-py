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

实验还会打印 split 后 Tensor 的 stride/contiguous 状态，以及包含 bias 的 Q/K/V/O 精确
参数量，用于区分“逻辑轴重排”和“物理内存布局”。

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

## 一个大投影如何等价于多组 Head 投影

实现通常不是创建 `H` 个独立 `nn.Linear(D,Dh)`，而是一次计算：

```python
q = q_proj(x)  # [B,T,D]
q = q.view(B, T, H, Dh)
```

从权重行的角度，`Wq: [D,D]` 可以按输出通道分块：

```text
Wq = concat(Wq_head_0, ..., Wq_head_H-1)
每块 shape [Dh,D]
```

因此一次大 GEMM 与多次小投影在数学上可对应，但大矩阵乘法通常更利于硬件吞吐。拆 head
只是重新解释输出通道，不会在此时复制参数。

注意：这不表示不同 head 正交或独立。所有 head 都读取同一个输入 `x`，训练目标也共同
作用于最终输出。

## Stride、Transpose 与 Contiguous

Tensor 除 shape 外还有 stride，描述每个轴移动一步要跨过多少个存储元素。示例：

```python
x = torch.arange(2 * 3 * 4).view(2, 3, 4)
y = x.transpose(1, 2)
print(x.shape, x.stride())
print(y.shape, y.stride())
```

`transpose` 通常只创建 view，底层数据没有按新顺序复制，因此 `y` 的逻辑相邻元素在内存中
可能不相邻。`view` 要求新 shape 能由现有 stride 表示；合并 head 时常见写法：

```python
context = context.transpose(1, 2).contiguous()
output = context.view(B, T, H * Dh)
```

`.contiguous()` 会按当前逻辑顺序生成连续副本。`reshape` 在可能时返回 view，不可能时会
复制，因此更宽容，但也可能隐藏一次额外内存开销。教学实现显式写 `contiguous().view()`
更容易看清语义和成本。

## Output Projection 不只是恢复 Shape

Merge 后：

```text
concat(head_0, ..., head_H-1): [B,T,D]
```

如果直接返回，来自不同 head 的通道只是并排放置。`Wo: [D,D]` 允许每个输出特征读取所有
head 的通道：

$$
y_{b,t,m}=\sum_{h,d} context_{b,h,t,d}\,W^O_{m,(h,d)}
$$

它同时完成：

1. 跨 head 信息混合。
2. 把输出映射回残差主干使用的表示基底。
3. 保持 `[B,T,D]`，使结果能与输入相加。

因此 `out_proj` 不是可随意省略的 reshape 辅助层。

## 参数量与 FLOPs 分开计算

参数量忽略 bias：

```text
Q/K/V/O projections = 4D^2
```

Self-Attention 长度为 `T` 时，主要乘法量级：

```text
Q/K/V projection: 3 * B * T * D^2
QK^T:             B * T^2 * D
weights @ V:      B * T^2 * D
output projection:B * T * D^2
```

合计近似：

```text
4BTD^2 + 2BT^2D
```

这说明：

- 短序列、大 `D` 时 projection 可能占主要计算。
- 长序列时 `T^2D` 项迅速主导。
- 增加 head 数但保持 `D` 不变，理论乘法总量近似不变；实际速度仍受 kernel shape、并行度
  和内存布局影响。

Cross-Attention 将二次项改为 `Ttarget * Ssource * D`。

## Self-Attention 的 Q/K/V 长度不变量

本项目 `MultiHeadAttention.forward(Q,K,V,mask)` 允许三者来源不同。应检查：

```text
Q batch == K batch == V batch
K length == V length
Q/K/V hidden dimension == D
```

但不要求：

```text
Q length == K length
```

Self-attention 恰好相等，cross-attention 可以不同。输出 token 数永远跟随 Q：

```text
output shape = [B,Tq,D]
```

因为每个 query 产生一个输出；K/V 只提供被读取的 memory。

## 逐行 Shape Instrumentation

调试时可以把 shape 断言直接写进最小实现：

```python
def split_heads(x, heads):
    batch, seq, d_model = x.shape
    assert d_model % heads == 0
    head_dim = d_model // heads
    result = x.view(batch, seq, heads, head_dim).transpose(1, 2)
    assert result.shape == (batch, heads, seq, head_dim)
    return result

def merge_heads(x):
    batch, heads, seq, head_dim = x.shape
    result = x.transpose(1, 2).contiguous().view(
        batch, seq, heads * head_dim
    )
    assert result.shape == (batch, seq, heads * head_dim)
    return result

x = torch.randn(2, 5, 12)
assert torch.equal(merge_heads(split_heads(x, 3)), x)
```

完整 forward 可记录：

```text
input Q/K/V
projected Q/K/V
split Q/K/V
scores
weights
context per head
merged context
output
```

只打印最终 output shape 无法定位中间轴交换错误。

## Head 冗余与诊断

多头提供表达机会，不保证每个 head 都有效。实际模型可能出现：

- 多个 head 学到高度相似的 attention pattern。
- 某些 head 权重近似均匀。
- 某些 head 长期只关注固定位置或特殊 token。
- 剪除少量 head 后质量变化很小。

仅凭 attention heatmap 不能判断 head 是否重要。更可靠的诊断需要结合 ablation、输出变化、
梯度或任务指标。标准 MHA 也没有正交约束，因此不要把 `H` 直接解释成 `H` 种独立知识。

## 本章调试不变量

1. `D == H * Dh`，所有 Q/K/V 投影输出最后一维都为 `D`。
2. Split 后轴顺序严格为 `[B,H,T,Dh]`。
3. K/V 的 sequence length 相同，输出 length 跟随 Q。
4. Score 最后两维为 `[Tq,Tk]`，Mask key 轴等于 `Tk`。
5. Merge 前先恢复 `[B,T,H,Dh]` 的逻辑顺序。
6. Output projection 后 shape 回到 `[B,Tq,D]`，可与 residual 相加。
7. 参数量统计使用 Parameter 实际 shape，不把 head 数重复乘入。

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
