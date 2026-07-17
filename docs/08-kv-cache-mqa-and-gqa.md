# 08 KV Cache、MHA、MQA 与 GQA

## 学习目标

读完后应能：

1. 区分 prefill 与 decode。
2. 计算指定模型配置的 KV Cache 内存。
3. 解释 MHA、MQA、GQA 的 head 共享关系。
4. 区分 KV Cache、GQA 和 FlashAttention 解决的问题。

## KV Cache 解决什么问题

生成第 `t` 个 token 时，历史 token 的 K/V 与前一步相同。若每一步都重新投影完整
prompt，会产生大量重复计算。

每层缓存：

```text
K_cache: [B, Hkv, T, Dh]
V_cache: [B, Hkv, T, Dh]
```

新一步只计算当前 token 的 Q/K/V，将新 K/V 追加到 cache，然后当前 Q 读取完整历史。

## 等价性

在相同权重、mask 和数值精度下：

```text
完整 causal attention 输出
==
逐 token cached attention 输出
```

```bash
python -m labs.lab06_kv_cache
```

实验逐步断言 cache length 从 1 增长到 T，并打印有无 cache 时需要执行 K/V projection 的
token position 数 `T(T+1)/2` 与 `T`。

若结果不一致，优先检查：

- cache 拼接维度。
- position index。
- causal mask。
- 当前 token 是否重复写入。

## Cache 的代价

KV Cache 随以下变量线性增长：

- 层数。
- batch size。
- 上下文长度。
- KV head 数。
- head dimension。

每 token、每层的 K/V 元素数：

```text
2 * Hkv * Dh
```

长上下文服务中，KV Cache 常成为主要容量瓶颈。

## MHA

每个 query head 都有独立 K/V：

```text
Hq = Hkv = H
```

表达能力强，但 cache 最大。

## MQA

所有 query heads 共享一组 K/V：

```text
Hkv = 1
```

Cache 最小，但共享程度最高。

## GQA

一组 query heads 共享一个 K/V head：

```text
1 < Hkv < Hq
```

GQA 在质量与推理内存之间折中：

```text
Hkv = Hq -> MHA
Hkv = 1  -> MQA
其他值   -> GQA
```

若 `Hq=8`，MHA、GQA-2、MQA 的 K/V cache 比例为：

```text
8 : 2 : 1
```

## Head 共享

K/V projection 只产生 `Hkv` 个 head。计算 attention 前，每个 K/V head 被复制或
广播给对应的 query head 组。Query head 数决定输出拆分，KV head 数决定 cache 规模。

```bash
python -m labs.lab10_mha_mqa_gqa
```

实验同时比较三种结构的参数量、compact cache elements/token，以及教学用逻辑展开后的 K
元素数，避免把 `repeat_interleave` 误认为真实 cache 必须复制。

## FlashAttention

FlashAttention 不改变精确 attention 数学结果。它通过分块与 online softmax，减少完整
`[T, T]` score/weight 矩阵在高带宽显存中的写入和读取。

它降低的是内存访问开销，不会把 dense attention 的理论连接数从 `T^2` 变成线性。

## 对照源码

- `labs/lab06_kv_cache.py`
- `labs/lab10_mha_mqa_gqa.py`

## 原始资料

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

## Prefill 与 Decode

LLM 推理通常分两阶段：

### Prefill

一次处理完整 prompt：

```text
[B, prompt_length] -> 为所有层生成 K/V cache
```

Prefill 可以并行处理 prompt 的所有位置，主要受计算吞吐影响。

### Decode

每次只输入一个新 token：

```text
[B, 1] -> 追加一格 K/V -> 预测下一个 token
```

Decode 串行执行，主要受 cache 读取、内存带宽和小矩阵计算影响。

因此“首 token 延迟”和“后续 tokens/s”是两个不同指标。

## Cache 内存的具体计算

近似公式：

```text
bytes =
  2                    # K 和 V
  * num_layers
  * batch_size
  * num_kv_heads
  * sequence_length
  * head_dim
  * bytes_per_element
```

例子：

```text
layers = 32
batch = 1
Hkv = 8
T = 4096
Dh = 128
dtype = fp16 = 2 bytes
```

结果约为：

```text
2 * 32 * 1 * 8 * 4096 * 128 * 2
= 536,870,912 bytes
~= 512 MiB
```

若 MHA 使用 `Hkv=32`，同条件约为 2 GiB。GQA 可以直接降低 cache 占用。

## Cache 的存储布局

常见布局：

```text
[B, Hkv, T, Dh]
```

也可能为了 kernel 或分页管理使用不同布局。代码中 append 的维度必须是 sequence
dimension，不能误拼到 head 或 hidden dimension。

## Beam Search 与 Cache

Beam Search 会把一个样本扩展成多个 beam。每个 beam 的历史 token 不同，因此 cache
也需要：

- 复制。
- 重排。
- 或使用共享前缀与索引管理。

只重排 token 序列而忘记同步 cache，会产生难以发现的错误输出。

## GQA 的共享例子

假设：

```text
Hq = 8
Hkv = 2
```

可以分成：

```text
query heads 0,1,2,3 -> KV head 0
query heads 4,5,6,7 -> KV head 1
```

每个 query head 仍有独立 Q，只共享 K/V。输出最后仍合并 8 个 query heads。

## MQA/GQA 的质量权衡

减少 K/V heads 会降低 cache 和投影参数，但共享过强可能损失部分表达能力。实际选择需要
同时测量：

- 任务质量。
- 首 token 延迟。
- Decode tokens/s。
- 最大 batch。
- 最大上下文。

不能只根据理论 cache 比例判断整体性能。

## FlashAttention 与 KV Cache 的区别

| 技术 | 主要阶段 | 解决问题 |
| --- | --- | --- |
| KV Cache | 自回归 decode | 避免重复计算历史 K/V |
| FlashAttention | Prefill/训练中的 dense attention | 减少 score 矩阵显存读写 |
| GQA/MQA | 训练与推理结构 | 减少 K/V heads 和 cache |

三者可以同时使用，不是互斥方案。

## 为什么缓存 K/V 而不是 Hidden 或 Logits

每一层 Attention 需要该层历史 token 经过该层 `k_proj/v_proj` 的结果。最终 hidden 或 logits
无法替代，因为下一 token 在每一层都要读取不同层的历史表示：

```text
layer 0 cache: K0/V0
layer 1 cache: K1/V1
...
layer L-1 cache: KL-1/VL-1
```

Q 只属于当前 query。历史 Q 在它产生输出后不会被未来 token 读取，因此不需要缓存。未来
token 会生成自己的 Q，再与所有历史 K 配对。

## Cached 与 Full 等价的归纳直觉

对位置 `t`，causal full attention 只能读取 `0..t` 的 K/V。Cached decode 在第 `t` 步也恰好
保存 `0..t`：

```text
K_cache_t = concat(K_0, ..., K_t)
V_cache_t = concat(V_0, ..., V_t)
```

若以下条件一致：

- 每个 token 的 hidden 输入一致。
- position encoding/position id 一致。
- LayerNorm、projection 权重一致。
- Dropout 关闭。
- Mask 可见集合一致。

则第 `t` 步 score 与 full forward 第 `t` 行相同，输出也相同。这个条件是逐层递归的：前一层
cached 输出一致，才能保证下一层输入一致。

## `torch.cat` Cache 的隐藏成本

Lab 便于教学地使用：

```python
key = torch.cat([old_key, new_key], dim=sequence_dim)
```

每次 `cat` 都可能分配新 Tensor 并复制全部历史。生成 `T` 步时，仅 cache append 就可能产生
平方级复制量。生产实现通常：

- 预分配 `[B,Hkv,max_len,Dh]`，按 position 原地写入。
- 使用分页/Paged KV Cache，把逻辑序列映射到固定大小 block。
- 使用专门 kernel 接受 cache pointer 与当前位置。

因此 Lab 验证的是数学等价性，不代表高性能 cache 管理方式。

## RoPE 与 Cache Position

若 RoPE 在写入 cache 前作用于 K，则缓存的是已经按绝对位置旋转的 K。新 token 位置必须从
`cache_length` 开始：

```python
position = past_key.size(sequence_dim)
q = apply_rope(q, position)
k_new = apply_rope(k_new, position)
```

常见错误是 decode 每步输入长度为 1，于是位置也总取 0。Shape 和 cache 长度都正常，但
Attention 相位错误，输出与 full forward 不一致。

## GQA 的投影参数量

设：

```text
Hq query heads
Hkv KV heads
Dh = D / Hq
K/V output width = Hkv * Dh
```

忽略 bias：

```text
Wq: D * D
Wk: D * (Hkv * Dh)
Wv: D * (Hkv * Dh)
Wo: D * D
```

总参数：

$$
2D^2 + 2D(H_{kv}D_h)
$$

MHA 中 `Hkv=Hq`，恢复 `4D^2`。MQA 中 `Hkv=1`，K/V 投影参数和 cache 都显著减少。
FFN、embedding 等参数不受影响，所以不能把 Attention 参数比例直接当作全模型压缩比例。

## Repeat、Broadcast 与物理 Cache

教学实现：

```python
expanded = compact_kv.repeat_interleave(repeats, dim=1)
```

逻辑上得到 `[B,Hq,T,Dh]`，便于复用普通 MHA 公式。但若真正物化 expanded K/V，会抵消
cache 内存优势。生产 GQA kernel 直接让多个 Q head 索引同一个 compact KV head：

```text
kv_head = query_head // queries_per_kv_head
```

应缓存 compact `[B,Hkv,T,Dh]`，只在计算语义上共享，不在存储上复制。

## Decode 为什么常受内存带宽限制

单步 decode 的 Q 长度为 1。每层需要读取全部历史 K/V，但小矩阵计算并不容易占满 GPU：

```text
read cache: O(T * Hkv * Dh)
attention compute: O(T * Hq * Dh)
```

随着 `T` 增长，读取 cache 的字节数线性增长。GQA/MQA 减少 Hkv，直接减少读流量，因此收益
不只体现在“能放更大 batch”，还可能提高 tokens/s。实际速度仍取决于 kernel 是否避免
物化 expanded KV。

## Batch、Beam 与 Cache 内存

公式中的 batch 应使用实际活跃序列数。Beam Search 宽度 `W` 在简单实现中近似把 batch
扩大 `W` 倍：

$$
cache\ bytes\propto B\times W\times T
$$

共享 prompt 前缀、copy-on-write 或 block table 可以减少物理复制，但 beam 分叉后的 token
仍需独立 cache。Beam 重排时至少要同步：

```text
token sequences
sequence lengths
beam scores
K/V block mapping or tensors
finished flags
```

## Cache Quantization 与 Sliding Window

进一步降低内存的策略包括：

- KV cache 使用更低位宽，并保存必要 scale。
- Sliding-window attention 只保留最近 `W` 个 token。
- 对不同层或 token 使用选择性压缩/淘汰。

这些方法会改变数值或可见上下文，不再是纯粹“避免重复计算”的完全等价优化。使用 sliding
window 时，mask 与 position 仍按绝对位置处理，cache 存储长度则被限制。

## 可运行的 Cache 计算器

```python
def kv_cache_mib(
    layers,
    batch,
    kv_heads,
    sequence_length,
    head_dim,
    bytes_per_element=2,
):
    elements = (
        2 * layers * batch * kv_heads * sequence_length * head_dim
    )
    return elements * bytes_per_element / (1024**2)

assert kv_cache_mib(32, 1, 8, 4096, 128) == 512
print("MHA:", kv_cache_mib(32, 1, 32, 4096, 128), "MiB")
print("GQA:", kv_cache_mib(32, 1, 8, 4096, 128), "MiB")
print("MQA:", kv_cache_mib(32, 1, 1, 4096, 128), "MiB")
```

实际服务还要加模型权重、activation、临时 workspace、allocator 碎片和请求调度开销。

## 本章调试不变量

1. 每层都有独立 K/V cache，cache sequence length 等于已处理 token 数。
2. Append/write 发生在 sequence 轴，K/V head 轴保持 `Hkv`。
3. 新 token position id 从 cache length 继续，不重置为 0。
4. Compact cache 不因 GQA head sharing 被物理扩成 `Hq` 份。
5. Cached/full 比较时关闭 dropout，并比较每层或每步最大误差。
6. Beam reorder 同步处理 token、score、length、finished 和 cache。
7. 内存估算使用实际 dtype 字节数、层数、活跃 batch/beam 和最大序列长度。

## 常见错误

1. Cache append 到错误维度。
2. 新 token 的 position id 没有随 cache 长度增长。
3. Beam 重排时没有重排 cache。
4. 把 query head 数当作 KV head 数。
5. 比较 cached/full 输出时 dropout 未关闭。
6. Cache dtype 与模型 dtype 不一致。

## 动手练习

1. 用上面的公式计算 `Hkv=1,2,8,32` 的 cache。
2. 修改 `lab06`，打印每一步 cache shape。
3. 修改 `lab10`，比较 MHA/GQA/MQA 参数量。
4. 在相同输入上断言 cached 与 full 输出误差小于阈值。

## 自测

1. Prefill 与 decode 的性能瓶颈有什么不同？
2. 为什么只缓存 K/V，不缓存 Q？
3. GQA 共享的是什么，保持独立的是什么？
4. FlashAttention 能否代替 KV Cache？
