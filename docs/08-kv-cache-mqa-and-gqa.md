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
