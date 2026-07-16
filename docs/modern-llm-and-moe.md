# 现代 LLM 组件与 MoE

经典 Transformer 给出了注意力、FFN、残差和归一化的骨架。现代 LLM 通常不改变
next-token prediction 这个核心目标，而是替换位置表示、归一化、FFN、注意力缓存和参数
组织方式，以提高稳定性、吞吐、上下文长度和参数容量。

## 1. 从经典 Transformer 到现代 Decoder-Only LLM

常见变化：

| 经典实现 | 现代 LLM 常见选择 | 主要目的 |
| --- | --- | --- |
| LayerNorm | RMSNorm | 更简单，减少均值计算 |
| Sinusoidal PE | RoPE | 把相对位置信息注入 Q/K |
| ReLU FFN | SwiGLU | 更强的门控非线性 |
| Multi-Head Attention | GQA / MQA | 减小 KV Cache |
| 重算全部历史 | KV Cache | 加速逐 token 解码 |
| 普通 attention kernel | FlashAttention | 降低显存读写 |
| Dense FFN | Sparse MoE | 增加参数容量但控制单 token 计算量 |

运行现代 block 实验：

```bash
python labs/lab07_modern_blocks.py
```

## 2. RMSNorm

LayerNorm 同时减均值、除标准差：

```text
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

RMSNorm 只按均方根缩放：

```text
RMS(x) = sqrt(mean(x^2) + eps)
RMSNorm(x) = weight * x / RMS(x)
```

它不做中心化，参数通常只有一个缩放向量。两者都在最后一个 hidden 维度上计算，不混合
token。RMSNorm 更简单，但不能据此断言对所有任务都更好；它是现代 LLM 中经过大规模
验证的常见工程选择。

## 3. RoPE

正弦位置编码把位置向量加到 hidden state。RoPE 则把 Q/K 的相邻维度成对旋转：

```text
[x_2i, x_2i+1] -> 旋转角度 theta(pos, i)
```

二维旋转：

```text
x' = x cos(theta) - y sin(theta)
y' = x sin(theta) + y cos(theta)
```

旋转保持向量范数。更重要的是，旋转后的 Q/K 点积自然包含位置差，使注意力更容易表达
相对距离。

RoPE 一般只作用于 Q 和 K，不作用于 V。上下文外推仍不是自动保证的；base、频率缩放和
训练长度都会影响长上下文效果。

## 4. SwiGLU

经典 FFN：

```text
FFN(x) = W_down activation(W_up x)
```

SwiGLU 增加门控分支：

```text
SwiGLU(x) = W_down (SiLU(W_gate x) * (W_up x))
```

`*` 是逐元素乘法。gate 决定哪些特征通过，up 分支提供候选特征。为了让总参数量与普通
FFN 接近，实际模型会相应调整 hidden dimension，而不是机械地沿用同一个 `4D`。

## 5. KV Cache

### 5.1 为什么需要缓存

生成第 `t` 个 token 时，历史 token 的 K/V 与前一步完全相同。如果每一步都把整个
prompt 重新投影，会产生大量重复计算。

KV Cache 保存每层历史：

```text
K_cache: [B, Hkv, T, Dh]
V_cache: [B, Hkv, T, Dh]
```

新一步只计算当前 token 的 Q/K/V，把新 K/V 追加到 cache，然后当前 Q 读取全部历史 K/V。

```bash
python labs/lab06_kv_cache.py
```

实验会证明：在相同权重和数值精度下，逐 token cached attention 与完整 causal attention
输出一致。

### 5.2 Cache 的代价

KV Cache 随层数、上下文长度、batch 和 KV head 数线性增长。长上下文服务中，它往往比
模型临时激活更先成为容量瓶颈。

## 6. MHA、MQA 和 GQA

### MHA

每个 query head 都有自己的 K/V head：

```text
Hq = Hkv = H
```

### MQA

所有 query head 共享一组 K/V：

```text
Hkv = 1
```

KV Cache 最小，但表达能力和训练行为可能受影响。

### GQA

多个 query head 共享一组 K/V，位于两者之间：

```text
1 < Hkv < Hq
```

GQA 常用于在质量和推理内存之间折中。Query head 数决定输出拆分，KV head 数决定 cache
大小，二者不能混为一个概念。

三者可以由同一个参数化实现表达：

```text
num_query_heads = Hq
num_kv_heads = Hkv

Hkv = Hq -> MHA
Hkv = 1  -> MQA
1 < Hkv < Hq -> GQA
```

K/V projection 只产生 `Hkv` 个 head；attention 计算前，把每个 K/V head 共享给对应的一组
query heads。KV Cache 每 token、每层的元素数为：

```text
2 * Hkv * Dh
```

因此同样 `Hq=8` 时，MHA/GQA-2/MQA 的 K/V cache 比例是 `8:2:1`。

```bash
python labs/lab10_mha_mqa_gqa.py
```

GQA 论文把它定义为 MHA 与 MQA 的插值：每组 query heads 共享单个 key head 和 value
head；`GQA-1` 等价于 MQA，group 数等于 query head 数时等价于 MHA。

## 7. FlashAttention 解决什么

FlashAttention 不改变精确 attention 数学结果。它通过分块计算和 online softmax，避免
把完整 `[T, T]` score/weight 矩阵频繁写入显存，再从显存读回。

关键收益来自更少的高带宽内存访问，而不只是减少浮点运算。它仍然是 dense attention，
理论连接数量仍为 `T^2`。

## 8. 从普通 Dense MoE 到 Sparse MoE

Dense Transformer 的每个 token 都通过同一个 FFN。Mixture-of-Experts 把一个 FFN 换成
多个专家。最直观的 Dense MoE 会计算所有 expert，再按 router 概率加权：

```text
y = sum_i softmax(router(x))_i * Expert_i(x)
```

它能验证“不同 token 使用不同 expert 权重”的原理，但每个 token 仍执行全部 experts，
计算量随 expert 数线性增加。

Sparse MoE 只执行 Top-k experts：

```text
Expert_1, Expert_2, ..., Expert_E
```

Router 为每个 token 计算专家分数：

```text
router_logits = x W_router       # [tokens, E]
router_probs = softmax(logits)   # [tokens, E]
```

只选择 Top-k 专家：

```text
indices, weights = topk(router_probs, k)
weights = weights / sum(weights)
```

最终输出：

```text
y_token = sum_j weight_j * Expert_j(x_token)
```

运行：

```bash
python labs/lab08_moe_routing.py
python labs/lab11_moe_variants.py
```

`lab11` 把 Dense MoE、Top-k Sparse MoE 和 shared-expert Sparse MoE 放在同一输入上对比。

## 9. 为什么 MoE 能增加参数但不同比例增加计算

假设有 8 个专家，每个专家都和原 dense FFN 一样大，Top-2 路由：

- 总 expert 参数约变为 8 倍。
- 每个 token 只执行 2 个专家，expert 计算约为 dense FFN 的 2 倍，而不是 8 倍。

因此 MoE 的卖点是“参数容量”和“每 token 激活参数量”解耦。它不是免费计算：router、
dispatch、跨设备通信、padding/capacity 浪费和两个专家的执行都要成本。

## 10. Dispatch 与 Combine

实际 MoE 层包含：

```text
tokens
  -> router top-k
  -> 按 expert 重排/发送 token
  -> 各 expert 批量计算
  -> 按原 token 顺序合并
  -> top-k 权重加权求和
```

单机实验可以用循环和 `index_add`。大模型会做 expert parallel：不同设备持有不同专家，
token 通过 all-to-all 通信发送到目标设备。通信不均衡会直接拖慢整层。

## 11. Capacity 与 Dropped Tokens

若大量 token 都选择同一个专家，该专家 batch 很大，其他专家空闲。系统通常为每个专家
设置容量：

```text
capacity ~= capacity_factor * tokens * top_k / num_experts
```

超过容量的 token 可以被丢弃、转到备选专家，或使用无丢弃实现等待处理。不同策略影响
训练稳定性、吞吐和模型质量。

## 12. 负载均衡辅助 Loss

只优化语言模型 loss 时，router 可能塌缩到少数专家。常见辅助目标让“概率重要性”和
“实际路由负载”更均匀。一个简化形式：

```text
importance_e = mean(router_prob_e)
load_e = fraction(top1_route == e)
L_balance = E * sum_e importance_e * load_e
```

还可能加入 router z-loss，限制 router logits 过大。辅助 loss 系数过小无法防塌缩，过大
又会强迫不必要的均匀路由，必须通过训练曲线和专家负载共同判断。

## 13. Shared-Expert Sparse MoE

DeepSeekMoE 区分两类专家：

- Shared experts：不进入 Top-k 竞争，对每个 token 始终激活。
- Routed experts：由 router 计算 affinity，只激活 Top-k。

教学化公式：

```text
moe(x) = sum_s SharedExpert_s(x)
       + sum_{i in TopK(x)} gate_i(x) * RoutedExpert_i(x)

block_output = x + moe(norm(x))
```

Shared expert isolation 的动机是让共享专家承载跨上下文的通用知识，减少 routed experts
之间重复学习相同通用能力，从而让 routed experts 更专门化。Shared experts 会增加每个
token 固定执行的计算量，所以不能把它们计入“只激活 Top-k 个专家”的数字里。

DeepSeekMoE 还强调 fine-grained expert segmentation：在总参数和激活参数预算可比时，把
expert 切得更细，并激活更多个较小 routed experts，以提供更灵活的知识组合。它与
shared expert isolation 是两项不同设计。

```bash
python labs/lab11_moe_variants.py
```

`SharedExpertSparseMoE` 对所有 token 执行 shared experts，再只 dispatch Top-k routed
experts。代码把残差放在模块外，便于明确 MoE 是 FFN 子层的替代品。

## 14. MoE 常见失败模式

### Router collapse

少数专家接收绝大多数 token。检查每层 expert token count、概率熵和辅助 loss。

### Expert under-training

部分专家长期没有足够 token，参数几乎不学习。增加数据多样性或调整均衡策略。

### Capacity overflow

热门专家超容量导致 token 丢弃或吞吐突降。

### 通信成为瓶颈

理论 FLOPs 很好看，但 all-to-all 占据主要时间。需要结合硬件拓扑和并行策略评估。

### 只看总参数量

MoE 的总参数、激活参数、KV Cache、通信量和实际 tokens/s 是不同指标，不能只报一个
“模型有多少 B 参数”。

## 15. MoE 与 Attention 的关系

MoE 通常替换 FFN，不替换 attention：

```text
x = x + Attention(Norm(x))
x = x + MoE(Norm(x))
```

Attention 负责 token 间通信；MoE expert 仍然逐 token 变换，只是不同 token 可以选择不同
FFN 参数。也存在 attention experts 等研究变体，但不是最常见的基础结构。

## 16. LoRA：从预训练到后训练

LoRA 不改变 Transformer 主干，而是在选定线性层旁增加低秩更新：

```text
y = xW + (alpha / rank) xAB
```

冻结 `W`，只训练小矩阵 `A/B`。常见目标包括 Q/V projection，也可扩展到 K/O 和 FFN。

```bash
python labs/lab09_lora_linear.py
```

实验包含三个重要断言：

1. `B=0` 时 LoRA 初始输出与基座完全一致。
2. 训练参数只来自低秩分支。
3. `W + Delta-W` 融合后与动态 LoRA 输出一致。

LoRA 属于参数高效后训练；MoE 属于模型架构和容量组织。二者可以同时存在，例如给 MoE
模型的 attention 或 expert projection 注入 LoRA。

## 17. 推荐学习路线

```text
Attention / Mask / MHA
  -> Pre-LN / FFN
  -> Decoder-Only next-token training
  -> KV Cache
  -> RMSNorm + RoPE + SwiGLU
  -> GQA / FlashAttention 原理
  -> Top-k MoE router
  -> load balance / capacity / expert parallel
  -> LoRA / SFT / preference optimization
```

每一步先验证 shape 和等价性，再谈大规模训练。现代 LLM 的复杂性大多来自这些模块在分布式
训练、长上下文和服务系统中的组合，而不是单个公式本身。

## 18. 原始资料

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)
