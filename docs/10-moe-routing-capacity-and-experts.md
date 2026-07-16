# 10 MoE Router、Capacity 与专家

Mixture-of-Experts 通常替换 Transformer 的 FFN 子层，不替换 Attention。

## 学习目标

读完后应能：

1. 手算 Top-K 路由权重。
2. 区分总参数、激活参数和实际计算。
3. 计算 expert capacity 并解释 overflow。
4. 识别 router collapse、负载不均和通信瓶颈。

```text
x = x + Attention(Norm(x))
x = x + MoE(Norm(x))
```

## Dense MoE

Dense MoE 计算全部 experts：

```text
y = sum_i softmax(router(x))_i * Expert_i(x)
```

它适合验证路由原理，但每个 token 仍执行所有 experts，计算量随 expert 数线性增加。

## Sparse Top-K MoE

Router：

```text
router_logits = x W_router
router_probs = softmax(router_logits)
indices, weights = topk(router_probs, k)
```

只执行 Top-K experts：

```text
y_token = sum_j weight_j * Expert_j(x_token)
```

Top-K 权重需要重新归一化。

## 参数容量与激活参数

假设 8 个 experts，Top-2：

- 总 expert 参数约为单个 FFN 的 8 倍。
- 每 token 只执行 2 个 experts。

MoE 将“总参数容量”和“每 token 激活参数量”解耦，但不是免费计算。Router、dispatch、
通信和 capacity 浪费都会带来成本。

## Dispatch 与 Combine

```text
tokens
  -> router top-k
  -> 按 expert 重排或发送
  -> expert 批量计算
  -> 恢复原 token 顺序
  -> 按路由权重合并
```

单机可以使用索引和 `index_add`；分布式 expert parallel 常需要 all-to-all 通信。

## Expert Capacity

热门 expert 可能接收过多 token。常见容量：

```text
capacity ~= capacity_factor * tokens * top_k / num_experts
```

超出容量的 token 可以：

- 丢弃。
- 路由到备选 expert。
- 使用无丢弃实现等待处理。

不同策略会影响吞吐和模型质量。

## 负载均衡 Loss

只优化语言模型 loss 时，router 可能塌缩到少数 experts。简化辅助目标：

```text
importance_e = mean(router_prob_e)
load_e = fraction(top1_route == e)
L_balance = E * sum_e importance_e * load_e
```

辅助系数过小无法防止塌缩，过大会强迫不必要的均匀路由。

## Shared Experts

Shared-expert Sparse MoE 分为：

- Shared experts：所有 token 始终执行。
- Routed experts：只执行 router 选中的 Top-K。

```text
moe(x) = sum SharedExpert(x)
       + sum TopKWeight_i * RoutedExpert_i(x)
```

Shared experts 承载通用知识，routed experts 更容易专门化，但会增加固定计算量。

## 常见失败模式

### Router Collapse

少数 experts 接收绝大多数 token。检查 token count、概率熵和辅助 loss。

### Expert Under-Training

部分 experts 长期没有足够样本，参数几乎不更新。

### Capacity Overflow

热门 experts 超容量，导致 token 丢弃或吞吐下降。

### 通信瓶颈

理论 FLOPs 较低，但 all-to-all 占据主要时间。

### 指标混淆

总参数、激活参数、KV Cache、通信量和 tokens/s 是不同指标，不能只报告参数量。

## 实验

```bash
python -m labs.lab08_moe_routing
python -m labs.lab11_moe_variants
```

`lab11` 在相同输入上比较 Dense MoE、Sparse MoE 与 Shared-Expert Sparse MoE。

## 原始资料

- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)

## 一个 Top-2 路由例子

某个 token 对 4 个 experts 的 router probability：

```text
E0: 0.10
E1: 0.55
E2: 0.25
E3: 0.10
```

Top-2 选择 E1 和 E2。重新归一化：

```text
sum = 0.55 + 0.25 = 0.80
weight(E1) = 0.55 / 0.80 = 0.6875
weight(E2) = 0.25 / 0.80 = 0.3125
```

最终：

```text
output =
  0.6875 * Expert1(x)
  + 0.3125 * Expert2(x)
```

未选中的 experts 不执行该 token 的 FFN。

## Token Flatten

Router 通常先把：

```text
[B,T,D]
```

展平为：

```text
[N,D], N = B*T
```

Router logits：

```text
[N,E]
```

路由、dispatch、combine 完成后再恢复 `[B,T,D]`。

必须保存 token 原始索引，否则 expert 输出无法放回正确位置。

## Capacity 数值例子

假设：

```text
tokens = 100
top_k = 2
experts = 4
capacity_factor = 1.25
```

每个 expert 容量约：

```text
1.25 * 100 * 2 / 4 = 62.5
```

实现通常向上取整为 63。若 E0 收到 80 个 token，就有 17 个超出容量。

## Dropped Token 会发生什么

若直接丢弃超容量 token：

- 该 token 的 MoE 分支可能输出 0。
- Residual 仍可保留原输入。
- 训练信号和模型质量可能受损。

不同实现可能将 token 发送给第二选择、提高 capacity，或使用无丢弃路由。

## 负载均衡的两个量

### Importance

Router 给某 expert 的平均概率。

### Load

实际有多少 token 把该 expert 选为 Top-1 或 Top-K。

概率看似均匀但实际 Top-1 集中，或反过来，都可能造成不平衡，因此辅助 loss 常同时考虑
两者。

## Router Z-Loss

Router logits 过大时 softmax 容易极度饱和。Z-loss 用于约束 logits 的整体尺度，
改善数值稳定性。它与 load balancing loss 目标不同：

- Balance loss 关注 expert 使用是否均匀。
- Z-loss 关注 router logits 是否过大。

## Shared Expert 的计算预算

假设：

```text
1 shared expert
8 routed experts
top_k = 2
```

每个 token 实际执行：

```text
1 shared + 2 routed = 3 experts
```

因此报告“Top-2”时不能忽略 shared expert 的固定计算。

## Expert Parallel

大模型可能让不同设备持有不同 experts：

```text
GPU0: E0,E1
GPU1: E2,E3
GPU2: E4,E5
GPU3: E6,E7
```

Router 选中远端 expert 后，token hidden state 需要通过网络发送。MoE 性能不仅取决于
FLOPs，还取决于 all-to-all 通信、设备拓扑和负载均衡。

## 训练时应监控什么

- 每层每个 expert 的 token count。
- Router probability entropy。
- Capacity overflow 数量。
- Dropped token 比例。
- Balance loss 与主 loss。
- 每个 expert 的梯度范数。
- 通信时间与 expert 计算时间。

只看总训练 loss 无法判断 router 是否健康。

## 推理时的挑战

Batch 中 token 路由到不同 experts，会产生不规则计算。小 batch 下每个 expert 收到的
token 太少，矩阵乘法效率可能很低。因此 MoE 参数量很大不代表单请求延迟一定理想。

## 动手练习

1. 手算上面的 Top-2 权重。
2. 修改 `lab08` 的 router bias，让所有 token 选择同一 expert。
3. 记录每个 expert 的 token count。
4. 在 `lab11` 中增加 shared expert 数，比较参数量和激活计算。
5. 实现一个简单 capacity，并统计 dropped tokens。

## 自测

1. Dense MoE 与 Sparse MoE 的计算差别是什么？
2. Top-K 权重为什么需要重新归一化？
3. Capacity factor 控制什么？
4. Shared expert 是否包含在 Top-K 数字中？
5. 为什么 MoE 在多 GPU 上容易受通信限制？
