# 10 MoE Router、Capacity 与专家

Mixture-of-Experts 通常替换 Transformer 的 FFN 子层，不替换 Attention。

## 学习目标

读完后应能：

1. 手算 Top-K 路由权重。
2. 区分总参数、激活参数和实际计算。
3. 计算 expert capacity 并解释 overflow。
4. 识别 router collapse、负载不均和通信瓶颈。

## 本章符号与 Shape

| 符号 | 含义 |
| --- | --- |
| `B,T,D` | 输入 batch、token 长度、hidden width |
| `N` | 路由 token 数，通常 `N=B*T`，排除项必须明确 |
| `E` | routed expert 数量 |
| `K` | 每个 token 选择的 expert 数 |
| `F` | 单个 expert 的 FFN 中间宽度 |
| `C` | 每个 expert 的 capacity |

Router 接收 flatten 后的 `[N,D]`，输出概率 `[N,E]`；Top-K indices/weights 为 `[N,K]`；
combine 后必须恢复原来的 `[B,T,D]`。`N*K` 是 assignment 数，不等于唯一 token 数。

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

`lab08` 还会输出 router entropy、Top-1 count、balance-loss 基线和 router gradient norm；
`lab11` 会验证只有实际收到 token 的 expert 才产生 expert 参数梯度。

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

## Top-K 选择的梯度路径

`topk` 返回离散 expert index。Index 在局部范围内是分段常量，不能像普通连续函数一样对
“选择了哪个 expert”求导。训练信号主要通过：

- 已选 expert 的 combine weight。
- Router Softmax probability。
- Load-balance、z-loss 等辅助目标。
- 被选 expert 的输出对主任务 loss 的贡献。

未选 expert 不执行该 token，也不会从该 token 获得 expert 参数梯度。Router logit 只有在
选择边界变化或连续权重/辅助损失路径中才改变路由行为。

这解释了 router collapse 为什么可能自我强化：热门 expert 获得更多 token 和训练信号，
冷门 expert 更难变好。辅助损失、router noise 或初始化用于打破这一反馈。

## Top-K 重新归一化的语义

原 router probability 对全部 `E` 个 expert 求和为 1。只保留集合 `S` 后：

$$
\hat p_e=\frac{p_e}{\sum_{j\in S}p_j},\quad e\in S
$$

重新归一化使 routed mixture 权重和为 1。若不归一化，输出幅值还会乘以“Top-K 原始概率
质量”，相当于引入额外 gate 强度。

本项目：

- `TopKMoE` 和普通 `SparseMoE` 默认重新归一化。
- `SharedExpertSparseMoE` 内部 routed 分支设置 `renormalize_topk=False`，再与 always-on
  shared output 相加。

两种都是可能设计，但不能只凭 `top_k=2` 假设 combine 权重和一定为 1。

## Dispatch/Combine 的索引不变量

展平 token 后：

```text
tokens:      [N,D]
top_indices: [N,K]
top_weights: [N,K]
```

对 expert `e`：

```python
token_indices, choice_indices = torch.where(top_indices == e)
selected_inputs = tokens[token_indices]
selected_outputs = expert(selected_inputs)
weighted = selected_outputs * top_weights[
    token_indices, choice_indices
].unsqueeze(-1)
output.index_add_(0, token_indices, weighted)
```

同一个 token 会在 Top-K 中出现 K 次，`index_add_` 把多个 expert 贡献加回同一原始位置。
必须同时保留 token index 和该 expert 在 Top-K 中的 choice index，否则会取错权重。

## Capacity 的整数定义

总 assignment 数是 `N*K`。理想均匀负载为：

$$
\frac{NK}{E}
$$

常见容量：

$$
C=\left\lceil capacity\_factor\frac{NK}{E}\right\rceil
$$

还需要明确 capacity 是每个 expert、每个设备、每个 microbatch 还是整个 global batch 计算。
梯度累积不会自动把多个 microbatch 的 dispatch 合并；每次 forward 的容量利用率可能不同。

Top-2 的 second-choice assignment 也占容量。只按 token 数 `N` 而忘记乘 `K`，会把容量估小。

## Balance Loss 的基线

本项目简化形式：

$$
L_{balance}=E\sum_e importance_e\,load_e
$$

若 probability 和 Top-1 load 都完全均匀：

```text
importance_e = 1/E
load_e = 1/E
L_balance = 1
```

若所有 token 都路由到一个 expert，且 probability 也高度集中，loss 可接近 `E`。因此该辅助
loss 的绝对值不是“越接近 0 越好”；应知道公式基线并乘上配置中的辅助系数后再解释。

不同论文的 balance loss 定义、Top-1/Top-K load、是否跨设备聚合可能不同，不能只比较同名
指标。

## Router Z-Loss 的公式

一种常见形式：

$$
L_z=\frac{1}{N}\sum_n\left(\log\sum_e e^{z_{n,e}}\right)^2
$$

它惩罚 router `logsumexp` 过大，限制 logit 整体尺度。Softmax 对共同平移不敏感，但浮点
计算和优化器会受大 logits 影响，所以 z-loss 能改善数值行为。

Z-loss 不直接保证负载均匀；balance loss 也不直接限制 logits 绝对尺度，两者职责不同。

## Router 精度、噪声与抖动

Top-K 对接近的 logits 很敏感。低精度舍入可能改变 expert 排名，导致路由离散跳变。实践中
常让 router logits/Softmax 使用 FP32，即使 expert FFN 使用 BF16/FP16。

训练期还可能加入：

- Input jitter。
- Router logit noise。
- 随机 second-expert 策略。

这些机制用于探索和负载均衡，评估/推理通常关闭。忘记区分 train/eval 会让路由不可复现。

## 参数量与激活计算公式

若单个 expert FFN 参数为 `Pexpert`：

```text
总 expert 参数 = E * Pexpert
每 token routed expert 计算约 = K * expert_compute
```

Shared expert 另加固定项：

```text
active experts/token = num_shared + K
```

Router 参数约 `D*E`，通常小于 expert 总参数，但 router Softmax、Top-K、dispatch、padding 到
capacity 和通信都增加额外时间。稀疏激活降低的是 expert FFN 计算，不会降低 Attention、
embedding 或 KV Cache。

## Expert Parallel 的通信量直觉

若 token 路由到远端设备，dispatch 至少发送 hidden state `[D]`，combine 再返回 expert 输出
`[D]`。粗略每 assignment 通信量与：

```text
2 * D * bytes_per_element
```

成正比，还不含元数据、padding、网络协议和同步。Top-K 增大既增加 expert 计算，也增加潜在
通信。负载不均会产生 straggler：所有设备等待最忙 expert 完成。

## 可运行的路由健康检查

```python
def route_metrics(probabilities, top_indices, num_experts):
    top1 = top_indices[:, 0]
    counts = torch.bincount(top1, minlength=num_experts)
    fractions = counts.float() / top1.numel()
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1)
    return {
        "counts": counts,
        "fractions": fractions,
        "mean_entropy": entropy.mean(),
        "unused_experts": (counts == 0).sum(),
    }

tokens = x.reshape(-1, x.size(-1))
probabilities = torch.softmax(moe.router(tokens).float(), dim=-1)
top_indices = probabilities.topk(moe.top_k, dim=-1).indices
metrics = route_metrics(probabilities, top_indices, moe.num_experts)
print(metrics)
```

应按层记录这些指标。全模型合并统计可能掩盖某一层已经 collapse。

## 本章调试不变量

1. `top_indices/top_weights` shape 为 `[N,K]`，每个 token 恰有 K 个 assignment。
2. Combine 使用正确 token index 与 choice index，输出恢复原 `[B,T,D]`。
3. 明确 Top-K 是否重新归一化，shared expert 是否另行相加。
4. Capacity 按 `N*K/E` 计算并记录 overflow/dropped 数。
5. Balance loss 解释基于其理论基线，不假设最优值为 0。
6. Router logits 精度、噪声和 train/eval 行为明确。
7. 每层分别监控 count、entropy、overflow、grad norm 和通信时间。

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
