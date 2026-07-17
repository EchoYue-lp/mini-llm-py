# 11 LoRA 低秩适配原理

LoRA 在冻结的线性层旁增加低秩更新，只训练少量参数。

## 学习目标

读完后应能：

1. 推导 LoRA A/B 的 shape 和参数量。
2. 解释 A 随机、B 为零时的输出与首步梯度。
3. 分析 rank、alpha、dropout 和目标层的影响。
4. 说明 adapter 保存、加载和 fuse 所需的信息。

## 核心公式

基座线性层：

```text
y = xW
```

LoRA：

```text
y = xW + (alpha / rank) * dropout(x) @ A @ B
```

Shape：

| 参数 | Shape |
| --- | --- |
| `x` | `[batch, seq, in]` |
| `W` | `[out, in]` |
| `A` | `[in, rank]` |
| `B` | `[rank, out]` |
| `x @ A @ B` | `[batch, seq, out]` |

可训练参数从 `in * out` 变为：

```text
in * rank + rank * out
```

当 rank 远小于输入和输出维度时，参数量显著下降。

## 为什么冻结 W

- 基座知识仍参与前向计算。
- 梯度不更新 `W`。
- 优化器只维护 A/B 的状态。
- 多个任务可以共享同一基座模型。

LoRA 学习的是相对基座的任务增量，不是替代基座。

## 为什么 A 随机、B 为零

```text
A = random
B = zero
A @ B = 0
```

训练第 0 步：

```text
y = xW
```

注入 LoRA 不会改变初始输出。若 A/B 都随机，模型会在训练前产生随机偏移；若两者都为
零，初始梯度容易失去有效的对称性破坏。

## Rank、Alpha 与 Dropout

默认示例：

```text
rank = 8
alpha = 16
scale = alpha / rank = 2
dropout = 0
```

### Rank

越大表示能力和参数量越高。常见实验值为 4、8、16。

### Alpha

`alpha / rank` 控制 LoRA 分支强度。调整 rank 时应同时观察 scale。

### Dropout

只作用于 LoRA 分支。小数据过拟合时可以测试 0.05 或 0.1，但必须由验证集决定。

## 目标层

注意力常见投影：

```text
q_proj, k_proj, v_proj, o_proj
```

项目默认从 q/v 开始：

```bash
python -m finetuning.train_lora_short --targets q_proj,v_proj
```

可扩展到：

```bash
python -m finetuning.train_lora_short --targets q_proj,k_proj,v_proj,o_proj
```

目标越多不一定越好，需要同时比较参数量、内存、速度和验证指标。

## Delta-W 与 Fuse

MLX Linear 将权重保存为 `[out, in]`，因此：

```text
Delta-W = scale * B^T @ A^T
W_fused = W + Delta-W
```

动态 adapter：

```text
xW + xAB
```

融合模型：

```text
x(W + Delta-W)
```

两者应在浮点误差范围内一致。

融合后推理图更简单，但失去快速切换 adapter 的能力。

## 实验

```bash
python -m labs.lab09_lora_linear
```

实验验证：

1. B 为零时初始输出与基座一致。
2. 第一训练步 A 梯度为零、B 梯度非零。
3. 只有低秩参数可训练，optimizer 只接收 A/B。
4. `rank(Delta-W) <= configured rank`。
5. Eval 模式下 Fuse 前后输出一致。

## 对照源码

- `labs/lab09_lora_linear.py`
- `finetuning/train_lora_short.py::EducationalLoRALinear`

## 一个参数量例子

假设原线性层：

```text
in = 1024
out = 1024
rank = 8
```

原权重：

```text
1024 * 1024 = 1,048,576 parameters
```

LoRA：

```text
A: 1024 * 8 = 8,192
B: 8 * 1024 = 8,192
总计 = 16,384
```

LoRA 参数约为原层的：

```text
16,384 / 1,048,576 = 1.5625%
```

若只在部分层和部分 projection 上注入，总模型可训练比例还会更低。

## LoRA 不是对 W 做实时低秩分解

容易产生的误解是：

```text
W 被分解成 A 和 B
```

实际是：

```text
W 保持不变
额外学习 Delta-W = BA
```

LoRA 假设任务适配所需的“更新量”近似低秩，不是说预训练权重 W 本身低秩。

## B=0 时第一步谁先学习

LoRA 分支：

```text
x @ A @ B
```

初始 `B=0` 时：

- 对 A 的梯度包含 B，因此第一步通常为 0。
- 对 B 的梯度包含已随机初始化的 A，因此可以非零。

第一步先更新 B，之后 B 不再为零，A 也开始收到梯度。这是 A 随机、B 为零仍能正常
训练的关键。

## Scale 为什么需要显式记录

若只保存 A/B，不记录：

- rank。
- alpha。
- scale。
- 目标层。

推理端无法正确重建 Delta-W。Adapter 不只是 safetensors 文件，还需要结构元数据。

不同实现可能使用：

```text
alpha / rank
alpha / sqrt(rank)
```

或其他缩放约定。加载 adapter 时必须与训练实现一致。

## LoRA 节省了哪些内存

冻结基座后，仍需保存基座权重用于前向，但通常不需要为它保存：

- 梯度。
- Adam 一阶动量。
- Adam 二阶动量。

训练内存节省主要来自这些状态和部分 activation 策略，而不仅是 adapter 文件很小。

粗略以 FP32 Adam 为例，一个可训练参数可能需要：

```text
weight 4 bytes
gradient 4 bytes
first moment 4 bytes
second moment 4 bytes
总计约 16 bytes
```

实际混合精度实现还可能保留 master weight，具体以框架为准。

## 目标层选择的含义

### Q/V

常见起点，参数少，能调整注意力检索与内容汇总。

### Q/K/V/O

覆盖完整 attention projection，参数更多。

### FFN Projection

可调整逐 token 特征变换，对任务适配可能更强，但参数和计算增加。

### Embedding 或 Output Head

某些任务需要调整词表相关表示，但结构与保存方式可能不同，不能默认按普通 LoRA 处理。

## Rank 越大为什么不一定更好

更大 rank：

- 可表达更复杂更新。
- 参数、梯度和优化器状态更多。
- 小数据上更容易过拟合。
- 训练时间和 adapter 文件变大。

最优 rank 由任务、数据量、目标层和基座模型共同决定。

## 多 Adapter

同一基座可对应多个 adapter：

```text
base model
  + translation adapter
  + tool-routing adapter
  + domain adapter
```

推理时可以切换 adapter，但同时组合多个 adapter 需要明确权重、目标层冲突和缩放规则。

## 权重布局：公式必须绑定框架约定

对行向量输入，教学公式：

```text
y = x @ W_math
W_math: [in,out]
```

PyTorch/MLX Linear 通常保存：

```text
weight: [out,in]
y = x @ weight.T
```

本项目 A/B：

```text
A: [in,r]
B: [r,out]
forward delta: x @ A @ B
stored Delta-weight: (A @ B).T = B.T @ A.T
```

因此 `delta_weight()` 返回 `[out,in]`：

```python
return scale * (A @ B).T          # PyTorch Lab
return (scale * B.T) @ A.T        # MLX implementation
```

二者数学相同。讨论 LoRA 时若只写 `BA` 而不标 shape，很容易在不同论文/框架命名约定间把
A、B 对调。

## 低秩更新限制了哪些方向

$$
\Delta W=A B
$$

满足：

$$
rank(\Delta W)\le r
$$

`A` 把输入映射到 r 维中间坐标，`B` 把这些坐标映射到输出。Delta-W 的列空间和行空间都
被这两个小矩阵限制，因此不是任意 full-rank 更新。

但多个 LoRA 层之间夹着非线性、Attention 和 residual。整个网络函数的变化不等于一个全局
rank-r 线性映射；“低秩”只描述每个被注入权重的局部参数增量。

训练后可检查：

```python
singular_values = torch.linalg.svdvals(delta_weight.float())
effective_rank = (singular_values > tolerance).sum()
assert effective_rank <= rank
```

浮点误差下理论零奇异值可能表现为很小非零值，因此需要 tolerance。

## 首步梯度的矩阵推导

令：

```text
Z = X A
U = Z B
G = dL/dU
```

则：

$$
\frac{\partial L}{\partial B}=Z^TG
$$

$$
\frac{\partial L}{\partial A}=X^TGB^T
$$

初始化 `B=0` 时：

```text
dL/dA = 0
dL/dB = (X A)^T G，通常非零
```

所以第一步 B 先离开零，后续 A 才获得梯度。若 A/B 都为零：

```text
Z = X A = 0 -> dL/dB = 0
dL/dA contains B -> 0
```

两者都无法启动。这里不是模糊的“缺少对称性”，而是可以直接由导数看出梯度为零。

## Alpha/Rank 缩放改变更新幅度

项目使用：

$$
scale=\frac{\alpha}{r}
$$

若增加 rank 而保持 alpha 不变，单个分支整体缩放减小；若保持 `alpha/r` 不变，则 alpha
需要随 rank 线性增加。不同实验比较 rank 时必须同时报告 alpha 和最终 scale。

其他实现可能使用 `alpha/sqrt(r)`。该约定旨在不同 rank 下维持不同的更新统计尺度，但与
`alpha/r` 训练出的 adapter 不兼容。Adapter config 中只记录 alpha 而省略 scaling 规则仍然
不够。

## LoRA Dropout 与 Fuse 的等价条件

训练时：

```text
xW + scale * dropout(x) A B
```

Dropout 使每次 forward 的分支随机，不能与一个固定 fused weight 逐次等价。评估时
`dropout` 关闭，动态 adapter 才满足：

$$
xW+x\Delta W=x(W+\Delta W)
$$

因此 fuse 对比必须：

```python
lora.eval()
with torch.no_grad():
    dynamic = lora(x)
    fused = lora.fuse()(x)
```

若在 train mode 比较，差异可能只是 dropout，而不是 fuse 公式错误。

## Fuse 的精度与生命周期

融合时要处理：

- Delta-W 的 dtype 是否先在 FP32 累加再转回。
- Bias 是否原样复制。
- 是否已经 fuse 过，避免重复加 Delta-W。
- 原 adapter 是否还要保留以支持卸载/切换。
- 量化基座是否允许直接相加，或需要反量化再重新量化。

动态 LoRA 在低精度下执行两次小 GEMM；fused 模型执行一次原 Linear。二者浮点运算顺序不同，
应使用相对/绝对容差，而不是要求 bitwise equal。

## 冻结与 Optimizer 参数集合

仅设置：

```python
base.requires_grad_(False)
```

还应确认 optimizer 只接收可训练参数：

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=...)
```

把冻结参数放进 optimizer 通常不会更新它们，但会增加遍历和状态风险。MLX 项目通过
`model.freeze()` 后只保存/更新 `trainable_parameters()`。

验证：

```text
trainable parameter names
trainable / total ratio
optimizer state keys
base weight checksum before/after
```

仅打印总参数量无法证明冻结正确。

## 目标模块匹配必须可审计

字符串目标如 `q_proj,v_proj` 可能因模型命名不同而一个都匹配不到，或误匹配非 Linear。
项目 `inject_lora()`：

1. 先冻结整个模型。
2. 只遍历最后 N 个 block。
3. 检查 leaf name 是否在 target set。
4. 验证目标是 Linear。
5. 替换并记录完整 module path。
6. 若替换数为 0，直接报错。

Adapter metadata 保存相对 key，使加载端能在相同 block 结构中重建 LoRA 模块。训练日志应
输出实际 replaced module list，而不只输出用户请求的 target 字符串。

## 多 Adapter 的线性与非线性

同一 Linear 上多个 adapter 可形式化为：

$$
W'=W+\lambda_1\Delta W_1+\lambda_2\Delta W_2
$$

在该层局部是线性叠加。但 adapter 分布在多层，经过非线性和 Attention 后，整个模型输出
不会等于各 adapter 输出的简单加权和。组合还要求：

- 同一个基座 checkpoint。
- 兼容 tokenizer 和模型结构。
- 目标层 shape 一致。
- 缩放与权重系数明确。

不能把来自不同 base revision 的 adapter 直接相加。

## 可运行的首步梯度检查

```python
import torch

torch.manual_seed(0)
x = torch.randn(8, 6)
A = torch.nn.Parameter(torch.randn(6, 2) * 0.01)
B = torch.nn.Parameter(torch.zeros(2, 4))
target = torch.randn(8, 4)

output = x @ A @ B
loss = (output - target).square().mean()
loss.backward()

assert torch.count_nonzero(A.grad) == 0
assert B.grad.abs().sum() > 0

with torch.no_grad():
    B -= 0.1 * B.grad
A.grad = None
B.grad = None

loss = ((x @ A @ B) - target).square().mean()
loss.backward()
assert A.grad.abs().sum() > 0
```

## 本章调试不变量

1. 每个 A/B shape 与目标 Linear 的 in/out layout 一致。
2. B=0 时注入前后 eval 输出一致。
3. 首步 A grad 为零、B grad 通常非零；第二步后 A 开始学习。
4. 只有 LoRA 参数可训练，基座权重 checksum 不变。
5. Adapter metadata 包含 base、rank、alpha、scale 规则、dropout、目标层和层数。
6. Fuse 在 eval mode 比较，处理 dtype/bias，且不重复融合。
7. 保存的 trainable key 与加载端重建出的 key 完全一致。

## 常见错误

1. 基座参数没有真正冻结。
2. A/B shape 与 Linear 权重布局相反。
3. 保存 adapter 时遗漏 scale 或目标层。
4. 训练和加载使用不同 LoRA 约定。
5. 认为 rank 越大一定越好。
6. Fuse 时 Delta-W 转置错误。
7. 注入后初始输出没有与基座做等价性检查。

## 动手练习

1. 计算 `in=4096,out=4096,rank=8/16/64` 的参数量。
2. 打印第 0 步 A/B 的梯度范数。
3. 将 A/B 都初始化为零，观察训练。
4. 对 q/v 与 q/k/v/o 比较可训练参数。
5. 验证动态 adapter 与 fused layer 输出误差。

## 自测

1. LoRA 低秩的是 W 还是 Delta-W？
2. B 为零时为什么模型初始输出不变？
3. 第一训练步 A 的梯度为什么可能为零？
4. LoRA 为什么能减少 optimizer memory？
5. Adapter 除 A/B 外还必须保存什么？
