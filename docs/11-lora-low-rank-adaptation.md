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
2. 只有低秩参数可训练。
3. Fuse 前后输出一致。

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
