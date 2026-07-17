# 05 FFN、残差与 Pre-LN Block

一个 Transformer block 由 token 间通信和逐 token 变换两部分组成。

## 学习目标

读完后应能：

1. 解释 Attention 与 FFN 的职责分工。
2. 计算 FFN 参数量并理解中间维度。
3. 比较 Pre-LN 与 Post-LN 的数据和梯度路径。
4. 说明 residual、dropout 和 final norm 的作用。

## Position-wise FFN

经典 FFN：

```text
FFN(x) = W2 ReLU(W1 x + b1) + b2
```

Shape：

```text
[B, T, D] -> [B, T, Dff] -> [B, T, D]
```

Attention 在 token 之间混合信息；FFN 对每个 token 独立做非线性变换，所有位置共享参数。

## 残差连接

```text
x = x + sublayer(x)
```

残差提供恒等路径，让深层梯度可以直接传播到浅层，也允许子层学习相对输入的增量。

## LayerNorm 与 Pre-LN

本项目采用 Pre-LN：

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Encoder-Decoder 的 decoder block 多一个 cross-attention：

```text
x = x + SelfAttention(LN(x))
x = x + CrossAttention(LN(x), encoder_memory)
x = x + FFN(LN(x))
```

Pre-LN 通常比 Post-LN 更容易训练深网络。由于最后一次残差后没有归一化，stack 末尾还
需要 final norm。

## Encoder 与 Decoder Block

Encoder：

- 双向 self-attention。
- FFN。

Decoder：

- causal self-attention。
- 可选 cross-attention。
- FFN。

项目为了复用实现，让 Decoder-Only 也使用通用 `DecoderLayer`。没有 encoder memory
时 cross-attention 不参与计算。

## 实验

```bash
python -m labs.lab03_pre_ln_block
```

观察：

- 将所有 residual branch 参数置零后，output 是否严格等于 input。
- 零分支时 `output.sum()` 对 input 的梯度是否全为 1。
- Pre-LN 前后的 activation 范围。
- final norm 是否影响输出尺度。

## 对照源码

- `models/decoder_encoder_layer.py`
- `models/layers.py::PositionwiseFeedForward`
- `tests/test_pre_ln.py`

## 为什么 Attention 后还需要 FFN

Attention 输出主要是不同 token 的 Value 加权组合。若只有 Attention，模型对每个位置的
非线性变换能力有限。FFN 在 token 已经交换信息之后，进一步提取和组合特征。

可以把一个 block 粗略理解为：

```text
Attention: 先和其他 token 沟通
FFN:       再独立思考和加工
```

## FFN 参数量

忽略 bias：

```text
W1: D * Dff
W2: Dff * D
总计: 2 * D * Dff
```

若 `Dff = 4D`：

```text
FFN 参数约 8D^2
```

这通常比 Attention 的约 `4D^2` 更多，因此许多 Transformer 的大部分参数位于 FFN。

## 非线性为什么重要

若连续堆叠的都是线性层：

```text
x W1 W2
```

仍可合并为一个线性变换。ReLU、GELU、SiLU 或 SwiGLU 引入非线性，让多层网络能表达
更复杂函数。

## LayerNorm 在算什么

对单个 token 的 hidden vector：

```text
x = [1, 2, 3]
mean = 2
centered = [-1, 0, 1]
```

LayerNorm 在最后一个 hidden dimension 上归一化，每个 token 独立处理，不跨 batch，
也不跨 sequence position。

归一化后再乘可训练 scale、加可训练 bias，因此模型仍能恢复需要的尺度和偏移。

## Pre-LN 与 Post-LN

Post-LN：

```text
x = LayerNorm(x + Attention(x))
```

Pre-LN：

```text
x = x + Attention(LayerNorm(x))
```

Pre-LN 的残差主路径更接近恒等映射，梯度可以绕过子层和归一化直接传播，因此深层训练
通常更稳定。

## 残差的 Shape 约束

```text
x + sublayer(x)
```

两者 shape 必须完全一致。因此 Attention 和 FFN 最终都要返回 `[B,T,D]`。这也是
output projection 和 FFN down projection 存在的原因。

## Dropout 放在哪里

常见位置：

- Attention weight。
- Attention output。
- FFN activation 或 output。
- Embedding + position 后。

`model.train()` 时 dropout 生效，`model.eval()` 时关闭。验证和推理忘记调用 eval
会导致结果随机。

## 一个 Block 的逐步执行

```text
input x                         [B,T,D]
norm1(x)                        [B,T,D]
self-attention                  [B,T,D]
x + attention                  [B,T,D]
norm2(x)                        [B,T,D]
FFN                             [B,T,D]
x + FFN                        [B,T,D]
```

Decoder block 若有 cross-attention，会在 self-attention 和 FFN 之间再增加一组
norm、cross-attention 和 residual。

## Token Mixing 与 Channel Mixing

Transformer block 可以从轴的角度拆成两类操作：

```text
Attention: 沿 T 轴混合 token 信息
FFN:       沿 D 轴混合单个 token 的特征
```

Attention 的权重依赖输入，输出位置 `t` 会读取其他位置；FFN 对每个 `(b,t)` 使用同一组
参数：

$$
FFN(x_{b,t,:}) = W_2\,\phi(W_1x_{b,t,:}+b_1)+b_2
$$

因此把序列展平为 `[B*T,D]` 后执行 FFN，再 reshape 回 `[B,T,D]`，数学结果相同。FFN
不会直接跨 token 通信，但会处理 Attention 已汇总到当前 token 的信息。

## FFN 的精确参数量与计算量

经典两层 FFN：

```text
linear1 weight [Dff,D], bias [Dff]
linear2 weight [D,Dff], bias [D]
```

参数量：

$$
2DD_{ff}+D_{ff}+D
$$

对 `[B,T,D]` 输入，主要乘法量近似：

$$
2BTDD_{ff}
$$

当 `Dff=4D` 时，FFN 权重约 `8D^2`，每 token 的主要乘法也约 `8D^2`。这解释了为何
FFN 常同时是参数和计算大户。激活张量 `[B,T,Dff]` 还会占用训练显存，不能只看权重。

## 非线性改变了什么

若去掉激活：

$$
W_2(W_1x)=W'x
$$

两层可折叠成一层，增加深度不会增加分段或门控能力。加入 ReLU/GELU/SiLU 后，不同输入
区域激活不同中间通道，模型可以学习条件化的特征变换。

ReLU 的负区间梯度为零；GELU/SiLU 更平滑。选择激活会改变 checkpoint 参数的训练分布，
不能在加载权重时随意替换，即使 shape 完全相同。

## LayerNorm 的完整公式

对单个 token `x in R^D`：

$$
\mu = \frac{1}{D}\sum_d x_d
$$

$$
\sigma^2 = \frac{1}{D}\sum_d(x_d-\mu)^2
$$

$$
LN(x)_d = \gamma_d\frac{x_d-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_d
$$

PyTorch LayerNorm 使用总体方差语义，相当于 `unbiased=False`。`epsilon` 放在平方根内部，
避免方差接近零时除零。不同模型的 epsilon 可能是 `1e-5`、`1e-6` 等，混用 checkpoint
配置会造成细小但逐层累积的差异。

LayerNorm 对每个 token 独立，因此 batch 中增加 PAD 不会直接改变有效 token 的归一化统计；
这与 BatchNorm 的跨样本统计不同。

## Pre-LN 与 Post-LN 的 Jacobian

Post-LN：

$$
x_{l+1}=N(x_l+F(x_l))
$$

$$
J_{post}=J_N(I+J_F)
$$

Pre-LN：

$$
x_{l+1}=x_l+F(N(x_l))
$$

$$
J_{pre}=I+J_FJ_N
$$

Pre-LN 保留显式 `I` 分支，使梯度可以绕过当前 Norm 与子层。这通常改善深层训练稳定性，
但不意味着梯度绝对不衰减、无需 warmup 或永不爆炸。多层总 Jacobian、残差分支尺度、
初始化和优化器仍共同决定训练行为。

## 残差流的激活尺度

Pre-LN 中每个子层输入先被归一化，但输出不断加到 residual stream：

```text
x_{l+1} = x_l + delta_l
```

若各层更新 `delta_l` 方差过大，residual stream 的 RMS 仍可能随深度增长。Final norm 负责在
输出 head 前重新控制尺度，却不会自动修复中间层数值问题。大型模型还可能配合残差缩放、
特殊初始化或更复杂 norm 方案。

调试时同时记录：

```text
norm input RMS
sublayer output RMS
residual before/after RMS
gradient norm
```

只看最终 logits 是否有限，无法定位是哪一层开始漂移。

## Dropout 的期望与路径

PyTorch inverted dropout 在训练时：

```text
保留概率 1-p
保留值除以 1-p
```

所以输出期望与未 dropout 值相同，但单次样本方差增加。残差结构中常写：

```python
x = x + dropout(sublayer(norm(x)))
```

Dropout 应只作用于分支输出，不应把 residual 主干一起随机清零。Attention weight dropout 与
Attention output dropout 也不是同一位置；修改位置会改变训练语义。

## 项目 DecoderLayer 的真实执行轨迹

```python
normed = norm1(x)
self_out = self_attn(normed, normed, normed, self_mask)
x = x + dropout1(self_out)

if encoder_memory is not None:
    normed = norm2(x)
    cross_out = cross_attn(normed, encoder_memory, encoder_memory, cross_mask)
    x = x + dropout2(cross_out)

normed = norm3(x)
ffn_out = ffn(normed)
x = x + dropout3(ffn_out)
```

Decoder-Only 调用时 `encoder_memory=None`，所以 cross-attention 分支及 `norm2` 不执行；
FFN 仍使用 `norm3`。阅读通用层时不能只按成员变量数量推断实际路径。

## 最小梯度与激活探针

```python
import torch

def rms(x):
    return x.float().pow(2).mean().sqrt().item()

activations = {}
handles = []

def save_output(name):
    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        activations[name] = rms(tensor.detach())
    return hook

for name, module in block.named_modules():
    if isinstance(module, torch.nn.LayerNorm):
        handles.append(module.register_forward_hook(save_output(name)))

output, _ = block(x, mask)
output.square().mean().backward()
print("activation RMS:", activations)
print("input grad norm:", x.grad.norm().item())

for handle in handles:
    handle.remove()
```

Hook 只用于诊断，长期保留未释放的 handle 或保存带计算图的 Tensor 会造成内存增长。

## 本章调试不变量

1. 每个子层最终输出 shape 与 residual 输入完全一致。
2. Norm 沿最后一个 hidden 轴，不跨 batch/sequence。
3. Decoder-Only 不执行 cross-attention；Encoder-Decoder 才传 memory。
4. Pre-LN stack 末端存在 final norm。
5. 训练与验证正确切换 `train()` / `eval()`。
6. 记录分支输出与 residual RMS，避免只观察最终 loss。
7. 参数量统计包含 FFN bias，并区分参数内存与 activation 内存。

## 常见错误

1. 残差两端 hidden dimension 不一致。
2. Pre-LN stack 末尾漏掉 final norm。
3. 验证时没有调用 `model.eval()`。
4. Decoder-Only 意外执行了 cross-attention。
5. FFN 中间维度过大导致参数和内存超预期。
6. 使用 inplace 操作破坏 autograd 需要的值。

## 动手练习

1. 计算 `D=128,Dff=512` 时 FFN 参数量。
2. 暂时移除 residual，比较 lab03 的梯度范数。
3. 在 train/eval 模式下重复运行带 dropout 的 block。
4. 打印 final norm 前后的均值和 RMS。

## 自测

1. Attention 与 FFN 分别负责什么？
2. 为什么 FFN 输入输出维度相同，中间维度却更大？
3. Pre-LN 为什么有利于梯度传播？
4. 为什么每个 stack 末尾仍需要 final norm？
