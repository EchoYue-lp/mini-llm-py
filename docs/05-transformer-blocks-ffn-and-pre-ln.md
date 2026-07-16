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

- 关闭残差后梯度如何变化。
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
