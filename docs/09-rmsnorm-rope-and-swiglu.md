# 09 RMSNorm、RoPE 与 SwiGLU

现代 Decoder-Only LLM 通常保留 Transformer 主干与 next-token prediction 目标，
但会替换归一化、位置表示和 FFN。

## 学习目标

读完后应能：

1. 手算 RMSNorm 并与 LayerNorm 比较。
2. 说明 RoPE 在现代 Attention 中的准确位置。
3. 推导 SwiGLU 三个投影的 shape 和参数量。
4. 阅读模型配置时识别现代 block 的关键差异。

## 常见替换

| 经典实现 | 现代常见选择 | 目的 |
| --- | --- | --- |
| LayerNorm | RMSNorm | 简化归一化 |
| Sinusoidal PE | RoPE | 将相对位置注入 Q/K |
| ReLU FFN | SwiGLU | 使用门控非线性 |

## RMSNorm

LayerNorm：

```text
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

RMSNorm：

```text
RMS(x) = sqrt(mean(x^2) + eps)
RMSNorm(x) = weight * x / RMS(x)
```

RMSNorm 不做中心化，通常只有缩放参数。两者都在最后一个 hidden dimension 上计算，
不会混合不同 token。

## RoPE

RoPE 将 Q/K 相邻维度按位置旋转：

```text
[x_2i, x_2i+1] -> rotate(theta(position, i))
```

旋转后的 Q/K 点积包含相对位置差。它不直接作用于 V，也不自动保证超出训练长度后的
效果；base、频率缩放与训练上下文都会影响外推。

## SwiGLU

经典 FFN：

```text
FFN(x) = W_down activation(W_up x)
```

SwiGLU：

```text
SwiGLU(x) = W_down(SiLU(W_gate x) * W_up x)
```

`W_gate` 决定哪些特征通过，`W_up` 提供候选特征。由于增加了一个投影分支，实际
模型通常会调整中间维度，使总参数量保持在目标预算内。

## 现代 Decoder Block

一个常见形式：

```text
x = x + Attention(RMSNorm(x), RoPE)
x = x + SwiGLU(RMSNorm(x))
```

具体模型可能使用不同的 bias、norm 位置、RoPE base、FFN 宽度和 residual 设计，不能
只凭模块名称假设实现完全一致。

## 实验

```bash
python -m labs.lab07_modern_blocks
```

重点观察：

- RMSNorm 是否保持 shape。
- RoPE 是否保持向量范数。
- SwiGLU 的 gate 与 up 分支 shape 是否一致。

## 对照源码

- `labs/lab07_modern_blocks.py`
- `labs/lab00_positional_encoding.py`

## 一个 RMSNorm 数值例子

设：

```text
x = [3, 4]
mean(x^2) = (9 + 16) / 2 = 12.5
RMS = sqrt(12.5) ~= 3.536
```

若 weight 初始为 `[1,1]`：

```text
RMSNorm(x) ~= [0.849, 1.131]
```

RMSNorm 不会减去均值，因此输出均值不一定为 0。

## Epsilon 为什么存在

若输入全为 0：

```text
RMS = 0
```

除以 0 会产生 NaN。实际公式加入很小的 `eps`：

```text
sqrt(mean(x^2) + eps)
```

Epsilon 太小可能在低精度下不稳定，太大又会改变归一化尺度。

## RMSNorm 与 LayerNorm 不是简单替换开关

更换 norm 会改变：

- 参数结构。
- 数值尺度。
- checkpoint key。
- 训练稳定性。
- 与 residual 的配合。

不能在已有 checkpoint 上直接把 LayerNorm 类名改成 RMSNorm 并期待权重兼容。

## RoPE 在 Attention 中的位置

典型顺序：

```text
hidden
  -> q_proj / k_proj
  -> split heads
  -> apply RoPE to Q/K
  -> QK^T
```

RoPE 应在 Q/K 已按 head 拆分后，根据实际 position index 应用。KV Cache decode 时，
新 token 必须使用其绝对 cache position，而不是每步都用位置 0。

## SwiGLU 的 Shape

设输入：

```text
x: [B,T,D]
```

两条上投影：

```text
gate = W_gate(x): [B,T,F]
up   = W_up(x):   [B,T,F]
```

门控：

```text
hidden = SiLU(gate) * up: [B,T,F]
```

下投影：

```text
W_down(hidden): [B,T,D]
```

Gate 和 up 的最后一维必须一致，才能逐元素相乘。

## SwiGLU 参数量

忽略 bias：

```text
W_gate: D * F
W_up:   D * F
W_down: F * D
总计: 3DF
```

普通两层 FFN 为 `2D Dff`。为了控制参数预算，SwiGLU 的 `F` 往往不会直接使用
普通 FFN 的 `4D`。

## SiLU 的直觉

```text
SiLU(x) = x * sigmoid(x)
```

负值不会像 ReLU 一样全部截断为 0，函数平滑，梯度也连续。与 gate 相乘后，网络可以
按输入动态控制特征通过程度。

## 一个现代 Block 的检查清单

阅读某个模型配置时确认：

1. 使用 LayerNorm 还是 RMSNorm。
2. Norm 在 residual 前还是后。
3. RoPE base 与最大位置。
4. Query heads 与 KV heads。
5. FFN 是 GELU、SwiGLU 还是其他门控。
6. Linear 是否带 bias。
7. Embedding 与 output head 是否共享。

只说“这是 Llama 风格 block”不够精确。

## 常见错误

1. RMSNorm 在 batch 或 sequence 维计算。
2. 忘记 epsilon。
3. RoPE position 在 decode 时重复从 0 开始。
4. RoPE 错误应用到 V。
5. SwiGLU gate/up 维度不一致。
6. 直接沿用普通 FFN 宽度导致参数暴增。

## 动手练习

1. 手算 `[3,4]` 的 RMSNorm。
2. 比较 LayerNorm 与 RMSNorm 输出均值。
3. 将 SwiGLU 的 gate 固定为大负数，观察输出。
4. 计算不同中间维度下普通 FFN 与 SwiGLU 参数量。

## 自测

1. RMSNorm 为什么不保证零均值？
2. RoPE 在 Q/K projection 前还是后应用？
3. SwiGLU 为什么需要三组线性投影？
4. 为什么现代模块不能只通过替换类名接入旧 checkpoint？
