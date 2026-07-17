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
- RoPE offset 改变时相位是否改变而范数不变。
- 经典 FFN 与等参数预算 SwiGLU 的 hidden size/权重数。

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

## RMSNorm 的尺度性质

忽略 epsilon 和可训练 weight，定义：

$$
RMSNorm(x)=\frac{x}{\sqrt{\frac{1}{D}\sum_dx_d^2}}
$$

对正标量 `c`：

$$
RMSNorm(cx)=RMSNorm(x)
$$

对负标量则保留符号翻转。这说明 RMSNorm 对输入整体幅值近似不敏感，让后续子层看到更
稳定的尺度。

但它不具备平移不变性：

```text
RMSNorm(x + constant) != RMSNorm(x)
```

因为它不减均值。LayerNorm 会移除沿全 1 向量方向的均值分量，RMSNorm 保留该信息。二者
不是“少一个减法、效果完全相同”，而是对表示空间施加不同约束。

## RMSNorm 的数值精度

低精度输入平方时可能溢出或损失精度。生产实现常在更高精度中计算统计量，再转回输入
dtype：

```python
def rms_norm(x, weight, eps):
    input_dtype = x.dtype
    x_float = x.float()
    inv_rms = torch.rsqrt(x_float.square().mean(-1, keepdim=True) + eps)
    return (x_float * inv_rms * weight.float()).to(input_dtype)
```

本项目 Lab 已使用 FP32 计算 RMS 统计量，再转回输入 dtype。用 FP16/BF16 扩展实验时，仍应
检查极大值、全零输入和接近常数输入。

Epsilon 的位置也属于模型定义：

```text
1 / sqrt(mean(x^2) + eps)
```

与 `1 / (sqrt(mean(x^2)) + eps)` 不完全相同，加载 checkpoint 时不能混淆。

## RMSNorm 的输出 RMS 为什么不一定正好为 1

若 `eps=0`、weight 全 1，归一化输出的 RMS 为 1。实际中：

- epsilon 会让极小输入的输出 RMS 小于 1。
- 可训练 weight 按维缩放后，整体 RMS 会改变。
- 有限精度带来舍入误差。

因此测试应检查公式一致与数值有限，而不是强制所有训练后输出 RMS 精确等于 1。

## RoPE 的 Rotary Dimension

有些模型只旋转每个 head 的前 `rotary_dim` 个通道，其余通道不变：

```text
Q = [Q_rotary, Q_pass]
K = [K_rotary, K_pass]
```

约束：

```text
0 < rotary_dim <= Dh
rotary_dim % 2 == 0
```

本项目 Lab 旋转整个 `Dh`。读取其他模型时必须检查 rotary percentage/dimension，不能只看
到 `apply_rope` 名称就假设全部通道参与。

RoPE base、scaling、position offset 和 pair layout 都属于 checkpoint 协议。常见 pair layout
既可能是相邻偶奇维，也可能是把前后半维配对；两者 shape 相同但旋转结果不同。

## SwiGLU 的参数等预算推导

普通 `Dff=4D` FFN 参数近似：

$$
2D(4D)=8D^2
$$

SwiGLU 中间维为 `F`：

$$
3DF
$$

令两者相等：

$$
3DF=8D^2\Rightarrow F=\frac{8}{3}D\approx2.67D
$$

所以现代模型常把 SwiGLU hidden size 设在约 `2.67D` 附近，再按硬件友好的倍数取整，而
不是直接使用 `4D`。如果 `F=4D`，SwiGLU 参数会变成 `12D^2`，比经典 FFN 多 50%。

## SwiGLU 的门控梯度

中间表示：

$$
h=SiLU(g)\odot u
$$

对 `u` 的梯度被 `SiLU(g)` 缩放，对 `g` 的梯度同时依赖 `u` 与 SiLU 导数：

$$
\frac{\partial h}{\partial u}=SiLU(g)
$$

$$
\frac{\partial h}{\partial g}=u\odot SiLU'(g)
$$

因此 gate 不只是“开/关”：它连续调节 up 分支的值和梯度。若 gate 极度负饱和，通道贡献
会接近零；若 up 分支接近零，gate 也难从该样本得到强梯度。

## Bias 与模型兼容性

本项目 `SwiGLU` 三个 Linear 都使用 `bias=False`。不同模型可能：

- Attention/FFN 全部无 bias。
- 只有部分 projection 有 bias。
- Norm 有或没有 bias。

Bias 会改变参数 key 和 forward。加载 state dict 时即使允许 `strict=False`，缺失 bias 也可能
造成静默行为差异。模型转换应逐层比较参数名称、shape 和 forward 最大误差。

## 最小模块等价检查

```python
import torch

torch.manual_seed(0)
x = torch.randn(2, 5, 16)

norm = RMSNorm(16)
normalized = norm(x)
manual = x * x.pow(2).mean(-1, keepdim=True).add(norm.eps).rsqrt()
manual = manual * norm.weight
assert torch.allclose(normalized, manual)

ffn = SwiGLU(16, 32)
output = ffn(x)
gate = torch.nn.functional.silu(ffn.gate(x))
up = ffn.up(x)
assert torch.allclose(output, ffn.down(gate * up))
assert output.shape == x.shape
```

## 本章调试不变量

1. RMSNorm 只沿最后一维统计，weight shape 等于 hidden size。
2. 低精度实验中统计量有限，epsilon 与 checkpoint 配置一致。
3. RoPE 只作用指定 rotary dimension 的 Q/K，position offset 与 cache 对齐。
4. Gate/up 输出 shape 完全一致，down projection 回到 `D`。
5. SwiGLU 参数预算使用 `3DF`，不沿用经典 `2DDff`。
6. Bias、RoPE pair layout、base 和 norm epsilon 都纳入模型兼容检查。

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
