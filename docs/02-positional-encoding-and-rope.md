# 02 位置编码与 RoPE

Self-Attention 本身不区分 token 顺序。若没有位置表示，交换输入位置不会让注意力知道
哪个 token 在前、哪个在后。

## 学习目标

读完后应能：

1. 解释 Attention 为什么需要额外位置表示。
2. 比较正弦位置、learned position 和 RoPE。
3. 写出 RoPE 输入输出 shape 和应用位置。
4. 区分“公式可计算更长位置”与“模型能泛化到更长上下文”。

## 正弦位置编码

经典公式：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/D))
PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
```

使用方式：

```text
x = token_embedding * sqrt(D) + positional_encoding
```

特点：

- 没有可训练参数。
- 每个位置对应确定向量。
- 可以生成训练长度之外的位置，但不保证模型自动具备长上下文能力。

## Learned Position Embedding

Learned position 是一个可训练表：

```text
P: [max_length, D]
```

优点是位置表示由数据学习；缺点是最大长度固定，超出表范围时必须扩展或重新训练。

## RoPE

RoPE 不把位置向量加到 hidden state，而是旋转 Q/K 的相邻维度：

```text
x' = x cos(theta) - y sin(theta)
y' = x sin(theta) + y cos(theta)
```

旋转保持范数，并让 Q/K 点积自然包含位置差。RoPE 通常只作用于 Q 和 K，不作用于 V。

## 三者区别

| 方法 | 作用位置 | 是否训练 | 常见限制 |
| --- | --- | --- | --- |
| Sinusoidal | 加到 hidden | 否 | 表达方式固定 |
| Learned | 加到 hidden | 是 | 最大长度固定 |
| RoPE | 旋转 Q/K | 否或带缩放配置 | 长度外推仍需专门设计 |

## 奇数维度

正弦编码和 RoPE 通常按相邻两维成对处理。实现必须正确处理奇数 `d_model`，最后一维
不能因 shape 不匹配而越界。

## 实验

```bash
python -m labs.lab00_positional_encoding
```

重点比较：

- 输出 shape 是否一致。
- 位置编码是“相加”还是“旋转”。
- 固定参数与可训练参数的差别。
- 第一组 sin/cos 从位置 1 平移到位置 3 是否等价于二维旋转。
- Learned position table 的参数量是否为 `max_len * D`。

## 对照源码

- `models/layers.py::PositionalEncoding`
- `labs/lab00_positional_encoding.py`
- `labs/lab07_modern_blocks.py::apply_rope`

## 为什么 Attention 不自动知道顺序

如果没有位置表示，Self-Attention 对输入排列具有等变性。简单说，交换两个 token，
输出也只会跟着交换，模型没有额外信息判断“谁先出现”。

例如：

```text
"狗 咬 人"
"人 咬 狗"
```

token 集合相同，但语义不同。位置信息让模型区分两种排列。

## 正弦编码的频率层次

不同维度使用不同频率：

- 高频维度变化快，容易区分相邻位置。
- 低频维度变化慢，能在更长距离上保持平滑变化。

把多种频率组合起来后，每个位置获得近似唯一的模式，同时相邻位置仍保持结构关系。

### 一个二维直觉

若只考虑一对 sin/cos：

```text
position 0 -> [sin(0), cos(0)] = [0, 1]
position 1 -> [sin(w), cos(w)]
position 2 -> [sin(2w), cos(2w)]
```

这些点位于单位圆上，位置增加等价于按固定角速度旋转。

## 为什么 RoPE 体现相对位置

设位置 `m` 的 Q 旋转角度为 `mθ`，位置 `n` 的 K 为 `nθ`。两者点积中出现的
有效角度差与：

```text
mθ - nθ = (m - n)θ
```

有关，因此注意力更容易表达相对距离 `m-n`，而不仅是绝对位置编号。

这不是说 RoPE “直接存储距离”，而是旋转后的点积结构天然依赖位置差。

## RoPE 的实现步骤

常见流程：

1. 将 Q/K 最后一维按偶数、奇数位置分组。
2. 为每个 sequence position 生成 cos/sin。
3. 对成对维度执行旋转。
4. 保持输出 shape 与输入完全一致。

典型输入：

```text
Q, K: [B, H, T, Dh]
cos/sin: 可广播到 [1, 1, T, Dh/2]
```

广播维度错误是最常见实现 bug。

## 最大长度与外推

三种方法都需要明确最大长度：

- Learned position 超过表范围会直接索引失败。
- Sinusoidal 可以继续计算，但模型未必学会使用更长位置。
- RoPE 可以生成更长角度，但注意力分布可能超出训练经验。

“公式可以算”不等于“模型在该长度下有效”。

## 排列等变性的矩阵证明

没有位置表示时，Self-Attention 对 token 排列等变。设 `P` 是一个置换矩阵，把输入行重新
排序：

$$
X' = PX
$$

线性投影后：

$$
Q'=PQ,\quad K'=PK,\quad V'=PV
$$

Score 变为：

$$
Q'K'^T = (PQ)(PK)^T = P(QK^T)P^T
$$

对每行做 Softmax 后，行列仍按同一置换重排，最终：

$$
Attention(PX) = P\,Attention(X)
$$

也就是说，交换输入 token 只会交换对应输出，模型本身不知道原顺序是什么。位置表示通过
引入与位置绑定的信号打破这种对称性。

## 正弦位置编码的频率公式

更适合实现的写法是先定义逆频率：

$$
\omega_i = 10000^{-2i/D}
$$

然后：

$$
PE(pos,2i)=\sin(pos\,\omega_i),\quad
PE(pos,2i+1)=\cos(pos\,\omega_i)
$$

代码中常用 `exp(log(...))` 生成频率：

```python
inverse_frequency = torch.exp(
    torch.arange(0, d_model, 2).float()
    * (-math.log(10000.0) / d_model)
)
angles = positions[:, None] * inverse_frequency[None, :]
```

这里的两个外积轴是：

```text
positions:         [T,1]
inverse_frequency: [1,ceil(D/2)]
angles:            [T,ceil(D/2)]
```

若 `D` 为奇数，偶数槽位比奇数槽位多一个，因此赋值 cos 时必须截断频率长度。本项目
`PositionalEncoding` 和 `lab00` 都显式处理了这一点。

### 为什么 sin/cos 能表达相对平移

利用和角公式：

$$
\sin((p+k)\omega)
= \sin(p\omega)\cos(k\omega)+\cos(p\omega)\sin(k\omega)
$$

$$
\cos((p+k)\omega)
= \cos(p\omega)\cos(k\omega)-\sin(p\omega)\sin(k\omega)
$$

因此同一频率对在位置 `p+k` 的表示，可以由位置 `p` 的 sin/cos 通过一个只依赖偏移 `k`
的二维旋转得到。这给网络提供了学习相对位移关系的结构，但不保证它一定学会任意长度的
精确相对推理。

## RoPE 的复数表示

将一对实数维度看成复数：

$$
\tilde q = q_{2i} + \mathrm{i}q_{2i+1}
$$

位置 `m` 的旋转是：

$$
\tilde q_m = \tilde q\,e^{\mathrm{i}m\theta_i}
$$

位置 `n` 的 key：

$$
\tilde k_n = \tilde k\,e^{\mathrm{i}n\theta_i}
$$

旋转后实数点积对应复数乘积的实部，其中位置相位变为：

$$
e^{\mathrm{i}(m-n)\theta_i}
$$

所以 Q/K 点积依赖相对位置 `m-n`。同时复数乘单位模长相位不会改变范数，这解释了：

```python
assert torch.allclose(q.norm(dim=-1), rope(q).norm(dim=-1))
```

RoPE 不是给 Value 加上“距离标签”；它改变的是 Q/K 的匹配几何，Value 仍负责提供内容。

## RoPE 广播与缓存布局

输入：

```text
Q/K: [B,H,T,Dh]
```

本项目实现把最后一维拆成相邻对：

```text
even/odd: [B,H,T,Dh/2]
cos/sin:  [1,1,T,Dh/2]
```

逐对旋转后 `stack(..., dim=-1)` 得到：

```text
[B,H,T,Dh/2,2] -> flatten -> [B,H,T,Dh]
```

生产实现通常预计算 cos/sin cache，避免每层、每步重复生成。增量解码时，新 token 必须
使用其绝对 cache position，而不是每次都从位置 0 重新旋转：

```python
position_ids = torch.arange(cache_length, cache_length + new_length)
```

否则 KV Cache 中旧 K 使用旧角度，新 K 却错误重置角度，full forward 与 cached decode
不会等价。

## 长上下文扩展为什么不是简单增大 `max_len`

Learned position 的问题是表不够长；正弦和 RoPE 的问题更隐蔽：公式虽然有定义，但训练时
见过的相位范围、相对距离和 Attention 模式有限。常见扩展策略会调整位置或频率，例如：

- Position interpolation：把更长位置压缩回训练位置范围。
- Frequency/NTK-aware scaling：改变不同频率的外推速度。
- 继续预训练：让模型实际适应更长序列分布。

这些策略改变 checkpoint 的位置约定。训练和推理必须使用同一 RoPE base、scaling 类型和
参数，否则模型不会报 shape 错，却会出现质量退化。

## 最小相对位置验证

```python
import torch

def rotate(pair, angle):
    c = torch.cos(torch.tensor(angle))
    s = torch.sin(torch.tensor(angle))
    x, y = pair
    return torch.stack((x * c - y * s, x * s + y * c))

q = torch.tensor([1.2, -0.4])
k = torch.tensor([0.3, 0.8])
theta = 0.17
m, n, shift = 3, 7, 11

score = rotate(q, m * theta) @ rotate(k, n * theta)
shifted = rotate(q, (m + shift) * theta) @ rotate(k, (n + shift) * theta)
assert torch.allclose(score, shifted, atol=1e-6)
```

整体平移 Q/K 的位置不改变相对差，因此这一二维 score 保持不变。

## 本章调试不变量

1. 加法位置编码与 hidden 的最后两维可广播为 `[B,T,D]`。
2. RoPE 的 Q/K 使用相同频率配置和 position ids。
3. 被旋转维度必须成对；未旋转的剩余维度保持原值。
4. RoPE 前后 Q/K shape、dtype、device 和每对范数保持一致。
5. Cached decode 的 position offset 等于已有 cache 长度。
6. 超长上下文实验同时记录训练长度、测试长度和 scaling 配置。

## 常见误区

1. 认为 Attention 自带顺序信息。
2. 将 RoPE 加到 V 上。
3. 认为 RoPE 自动支持任意长上下文。
4. 忽略 `Dh` 必须能按旋转对分组。
5. cos/sin 的 sequence 维与 head 维放反。

## 动手练习

1. 打印前 8 个正弦位置向量，观察不同维度变化速度。
2. 计算两个位置向量的余弦相似度。
3. 对同一个 Q/K 应用位置 0 和位置 5 的 RoPE，比较范数。
4. 将输入序列整体平移一个位置，观察 attention score 如何变化。

## 自测

1. 为什么纯 Self-Attention 无法区分“人咬狗”和“狗咬人”？
2. 正弦编码为什么同时使用 sin 和 cos？
3. RoPE 为什么只旋转 Q/K？
4. 位置编码能计算到 8192 是否意味着模型能可靠使用 8192？
