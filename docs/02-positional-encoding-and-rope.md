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
