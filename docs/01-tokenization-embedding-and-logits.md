# 01 Tokenizer、Embedding 与 Logits

模型不直接处理字符串。文本先变成 token id，再通过 embedding 变成连续向量。

## 学习目标

读完后应能：

1. 解释字符、单词和子词 tokenizer 的取舍。
2. 推导 `[B,T]` token id 到 `[B,T,D]` hidden state 的过程。
3. 区分 token id、embedding、logits、probability 和 loss。
4. 说明 PAD、BOS、EOS 为什么影响模型结构与训练。
5. 解释词表大小 `V` 与隐藏维度 `D` 为什么不是同一种容量。
6. 区分静态 token embedding 与上下文化 hidden state。

## Tokenizer

```text
"I love AI" -> [40, 1842, 9552]
```

常见特殊 token：

| Token | 用途 |
| --- | --- |
| `PAD` | 补齐 batch，不参与 loss |
| `BOS` | 序列开始 |
| `EOS` | 序列结束 |
| `UNK` | 未知内容 |

本项目的翻译任务使用 SentencePiece，文本生成使用 GPT-2 tokenizer。GPT-2 原始词表
没有 PAD，因此 `utils/tokenizer_utils.py` 会增加独立的 `<|pad|>`。不能把 id 0
当作 PAD，因为它是 GPT-2 词表中的真实 token。

## Token Embedding

Embedding 是一个可训练查表矩阵：

```text
E:   [V, D]
ids: [B, T]
E[ids] -> [B, T, D]
```

其中：

- `B`：batch size。
- `T`：序列长度。
- `V`：词表大小。
- `D`：隐藏维度 `d_model`。

项目按经典 Transformer 做法将 embedding 乘以 `sqrt(D)`，避免初始化早期 token
embedding 相对位置编码过小。

## 从 Hidden State 到 Logits

Transformer block 保持 hidden shape：

```text
[B, T, D] -> [B, T, D]
```

最终线性层把隐藏维投影到词表：

```text
[B, T, D] -> [B, T, V]
```

`[B, T, V]` 就是 logits。每个位置都有一个长度为 `V` 的分数向量，用于预测
下一个 token。

## Padding 的两层职责

PAD 必须同时处理：

1. Attention mask：PAD 不能作为有效 key 被读取。
2. Loss ignore：PAD target 不能贡献交叉熵。

这两个机制解决不同问题，不能只做其中一个。

## 对照源码

| 内容 | 文件 |
| --- | --- |
| GPT-2 tokenizer 加载 | `utils/tokenizer_utils.py` |
| SentencePiece 封装 | `utils/sentencepiece_tokenizer.py` |
| Embedding 与输出投影 | `models/transformer_models.py` |
| 数据预处理 | `scripts/preprocess.py`、`scripts/retokenize_dataset.py` |

## 从字符到 Token：为什么不直接按字处理

若模型按 Unicode 字符建词表，会遇到两个问题：

1. 字符种类多，包含生僻字、符号、不同语言和组合字符。
2. 英文单词若按字符拆分，序列会很长。

若按完整单词建词表，又会遇到未登录词：

```text
training, trained, trainer, retraining
```

子词 tokenizer 在字符和单词之间折中。例如可能拆成：

```text
re + train + ing
```

这样词表可控，又能组合出未见过的新词。

### BPE 与 SentencePiece 的直觉

BPE 从小单位开始，反复合并语料中常见的相邻片段。SentencePiece 将文本视为原始字符
流，不强制先按空格切词，适合中文等没有天然空格边界的语言。

Tokenizer 的训练与神经网络训练是两件事：

- Tokenizer 决定字符串如何变成 id。
- Transformer 学习 id 序列之间的统计关系。

更换 tokenizer 会改变整个词表，旧 checkpoint 的 embedding shape 通常不再兼容。

## 世界文字很多，为什么 `D=512` 还能表示

先区分三个容易混在一起的量：

| 量 | 例子 | 表示什么 |
| --- | ---: | --- |
| Unicode/文本可能性 | 极大且开放 | 人类可能输入的字符串 |
| Tokenizer 词表 `V` | 数万到数十万 | 当前模型认识的离散子词编号 |
| Hidden dimension `D` | 512、768、4096 | 每个位置可携带的连续特征带宽 |

模型不是给世界上每句话都分配一个 512 维向量，也不是把所有文字一次性塞进一个向量。
Tokenizer 先用有限子词表把任意字符串组合成 id 序列：

```text
任意文本
-> 多个 subword/byte token
-> [id_1, id_2, ..., id_T]
-> T 个 D 维向量
```

一段长度为 `T` 的文本进入模型后，初始表示是 `[T,D]`，并非只有 `[D]`。随着层数增加，
每个位置的 D 维 hidden state 又会通过 Attention 读取其他位置。因此模型对文本的内部状态
分布在：

```text
sequence positions * hidden dimensions * network layers
```

模型知识还存储在全部层的参数中，不只存储在 token embedding 的 512 个数里。

## One-Hot 到 Embedding：概念矩阵，而非真实大输入

若词表大小为 `V`，token id `i` 可以概念化为第 i 个标准基向量：

$$
e_i=[0,\ldots,0,1,0,\ldots,0]\in\mathbb{R}^{V}
$$

Embedding table：

$$
E\in\mathbb{R}^{V\times D}
$$

矩阵乘法：

$$
e_iE=E_{i,:}
$$

因为 one-hot 只有一个 1，结果就是取 E 的第 i 行。工程上不会真的创建一个长度 V 的
one-hot 再做稀疏矩阵乘法，而是直接查表：

```python
hidden = embedding(token_ids)
```

所以“从 V 维降到 D 维”是理解线性映射的数学视角，不是运行时先分配巨大 one-hot Tensor
再压缩。

Embedding 也没有消除全部词表成本。参数量仍是：

$$
V\times D
$$

例如 `V=50,000,D=512`：

```text
25,600,000 parameters
FP32 约 97.7 MiB
FP16/BF16 约 48.8 MiB
```

它节省的是每个 token 的运行时表示和后续计算宽度，同时让不同 token 能共享连续特征，
并不是把词表参数免费消掉。

## 这是有损表示，不是无损压缩

One-hot 中 V 个 token 是 V 个线性独立方向。映射到 D 维后：

$$
rank(E)\le D
$$

当 `D << V` 时，不可能保留 V 个彼此正交的独立方向，也不可能无损保留 one-hot 空间中
任意定义的全部关系。这是主动引入的瓶颈：模型假设语言任务不需要把每个 token 当成完全
无关的孤岛，而可以复用更少的特征方向。

低维空间仍能放置远多于 D 个不同点。`D=2` 的平面就能放无限多个坐标点，所以
`V>D` 不会强迫两个 token 向量完全相等。但维度会限制：

- 可用的线性独立方向数。
- 能在噪声和有限精度下稳定分开的几何结构。
- 后续层一次能传递多少特征。
- 模型可表达关系的复杂度和训练成本。

用 `10^512` 之类的组合数来说明“容量无限”并不严谨。实际容量受到浮点精度、参数数量、
优化数据、噪声、网络深度和可泛化结构的共同限制。

## 为什么语言规律允许共享特征

如果每个 token 的行为都完全独立，低维 embedding 会严重欠拟合。但语言具有大量共享结构：

- “苹果”和“香蕉”常出现在水果、食用、颜色等相似上下文。
- 不同动词共享时态、主谓关系和论元结构。
- 子词如 `train`、`ing` 会跨多个单词重复出现。
- 多种语言共享数字、实体、语义类别和翻译监督。

模型可以复用特征方向来处理这些规律。关键不是某一维被人工命名为“是否水果”，而是整个
向量方向与后续权重共同工作。神经网络的表示通常是 distributed representation：一个概念
由多个维度共同表达，一个维度也同时参与多个概念。

因此下面的说法只能作为比喻，不能当作事实：

```text
第 17 维 = 是否是生物
第 203 维 = 褒义程度
```

旋转整个 embedding 空间并相应旋转后续权重，模型函数可以保持不变，但单个维度的含义会
全部改变。这说明语义更常存在于方向、子空间和网络计算中，而非固定坐标轴标签。

## 低维结构是如何学出来的

Embedding 初始化时通常接近随机。训练不会先计算一个完美语义空间再写入 E，而是端到端
优化 next-token loss：

```text
token ids
-> embedding rows
-> Transformer layers
-> logits
-> cross entropy
-> gradients update E and all other trainable weights
```

对 token `i`：

$$
\frac{\partial L}{\partial E_{i,:}}
=\sum_{b,t:id_{b,t}=i}\frac{\partial L}{\partial h_{b,t,:}}
$$

若两个 token 在相似上下文中需要支持相似预测，它们长期收到的梯度统计会有共同结构，向量
和网络对它们的处理逐渐变得相似。这里没有一个显式规则强迫“苹果靠近香蕉”；接近关系是
完成训练目标的一种可能内部解法，也可能只在某些层或某些方向出现。

历史上可以把演进粗略理解为：

```text
one-hot
-> 共现计数矩阵
-> SVD/LSA 等低秩分解
-> Word2Vec 等预测式静态 embedding
-> 端到端训练的 contextual Transformer representation
```

现代 LLM 不会先独立训练 Word2Vec 再固定使用，而是在完整语言模型目标中联合学习 embedding
和 Transformer。

## Token Embedding 不是最终语义

Embedding table 对相同 id 总返回同一初始行：

```text
"银行 的 bank"
"河岸 的 bank"
```

其中 `bank` 的初始 token embedding 相同。进入 Transformer 后，Attention 读取不同上下文，
每层 hidden state 会变成不同向量：

```text
static embedding:       E[bank]
layer 1 hidden:         h_bank^(1)(context)
...
layer L hidden:         h_bank^(L)(context)
```

真正用于预测的 contextual hidden 不再只是查表结果。这也是为什么不能只对输入 embedding
做余弦相似度，就断言模型最终如何理解一个多义词。

## 为什么是 512，而不是 50 或 50,000

`D` 是经验选择的容量与成本折中，不存在“世界语言理论上恰好需要 512 维”的定理。

增大 D 会增加：

```text
Embedding 参数: O(V D)
Attention/FFN 投影参数: 通常 O(D^2)
每 token activation: O(D)
矩阵乘法成本: 随 D 或 D^2 增长
```

过小 D 会形成强瓶颈，模型难以同时携带词法、语法、位置、实体和任务状态；过大 D 会增加
参数、显存、延迟和过拟合风险。512、768、4096 等值来自模型规模、层数、head 数、数据和
硬件的联合设计，并通过实验验证。

`D` 还必须满足结构约束，例如标准 MHA 中：

```text
D % num_heads == 0
```

所以它也会按硬件友好和 head dimension 友好的倍数选择。

## 关于语义几何的三个限定

### 向量算术不是普遍定律

`king - man + woman ~= queen` 是早期静态 embedding 中著名的观察，但对不同训练算法、
词表、随机种子和现代 contextual representation 不保证稳定成立。它说明某些关系可能近似
线性化，不代表整个语言流形都服从简单平移。

### 余弦相似不等于完整语义相同

余弦只比较两个向量方向。模型后续还会使用 LayerNorm、线性投影、非线性和上下文。一个
embedding 空间中的近邻只是诊断线索，不是业务等价证明。

### 多语言表示不会自动完全重合

共享 tokenizer、参数、翻译数据和跨语言上下文可能让不同语言表示对齐，但“Apple”和
“苹果”不需要在每一层完全相同。模型可能使用复杂的共享子空间和上下文变换，而不是一
个固定旋转把所有语言逐词对齐。

## 可运行的高维到低维实验

项目新增：

```bash
python -m foundations.f02_embedding_geometry
```

该实验依次验证：

1. `one_hot @ embedding_table` 与直接 lookup 完全相同。
2. `[V,D]` embedding table 的矩阵秩不超过 D。
3. 词语共现矩阵 `[V,V]` 可通过截断 SVD 得到 `[V,D]` 低维表示。
4. 共享上下文的 token 在低维空间中具有更高余弦相似度。
5. 同一个静态 token embedding 加入不同上下文后会形成不同 contextual state。

SVD 只是帮助建立几何直觉；本项目 Transformer 的 embedding 实际由梯度下降端到端学习。

## 一个完整的 Shape 例子

假设：

```text
B = 2
T = 4
V = 10
D = 3
```

输入：

```text
ids shape = [2, 4]
[[2, 5, 7, 3],
 [2, 8, 3, 0]]
```

其中第二条最后一个 0 是 PAD。Embedding 表：

```text
E shape = [10, 3]
```

查表后：

```text
hidden = E[ids]
hidden shape = [2, 4, 3]
```

Embedding 不是把 id 当连续数值。例如 id 8 并不比 id 2 “更大”，它们只是查表索引。

## Embedding 为什么能学到语义

初始化时，每个 token 向量近似随机。训练过程中，如果两个 token 出现在相似上下文并
需要产生相似预测，它们收到的梯度方向也会相似，向量逐渐形成可利用的几何关系。

这不意味着单个维度有固定人类含义。语义通常分布在多个维度和多个层中。

## Logits、Softmax 与概率

假设某个位置的 logits：

```text
[2.0, 1.0, 0.0]
```

Softmax：

```text
[0.665, 0.245, 0.090]
```

若正确 token 是第 0 个：

```text
loss = -log(0.665) ~= 0.408
```

若正确 token 是第 2 个：

```text
loss = -log(0.090) ~= 2.408
```

训练通过梯度提高正确 token 的相对 logit，不要求所有错误 logit 同时变成负数。

## 为什么训练时通常直接传 Logits

`CrossEntropyLoss` 内部会稳定地完成 `log_softmax + negative log likelihood`。
不要先手动 softmax 再传入，否则会降低数值稳定性并改变 API 预期。

## Weight Tying

部分语言模型让输入 embedding 与输出 projection 共享权重：

```text
input embedding:  [V, D]
output projection: [D, V]
```

共享可以减少参数，并让“读 token”和“预测 token”使用同一表示空间。本项目当前实现
应以源码为准，不要假设所有模型都自动 tying。

## Tokenizer 是模型接口协议

Tokenizer 不只是预处理工具，它定义了模型输入输出的离散协议。一个 checkpoint 至少隐含
依赖以下信息：

- token 到 id 的完整映射。
- 特殊 token 的 id 和语义。
- 文本规范化规则。
- pre-tokenization 和子词合并规则。
- 是否自动添加 BOS/EOS。
- decode 时如何处理空格、字节和特殊 token。

只比较 `vocab_size` 不足以判断两个 tokenizer 兼容。两个词表都可能有 50,000 个 token，
但只要 id 421 对应的字符串不同，同一 embedding 行就会被解释成不同符号，模型输出立即
失去语义。

因此 checkpoint 与 tokenizer 的核心不变量是：

```text
same token string <-> same token id <-> same embedding/output row
```

保存模型时应同时保存 tokenizer 文件或可验证的 tokenizer 标识，而不是只记录词表大小。

### Encode/Decode 不一定是字符级严格逆变换

理想情况下：

```python
decode(encode(text)) == text
```

但实际 tokenizer 可能执行 Unicode normalization、空白规范化、byte fallback 或特殊 token
过滤。更可靠的测试是明确当前 tokenizer 的契约：

```python
ids = tokenizer.encode(text, add_special_tokens=False)
decoded = tokenizer.decode(ids, skip_special_tokens=False)
print(repr(text), repr(decoded))
```

GPT-2 byte-level BPE 还会把空格信息编码进 token 片段，因此：

```text
"hello" 与 " hello"
```

可能得到不同 token。调试生成结果时，不能只查看 id，应同时打印 token 字符串和
`repr(decoded_text)`。

## 词表大小变化为什么会破坏模型 shape

假设旧模型词表为 `V`：

```text
embedding.weight: [V,D]
out_proj.weight:  [V,D]
out_proj.bias:    [V]
```

新增一个 PAD 后，新词表变成 `V+1`。Tokenizer 能产生新 id，不代表模型自动拥有对应参数。
若模型仍按旧 `V` 构造：

- 输入 PAD id 可能触发 embedding 越界。
- 输出层无法为新 token 产生 logit。
- checkpoint 加载会出现 size mismatch。

正确顺序是：先得到最终 tokenizer，再用 `len(tokenizer)` 构造模型。对已经存在的模型做
词表扩展时，需要显式 resize embedding/output 层，并决定新行如何初始化。

本项目 `load_gpt2_tokenizer()` 会在缺失时增加 PAD，因此模型配置使用：

```python
tokenizer = load_gpt2_tokenizer(...)
model = DecoderOnlyModel(vocab_size=len(tokenizer), ...)
```

不能在模型构造后才临时给 tokenizer 增加 token。

## Embedding 查表的梯度到底更新哪些行

对一个 batch：

```text
ids = [[2, 5, 2]]
```

forward 只读取 embedding 的第 2、5 行。反向传播时，也只有实际被索引的行收到来自该 batch
的梯度；id 2 出现两次，其梯度贡献会累加。概念上：

$$
\frac{\partial L}{\partial E_{v,:}}
= \sum_{b,t:\,id_{b,t}=v}
  \frac{\partial L}{\partial h_{b,t,:}}
$$

这解释了三个训练现象：

1. 低频 token 的 embedding 更新机会少。
2. 高频 token 的同一行会收到大量上下文的累积信号。
3. PAD 若没有从 loss 和有效上下文中排除，其 embedding 会学习到无意义的批处理模式。

`nn.Embedding` 默认产生普通 dense gradient；`sparse=True` 才会请求稀疏梯度，并要求优化器
支持相应形式。“只更新被查到的行”描述的是非零梯度结构，不等于默认 `.grad` 一定使用
稀疏存储。

## Embedding 缩放的尺度分析

项目使用：

```python
hidden = embedding(ids) * math.sqrt(d_model)
```

经典 Transformer 初始化中，embedding 各维数值通常较小，而固定正弦位置编码分量位于
`[-1,1]`。乘 `sqrt(D)` 的目的，是避免训练初期位置项在相加时相对 token 表示过强。

这不是所有现代模型都必须使用的规则：

- Learned position 的初始化尺度可能不同。
- RoPE 不与 token embedding 直接相加。
- RMSNorm/初始化方案也会改变激活尺度。

因此是否缩放必须与模型架构、checkpoint 训练约定一致，不能在推理时随意增加或删除。

## 从 Hidden 到 Logits 的逐元素公式

输出投影 `nn.Linear(D,V)` 保存：

```text
weight: [V,D]
bias:   [V]
```

位置 `(b,t)` 对词表项 `v` 的 logit：

$$
z_{b,t,v} = b_v + \sum_{d=1}^{D} h_{b,t,d} W_{v,d}
$$

它是未归一化兼容分数，不需要处于 `[0,1]`，也不要求总和为 1。Softmax 对所有 logits
同时加常数不敏感，所以模型主要学习类别间相对差值。

若使用 weight tying：

```python
out_proj.weight = embedding.weight
```

则输出 logit 近似衡量 hidden 与各 token embedding 行的匹配程度。共享的是同一个 Parameter，
不是每步手动复制两份权重；否则优化器状态和梯度可能分叉。

## 最小可运行检查

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
embedding = torch.nn.Embedding(num_embeddings=7, embedding_dim=4)
ids = torch.tensor([[2, 5, 2]])
hidden = embedding(ids)
assert hidden.shape == (1, 3, 4)
assert torch.equal(hidden[0, 0], hidden[0, 2])

projection = torch.nn.Linear(4, 7)
logits = projection(hidden)
labels = torch.tensor([[5, 2, 1]])
loss = F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1))
loss.backward()

assert embedding.weight.grad[2].abs().sum() > 0
assert embedding.weight.grad[5].abs().sum() > 0
assert embedding.weight.grad[0].abs().sum() == 0
```

## 本章调试不变量

1. 所有 token id 都满足 `0 <= id < vocab_size`。
2. `len(tokenizer)` 与模型 embedding/output 词表轴一致。
3. PAD、BOS、EOS、UNK 的 id 与数据制作阶段完全一致。
4. 同一个 tokenizer 配置下，训练与推理的 special-token 策略一致。
5. `input_ids.dtype == torch.long`。
6. Logits 最后一维等于当前词表大小，交叉熵输入未提前 Softmax。
7. Decode 异常时同时检查 token 字符串、id 和原始 `repr(text)`。

## 常见错误

1. 把 PAD id 写死为 0。
2. 更换 tokenizer 后继续加载旧 embedding。
3. 预处理时忘记 EOS，导致文档边界消失。
4. 把 token id 当作有大小关系的连续特征。
5. 对 logits 手动 softmax 后再调用交叉熵。
6. 忽略 tokenizer 新增 token 后词表大小发生变化。

## 动手练习

1. 用项目 tokenizer 编码中英文各三句话，比较 token 数量。
2. 打印 PAD、BOS、EOS 的 id，确认它们是否互不相同。
3. 创建一个 `nn.Embedding(10, 3)`，观察相同 id 是否得到相同向量。
4. 手算三个 logits 的 softmax 和正确类别 loss，再与 PyTorch 对比。

## 自测

1. 为什么 tokenizer 不是神经网络主体的一部分？
2. 为什么 id 100 不表示比 id 10 更重要？
3. 为什么新增 PAD token 后模型词表大小必须同步变化？
4. Logits 与概率有什么区别？
5. `V=50,000,D=512` 是否意味着 50,000 个 token 中必然有相同向量？为什么？
6. 从 one-hot 到 embedding 为什么是有损映射，却仍可能改善语言建模？
7. 同一个 token id 在两个句子中的初始 embedding 和最终 hidden state分别是否相同？
8. 增大 `D` 会同时增加哪些参数、activation 和计算成本？
