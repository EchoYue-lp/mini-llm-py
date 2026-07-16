# 01 Tokenizer、Embedding 与 Logits

模型不直接处理字符串。文本先变成 token id，再通过 embedding 变成连续向量。

## 学习目标

读完后应能：

1. 解释字符、单词和子词 tokenizer 的取舍。
2. 推导 `[B,T]` token id 到 `[B,T,D]` hidden state 的过程。
3. 区分 token id、embedding、logits、probability 和 loss。
4. 说明 PAD、BOS、EOS 为什么影响模型结构与训练。

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
