# 06 Encoder-Decoder 与翻译训练

Encoder-Decoder 适合 source 和 target 不同的任务，例如机器翻译。

## 学习目标

读完后应能：

1. 画出 source、encoder memory 和 target 的完整数据流。
2. 推导三种 mask 与 cross-attention score shape。
3. 解释 teacher forcing、next-token shift 和 exposure bias。
4. 区分训练 loss、生成质量和翻译评测。

## 数据流

```text
source ids
  -> source embedding + position
  -> encoder blocks
  -> encoder memory

target ids
  -> target embedding + position
  -> causal self-attention
  -> cross-attention over encoder memory
  -> output logits
```

Encoder 可以双向读取完整 source。Decoder 只能读取已出现的 target token，但可读取完整
encoder memory。

## Teacher Forcing

目标序列：

```text
[BOS, 我, 喜欢, AI, EOS]
```

训练输入和标签错开一位：

```text
input:  [BOS, 我,   喜欢, AI]
label:  [我,  喜欢, AI,   EOS]
```

训练时使用真实历史 token，因此一个 batch 内所有 target 位置可以并行计算。

## 三种 Mask

| Mask | 作用 |
| --- | --- |
| Source padding | Encoder 不读取 source PAD |
| Target causal + padding | Decoder 不看未来，也不读取 target PAD |
| Cross padding | Decoder 不读取 source PAD |

Target PAD 还必须通过 loss 的 `ignore_index` 排除。

## 最小 Copy Task

Copy task 让 target 等于 source，没有语言歧义，适合先验证：

- BOS/EOS。
- teacher forcing。
- causal mask。
- cross-attention。
- next-token shift。

```bash
python -m labs.lab04_tiny_copy_task --steps 400
```

## 完整翻译流程

```bash
python -m scripts.download_datasets --translation
python -m scripts.train_sentencepiece
python -m scripts.retokenize_dataset
python -m scripts.train_encoder_decoder
python -m scripts.translate
```

## 解码

翻译常用 Beam Search：

- 同时保留多个候选。
- 累加 log probability。
- 使用长度惩罚避免过短序列占优。

Greedy 更快，Top-K/Top-P 更有随机性，但翻译通常更强调确定性和完整性。

## 常见错误

1. Encoder 输出忘记 final norm。
2. 手动推理时遗漏 embedding scaling。
3. Cross mask 使用了错误的 target 长度。
4. BOS 与 EOS 相同时初始 beam 被错误判定完成。
5. checkpoint 与 tokenizer 词表不一致。

## 对照源码

- `models/transformer_models.py::EncoderDecoderModel`
- `scripts/train_encoder_decoder.py`
- `utils/translation_utils.py`
- `tests/test_embedding_scaling.py`

## 一个具体的 Batch

假设：

```text
B = 2
source length S = 5
target length T = 4
D = 8
Vtgt = 100
```

Source：

```text
src ids:      [2, 5]
src hidden:   [2, 5, 8]
memory:       [2, 5, 8]
```

Target 输入：

```text
tgt input ids: [2, 4]
tgt hidden:    [2, 4, 8]
```

Cross-attention score：

```text
[B,H,T,S]
```

最终 logits：

```text
[2, 4, 100]
```

Target label：

```text
[2, 4]
```

交叉熵会把 logits 展平为 `[B*T,V]`，label 展平为 `[B*T]`。

## 一次训练迭代

```text
1. DataLoader 取 source/target
2. 移到 device
3. 构造 source padding mask
4. target 右移，得到 decoder input 与 label
5. 构造 target causal + padding mask
6. 构造 cross-attention source mask
7. 前向得到 logits
8. 计算忽略 PAD 的 cross entropy
9. backward
10. 梯度裁剪
11. optimizer.step
12. scheduler.step
```

任何一步使用错误长度，后续都可能 shape 正确但语义错误，因此不能只依赖“代码能运行”。

## Teacher Forcing 与推理差异

训练：

```text
decoder 每个位置读取真实历史 target
```

推理：

```text
decoder 读取模型自己之前生成的 token
```

模型在推理中一旦生成错误 token，后续输入分布会偏离训练数据，这称为 exposure bias。
Beam Search 能保留多个候选，但不能彻底消除该问题。

## 为什么 Encoder 只计算一次

翻译生成过程中 source 不变：

```text
source -> encoder memory
```

Memory 可以在整个 target 解码期间复用。若每生成一个 token 都重新运行 encoder，会浪费
大量计算。

## Loss 与 BLEU

Validation loss 衡量 teacher-forcing 下的 token 概率；BLEU 衡量生成文本与参考译文的
n-gram 重合。二者相关但不等价：

- Loss 低不保证 Beam Search 译文自然。
- BLEU 高也不代表每句翻译都正确。
- 教学项目还应直接查看真实翻译样例。

## Checkpoint 应保存什么

为了恢复训练，需要：

- 模型权重。
- Optimizer 状态。
- Scheduler 状态。
- 当前 epoch。
- 最佳 validation loss。
- 模型结构配置。
- Tokenizer 词表信息。
- CUDA AMP GradScaler 状态。

只保存 `state_dict` 可以推理，但无法精确继续训练。

## 数据问题比模型问题更常见

翻译质量差时先检查：

1. Source 与 target 行是否一一对齐。
2. 中文是否被错误清洗或拆分。
3. BOS/EOS 是否存在。
4. 过长样本是否大量截断。
5. Train/validation 是否泄漏。
6. Tokenizer 是否与 checkpoint 一致。

## 动手练习

1. 用 `B=2,S=5,T=4,D=8` 写出所有 mask shape。
2. 在 copy task 中去掉 cross-attention，观察 loss。
3. 将 target causal mask 关闭，解释为什么训练 loss 会异常好。
4. 比较 Greedy 与 Beam Search 的输出。

## 自测

1. Decoder 为什么需要 self-attention 和 cross-attention 两种注意力？
2. Cross-attention score 的 shape 为什么是 `[T,S]`？
3. Teacher forcing 为什么能并行训练？
4. 为什么恢复训练需要 optimizer 和 scheduler 状态？
