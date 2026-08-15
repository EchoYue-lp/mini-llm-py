# 06 Encoder-Decoder 与翻译训练

Encoder-Decoder 适合 source 和 target 不同的任务，例如机器翻译。

## 学习目标

读完后应能：

1. 画出 source、encoder memory 和 target 的完整数据流。
2. 推导三种 mask 与 cross-attention score shape。
3. 解释 teacher forcing、next-token shift 和 exposure bias。
4. 区分训练 loss、生成质量和翻译评测。

## 本章符号与 Shape

| 符号 | 含义 | 典型 Tensor |
| --- | --- | --- |
| `B` | batch size | source ids `[B,Ts]` |
| `Ts` | source 序列长度 | encoder memory `[B,Ts,D]` |
| `Tt` | target/decoder 序列长度 | decoder hidden `[B,Tt,D]` |
| `D` | model dimension | cross-attention Q/K/V 特征宽度 |
| `Vs/Vt` | source/target 词表大小 | target logits `[B,Tt,Vt]` |

Cross-attention 中 Q 来自 decoder `[B,H,Tt,Dh]`，K/V 来自 encoder
`[B,H,Ts,Dh]`，所以 score 是 `[B,H,Tt,Ts]`。不要用一个模糊的 `T` 同时代表 source 与
target 长度。

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
- 随机均匀预测基线 `log(V)` 与训练 loss 的关系。

```bash
python -m labs.lab04_tiny_copy_task --steps 400
```

实验会直接断言 `decoder_input[:,1:] == labels[:,:-1]`，避免只凭 loss 下降判断 shift 正确。

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
- Python、PyTorch 以及 CUDA 的 RNG 状态。

只保存 `state_dict` 可以推理，但无法精确继续训练。本项目的完整 checkpoint 会恢复 RNG；
若使用 `resume_training --epochs N` 开启一个额外训练阶段，则保留 optimizer moments，但从命令
配置的 `lr` 开始一条新的、长度为 N 个 epoch 的衰减曲线，避免越过旧 scheduler 的终点。

## 数据问题比模型问题更常见

翻译质量差时先检查：

1. Source 与 target 行是否一一对齐。
2. 中文是否被错误清洗或拆分。
3. BOS/EOS 是否存在。
4. 过长样本是否大量截断。
5. Train/validation 是否泄漏。
6. Tokenizer 是否与 checkpoint 一致。

## 条件概率分解

翻译模型学习：

$$
P(y_1,\ldots,y_T\mid x_1,\ldots,x_S)
=\prod_{t=1}^{T}P(y_t\mid y_{<t},x_{1:S})
$$

Encoder memory 表示条件 `x`，decoder causal self-attention 表示 `y_{<t}`，cross-attention
把二者结合。训练 loss 是各目标 token 负对数概率之和或平均：

$$
L=-\sum_t\log P(y_t\mid y_{<t},x)
$$

这解释了 Decoder 为什么同时需要两种 Attention：只有 self-attention 会变成无条件语言
模型；只有 cross-attention 则无法利用已生成 target 的语法与上下文。

## Cross-Attention 的逐元素计算

Decoder hidden 产生 query，encoder memory 产生 key/value：

```text
decoder hidden: [B,T,D]
memory:         [B,S,D]
Q:              [B,H,T,Dh]
K/V:            [B,H,S,Dh]
score:          [B,H,T,S]
```

对 target 位置 `t`、source 位置 `s`：

$$
score_{t,s}=\frac{q_t\cdot k_s}{\sqrt{D_h}}
$$

每个 target query 沿 source 轴 `S` 做 Softmax，再汇总 encoder Value。Cross mask 的 key 轴
必须对应 source padding；把 target mask 传进去，在 `S==T` 的 batch 中甚至可能不报错，却
会屏蔽错误位置。

## Teacher Forcing 为什么能并行

训练时完整 target 已知，但 causal mask 保证位置 `t` 只读取 `<=t` 的 decoder input。所有
位置的条件历史可以一次放入矩阵：

```text
decoder_input = [BOS, y1, y2, ..., y(T-1)]
labels        = [y1,  y2, y3, ..., yT]
```

并行计算不等于允许看未来。若关闭 causal mask，位置 `t` 可直接读取包含答案的后续 target
embedding，训练 loss 会虚假下降，生成时却无法复现该信息路径。

Exposure bias 描述训练使用真实历史、推理使用模型历史的分布差异。它是真实问题，但不是
看到生成错误就能唯一归因的诊断标签；数据、搜索、校准和模型容量也可能是原因。

## 特殊 Token 与截断边界

一个目标序列通常需要：

```text
[BOS, content..., EOS]
```

若在截断时简单取前 `max_len` 个 token，可能把 EOS 截掉，使模型看不到停止监督。更稳妥的
策略是预留 EOS 位置：

```python
content = content[: max_len - 2]
target = [bos_id] + content + [eos_id]
```

`collate_fn_with_padding` 和 SentencePiece 平行语料预处理均按此规则保留目标 EOS；预处理还会
按原始行号成对读取双语文本，任意一侧缺行或单侧空行都会立即报错。

本项目预处理、collate 和训练脚本之间必须约定由谁添加特殊 token。若 tokenizer 已自动加
BOS/EOS，数据代码再加一次，会出现重复边界。

## Token 平均与 Sequence 平均

`CrossEntropyLoss(ignore_index=PAD)` 默认对所有有效 target token 求平均。长句贡献更多
token 项，但每个 token 权重相同。另一种做法是先计算每句平均，再对 batch 求平均，此时短句
和长句权重相同。

两种目标不同，不能只比较标量 loss 而忽略 reduction。日志中若用“batch loss 的平均”汇总
不同有效 token 数的 batch，也可能产生偏差。严格 corpus loss 应累计：

```text
sum(valid token NLL) / sum(valid token count)
```

## Beam Search 的分数

序列概率连乘会快速下溢，所以 Beam 累加 log probability：

$$
score(y_{1:t})=\sum_{i=1}^{t}\log P(y_i\mid y_{<i},x)
$$

因为每个 log probability 小于等于 0，长序列天然累加更多负数，容易偏向过短输出。项目
使用近似长度惩罚：

```text
normalized_score = log_prob / length^alpha
```

`alpha` 不是概率模型参数，而是搜索启发式。比较实验时必须固定 beam width、length penalty、
最大长度和停止条件，否则 BLEU 变化可能来自解码器而非模型。

## Encoder 复用在当前代码中的差异

`beam_search_translate()` 手动执行一次 encoder，并在所有 beam/step 中复用 memory。这符合
理想推理路径。

`greedy_translate()` 为了代码简单，每步调用完整 `model(src,tgt,...)`，因此会重复运行
encoder。输出语义仍正确，但计算浪费。把它优化为生产路径时，应拆出：

```python
memory = encode(src, src_mask)       # once
next_logits = decode(tgt, memory)    # every step
```

进一步还可缓存 decoder self-attention K/V；cross-attention 的 encoder K/V 也可预投影复用。

## 一批数据的可执行 Shape Trace

```python
src, tgt = next(iter(train_loader))
src = src.to(device)
tgt = tgt.to(device)

tgt_input = tgt[:, :-1]
labels = tgt[:, 1:]
src_mask = create_padding_mask(src, pad_token_id)
tgt_mask = combine_masks(
    create_causal_mask(tgt_input.size(1), src.device),
    create_padding_mask(tgt_input, pad_token_id),
)
cross_mask = create_padding_mask(src, pad_token_id)

logits, attention = model(
    src,
    tgt_input,
    src_mask=src_mask,
    tgt_mask=tgt_mask,
    cross_mask=cross_mask,
)

assert logits.shape[:2] == labels.shape
assert src_mask.shape == (src.size(0), 1, 1, src.size(1))
assert cross_mask.size(-1) == src.size(1)
```

## 从零诊断翻译训练

按以下顺序能减少无效调参：

1. 检查随机样本的 source/target 文本与 id 是否对齐。
2. 在极小数据上过拟合一个 batch，确认 loss 能接近零。
3. 关闭 dropout，固定 seed，验证 train/eval forward 可重复。
4. 打印三种 mask 的一条样本，确认可见位置。
5. Greedy 解码 copy task，确认 BOS/EOS 和停止逻辑。
6. 再扩大数据并比较 validation token NLL 与生成样例。
7. 最后才调整 beam、长度惩罚和模型规模。

如果连一个 batch 都无法过拟合，优先寻找数据、shift、mask、优化器或实现错误，而不是增加
层数。

## 本章调试不变量

1. `src/tgt` batch size 相同，source/target 样本一一对应。
2. `tgt_input.shape == labels.shape`，内容严格错开一位。
3. Target self-mask key 轴是 target length，cross-mask key 轴是 source length。
4. Encoder memory 在生成过程中不随 target step 改变。
5. BOS/EOS 添加一次且 id 与 tokenizer/checkpoint 一致。
6. Loss 与 BLEU/生成样例分别记录，不能互相替代。
7. Beam 对 token 序列、累计分数和 cache 的重排保持同步。

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
