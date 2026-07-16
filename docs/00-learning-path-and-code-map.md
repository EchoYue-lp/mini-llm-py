# 00 学习路线与代码地图

根目录 `README.md` 负责环境、命令和完整项目说明。本目录只解释模型原理与实验。

## 推荐顺序

| 顺序 | 文档 | 对应实验 |
| --- | --- | --- |
| 00 | 学习路线与代码地图 | 全部 |
| 01 | Tokenizer、Embedding 与 Logits | 训练脚本 |
| 02 | 位置编码与 RoPE | `lab00` |
| 03 | Scaled Attention 与 Mask | `lab01` |
| 04 | Multi-Head Attention | `lab02` |
| 05 | FFN、残差与 Pre-LN Block | `lab03` |
| 06 | Encoder-Decoder 与翻译训练 | `lab04` |
| 07 | Decoder-Only、Loss 与生成 | `lab05` |
| 08 | KV Cache、MHA、MQA 与 GQA | `lab06`、`lab10` |
| 09 | RMSNorm、RoPE 与 SwiGLU | `lab07` |
| 10 | MoE Router、Capacity 与专家 | `lab08`、`lab11` |
| 11 | LoRA 低秩适配原理 | `lab09` |
| 12 | LoRA 训练、Checkpoint 与过拟合 | MLX 训练代码 |
| 13 | 工具路由数据与评测 | 数据与评测代码 |
| 14 | MLX LoRA 完整实验 | 完整后训练流程 |

## 实验命令

```bash
python -m labs.lab00_positional_encoding
python -m labs.lab01_attention_basics
python -m labs.lab02_multi_head_attention
python -m labs.lab03_pre_ln_block
python -m labs.lab04_tiny_copy_task --steps 400
python -m labs.lab05_tiny_language_model --steps 100
python -m labs.lab06_kv_cache
python -m labs.lab07_modern_blocks
python -m labs.lab08_moe_routing
python -m labs.lab09_lora_linear
python -m labs.lab10_mha_mqa_gqa
python -m labs.lab11_moe_variants
```

## 源码阅读顺序

```text
models/layers.py
  -> models/decoder_encoder_layer.py
  -> models/transformer_models.py
  -> utils/mask_utils.py
  -> scripts/train_decoder.py
  -> scripts/train_encoder_decoder.py
  -> utils/generation_utils.py
  -> utils/translation_utils.py
  -> finetuning/train_lora_short.py
  -> evaluation/tool_router.py
```

## 学习方法

每次只追踪四件事：

1. 输入和输出 shape。
2. 当前 token 能看到哪些位置。
3. 哪些参数参与训练。
4. loss 对应哪个目标 token。

只要这四项明确，大多数 Transformer 代码都可以逐层推导。

## 开始前需要恢复的最低数学

不需要先学完整的线性代数和概率论，但下面这些概念必须能看懂：

### 向量与矩阵

```text
向量 x: [D]
矩阵 W: [D, M]
x @ W: [M]
```

矩阵乘法最重要的规则是中间维相同。阅读代码时先看 shape，通常比先看变量名更可靠。

### Softmax

Softmax 将任意分数转换成概率：

```text
softmax([2, 1, 0]) ~= [0.665, 0.245, 0.090]
```

所有值为正且总和为 1。分数差越大，概率越集中。

### Log 与交叉熵

若正确 token 的概率是 `p`：

```text
loss = -log(p)
```

| 正确概率 | Loss |
| ---: | ---: |
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |

模型越确信正确答案，loss 越小。

### 梯度

梯度表示“参数改变一点，loss 会朝哪个方向变化”。训练循环的本质是：

```text
forward -> loss -> backward -> update parameters
```

暂时不需要手推复杂导数，但要知道哪些参数被冻结、哪些参数会收到梯度。

## 每个阶段的达标标准

| 阶段 | 至少应能回答 |
| --- | --- |
| Tokenizer | 文本为什么必须变成 id，PAD 为什么不能参与 loss |
| Embedding | `[B,T]` 如何变成 `[B,T,D]` |
| Attention | `QK^T` 为什么得到 token-to-token 权重 |
| Mask | 某个 token 当前允许看到哪些位置 |
| Block | Attention、FFN、残差和 Norm 各自负责什么 |
| Loss | 第 `t` 个位置到底预测哪个 token |
| Generation | 训练并行、推理逐 token 的区别 |
| KV Cache | 哪些历史计算可以复用 |
| LoRA | 为什么只训练少量矩阵也能适配任务 |
| Evaluation | 为什么训练 loss 低不等于任务正确 |

## 推荐的学习循环

每学习一个主题，按相同流程操作：

1. 先阅读对应文档，不急着运行完整训练。
2. 在纸上写出一个最小 shape 例子。
3. 运行对应 lab，确认输出与推导一致。
4. 修改一个变量，例如 head 数或序列长度。
5. 故意制造一个错误，阅读异常信息。
6. 回到完整源码，找到同一个概念的工程实现。

这种方式比连续阅读大量文章更容易形成长期记忆。

## 建议维护一张 Shape 表

例如 Decoder-Only：

```text
input ids       [B, T]
embedding       [B, T, D]
Q/K/V           [B, H, T, Dh]
attention score [B, H, T, T]
block output    [B, T, D]
logits          [B, T, V]
labels          [B, T]
```

每次调试先更新这张表。若实际 shape 与预期不同，优先修复 shape，而不是继续猜训练参数。

## 自测

完成本项目后，应能不看代码回答：

1. 为什么 causal mask 不能只在 loss 中处理？
2. 为什么 Transformer block 输入输出通常都是 `[B,T,D]`？
3. 为什么训练能一次计算所有位置，而生成必须逐 token？
4. 为什么 KV Cache 保存 K/V 而不是只保存最终 logits？
5. 为什么 LoRA 的 B 通常初始化为零？
6. 为什么工具路由必须看完全正确率，而不能只看语言模型 loss？
