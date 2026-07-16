# Transformer 英中翻译与文本生成

基于 PyTorch 从零实现的 Transformer 模型，支持英中机器翻译和文本生成任务。

## 📚 学习路线

这个仓库现在分为三层：

1. `labs/`：单概念、可直接运行的小实验。
2. `models/`、`utils/`：从零实现的完整 Transformer 组件。
3. `scripts/`：数据、训练、checkpoint 和生成闭环。

建议先读 [文档索引](docs/README.md)，再按顺序运行 `labs/`。内容覆盖位置编码、
Self-Attention、Multi-Head Attention、Decoder/CausalLM、KV Cache、LoRA、
MHA/MQA/GQA、Dense/Sparse/Shared-Expert MoE，以及训练这些结构所需的 mask、归一化、
FFN、next-token loss 和解码策略。

- [Transformer 基础：从 token 到 logits](docs/transformer-fundamentals.md)
- [现代 LLM 组件与 MoE](docs/modern-llm-and-moe.md)
- [两个学习仓库的审查与合并建议](docs/repository-merge-review.md)
- [12 个 Transformer 小实验](labs/README.md)

## 🌟 项目特点

- **双架构支持**
  - Encoder-Decoder：用于英中翻译（Pre-LN，6层编码器+6层解码器）
  - Decoder-Only：用于文本生成（Pre-LN，2层解码器）

- **智能分词方案**
  - 翻译任务：自训练 SentencePiece（16k词汇，中文高效）
  - 生成任务：GPT-2 Tokenizer（50k词汇，英文覆盖广）

- **完整训练机制**
  - 混合精度训练（AMP）- 在支持的 CUDA 设备上降低显存并提高吞吐
  - 梯度累积 - 模拟大 batch size
  - 学习率调度 - Warmup + Cosine Decay
  - TensorBoard 可视化 - 实时监控训练进度

- **多种解码策略**
  - Beam Search（推荐用于翻译）
  - Top-P/Top-K 采样（用于创造性生成）
  - Greedy Decoding（快速推理）

## 📦 环境要求

**Python**: 3.10+（推荐 3.11 或 3.12）

**安装依赖**：
```bash
pip install -r requirements.txt
```

### 硬件分档

本项目默认模型很小：Decoder-Only 约 `13.45M` 参数，Encoder-Decoder 约 `8.94M`
参数；FP32 权重本身分别约 `51 MiB` 和 `34 MiB`。训练时还需要梯度、Adam 一阶/二阶
状态和激活，显存/内存消耗主要由 `batch_size`、序列长度和层数决定，而不是只看权重文件。

| 使用范围 | 最小配置（能运行） | 推荐配置（体验较好） | 说明 |
| --- | --- | --- | --- |
| 文档、单元测试、`lab00-03`、`lab06-11` | 2 核 CPU、4 GB RAM | 4 核以上 CPU、8 GB RAM | 不需要 GPU，单次运行通常为秒级 |
| 微型训练 `lab04-05` | 4 核 CPU、4 GB RAM | 8 核 CPU、8 GB RAM，或任意可用 CUDA/MPS GPU | 默认任务可在 CPU 上完成，用于验证 loss 下降和生成结果 |
| 完整 Transformer 前向、推理和小 batch smoke training | 4 核 CPU、8 GB RAM；或 4 GB 显存/统一内存余量 | 8 GB 以上显存，或 16 GB Apple 统一内存 | 建议先用 `batch_size=4-16`、`max_len=32-64` 验证链路 |
| 默认 IWSLT/WikiText 完整训练 | 6 GB 显存；或 16 GB Apple 统一内存；CPU 需 16 GB RAM 但很慢 | 8-12 GB CUDA 显存 + 16 GB RAM；或 24 GB Apple 统一内存 | 默认 `batch_size` 较大，长期训练优先 CUDA；4 GB GPU 需明显降低 batch/长度 |

磁盘建议至少预留 `10 GB`，推荐 `20 GB`，用于 Python 环境、原始数据、tokenized 数据、
checkpoint 和 TensorBoard 日志。若后续合并 Qwen/LoRA 后训练子项目，还需额外预留模型与
adapter 空间。

### 不同设备的注意事项

- **CUDA**：当前训练脚本会在 CUDA 上启用 autocast 和 GradScaler；这是完整训练的首选。
  参考 [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。
- **Apple Silicon / MPS**：PyTorch 可通过 [MPS 后端](https://docs.pytorch.org/docs/stable/notes/mps.html)
  使用 Metal GPU，但当前项目的 AMP 分支
  只针对 CUDA，因此按 FP32 和更高统一内存占用预估。统一内存还要与 macOS 和其他应用共享。
- **CPU**：所有实验和测试均可运行，适合理解原理和 smoke test；不建议用 CPU 跑多轮完整
  数据集训练。

“最小配置”表示通过降低 batch 或序列长度能够完成流程，不代表保持 README 默认参数时
仍有理想速度。

`requirements.txt` 已包含 PyTorch；若云平台要求特定 CUDA wheel，请按平台说明先安装
对应 PyTorch，再安装其余依赖。

## 🚀 快速开始

### 方案一：英中翻译（推荐）

#### 1. 准备数据

下载 IWSLT2017 英中翻译数据集：

```bash
python -m scripts.download_datasets --translation
```

数据将保存到 `data/iwslt2017/`：
```
data/iwslt2017/
├── train.en.txt / train.zh.txt
├── validation.en.txt / validation.zh.txt
└── test.en.txt / test.zh.txt
```

#### 2. 训练 SentencePiece 分词器

```bash
python -m scripts.train_sentencepiece
```

生成 `tokenization/sentencepiece_enzh.model` 和 `.vocab` 文件。

#### 3. 预处理数据

```bash
python -m scripts.retokenize_dataset
```

将文本转换为 token IDs，保存到 `data/iwslt2017/*_ids_sp.pt`。

#### 4. 训练模型

```bash
python -m scripts.train_encoder_decoder
```

**默认配置**：
- d_model=128, 6层编码器+6层解码器, 4个注意力头
- Batch size=64, 梯度累积=6（有效batch=384）
- 学习率=2e-4（cosine调度+warmup）
- 自动混合精度（AMP）

**训练日志格式示意**（实际 loss 和翻译结果取决于数据与训练时长）：
```
Epoch 1 Train Loss: 5.2341, LR: 2.50e-04
Epoch 1 Val Loss: 4.8765
────────────────────────────────────────
Demo 翻译测试:
  EN: Hello, how are you?
  ZH: 你好，你好吗？
────────────────────────────────────────
✓ 保存最佳模型 (Val Loss: 4.8765)
```

#### 5. 测试翻译

```bash
python -m scripts.translate
```

交互式翻译：
```
EN> Hello world
策略 [1-Beam/2-TopP/3-Greedy/4-TopK] (默认1): 1
ZH> 你好世界
```

---

### 方案二：文本生成

#### 1. 准备数据

下载 WikiText-2 文本生成数据集：

```bash
python -m scripts.download_datasets --generation
```

数据将保存到 `data/wikitext2/`：
```
data/wikitext2/
├── train.txt
├── validation.txt
└── test.txt
```

#### 2. 训练模型

先把文本转换为带 EOS 边界的固定长度 token 序列：

```bash
python -m scripts.preprocess --generation
```

再训练模型：

```bash
python -m scripts.train_decoder
```

#### 3. 生成文本

```bash
python -m scripts.generate
```

---

## 📊 训练监控

启动 TensorBoard：
```bash
tensorboard --logdir=runs --port=6006
```

访问 `http://localhost:6006` 查看：
- 训练/验证损失曲线
- 学习率变化
- 多模型对比

## 📁 项目结构

```
.
├── models/                      # 模型定义
│   ├── transformer_models.py   # Encoder-Decoder 和 Decoder-Only 模型
│   ├── layers.py                # Multi-Head Attention, FFN, 位置编码
│   └── decoder_encoder_layer.py # Encoder/Decoder 层实现
│
├── scripts/                     # 训练和推理脚本
│   ├── train_encoder_decoder.py # 翻译模型训练
│   ├── train_decoder.py         # 生成模型训练
│   ├── translate.py             # 翻译推理
│   ├── generate.py              # 文本生成
│   ├── train_sentencepiece.py  # 训练 SentencePiece
│   └── retokenize_dataset.py   # 数据预处理
│
├── utils/                       # 工具函数
│   ├── translation_utils.py    # Beam Search, Top-P 等解码策略
│   ├── generation_utils.py     # 文本生成工具
│   ├── mask_utils.py            # Attention Mask 创建
│   ├── scheduler_utils.py       # 学习率调度器
│   ├── sentencepiece_tokenizer.py # SentencePiece 封装
│   └── checkpoint_utils.py      # 模型保存/加载
│
├── labs/                        # 单概念可运行实验（Attention 到 MoE/LoRA）
├── docs/                        # Transformer、现代 LLM、合并审查文档
├── data/                        # 数据目录
│   ├── iwslt2017/              # 翻译数据集
│   └── wikitext2/              # 生成数据集
│
├── tokenization/               # 分词器
│   ├── gpt2/                   # GPT-2 tokenizer
│   └── sentencepiece_enzh.model # 自训练 SentencePiece
│
└── test/                       # 单元测试
    ├── test_padding_mask.py
    ├── test_beam_search.py
    └── ...
```

## ⚙️ 模型配置

### 翻译模型（Encoder-Decoder）

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 128 | 模型维度 |
| 层数 | 6+6 | 编码器+解码器 |
| 注意力头数 | 4 | Multi-Head Attention |
| FFN 维度 | 512 | 前馈网络 |
| 最大序列长度 | 96 | tokens |
| Dropout | 0.1 | 正则化 |
| Tokenizer | SentencePiece | 16k词汇 |
| 参数量 | ~8.94M | 当前代码实测，包含 embedding |

### 生成模型（Decoder-Only）

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 128 | 模型维度 |
| 层数 | 2 | Decoder层 |
| 注意力头数 | 4 | Multi-Head Attention |
| FFN 维度 | 512 | 前馈网络 |
| 最大序列长度 | 96 | tokens |
| Tokenizer | GPT-2 | 50k词汇 |
| 参数量 | ~13.5M | 包含 embedding |

## 🔧 训练技巧

### 1. 混合精度训练（AMP）
```python
use_amp = True  # CUDA only
# 实际显存与速度收益取决于 GPU、batch、序列长度和算子
```

### 2. 梯度累积
```python
batch_size = 64
gradient_accumulation_steps = 6
# 有效 batch size = 384
```

### 3. 学习率调度
- Warmup: 前 5-10% steps 线性增长
- Cosine Decay: 后续平滑衰减到 0

### 4. 梯度裁剪
```python
max_grad_norm = 1.0  # 防止梯度爆炸
```

## 💡 常见问题

### Q: 显存不足（OOM）？
**A**: 减小 batch_size 或增加 gradient_accumulation_steps：
```python
batch_size = 32  # 从 64 减少
gradient_accumulation_steps = 12  # 从 6 增加
```

### Q: 训练速度慢？
**A**:
1. 启用 AMP（仅 CUDA）：`use_amp = True`
2. 减少 max_len：`max_len = 64`
3. 使用更小的模型：`d_model = 64, num_layers = 4`

### Q: 翻译质量差？
**A**:
1. 训练更多 epochs（30-50）
2. 增大模型：`d_model = 256, num_layers = 12`
3. 增加数据集
4. 调整 beam_width（3-10）

### Q: 为什么用不同的 Tokenizer？
**A**:

| 任务 | Tokenizer | 中文效率 | 原因 |
|------|-----------|----------|------|
| 翻译 | SentencePiece (16k) | ✅ 高 | 1汉字≈1token，序列更短 |
| 生成 | GPT-2 (50k) | ❌ 低 | 1汉字≈3-5tokens，但英文覆盖广 |

**示例**："我喜欢机器翻译"
- SentencePiece: ~14 tokens
- GPT-2: ~80 tokens
- **效率提升**: 5.7x

## 📈 性能测量

训练时间和峰值显存受 PyTorch/CUDA 版本、数据长度分布、磁盘、设备功耗和后台进程影响，
不再把单次机器记录当作通用基准。比较硬件或参数时，应固定随机种子和训练配置，并记录：

- 每个 epoch 的 wall-clock 时间和 tokens/s；
- CUDA 的 `torch.cuda.max_memory_allocated()` 峰值；
- train/validation loss，而不是只比较速度；
- `batch_size`、梯度累积、`max_len`、AMP 状态和软件版本。

对本项目而言，若只购买或租用一套通用学习环境，优先级通常是：`8-12 GB` CUDA 显存、
`16 GB` 系统内存、`20 GB` 可用磁盘；若还要运行 MLX/Qwen 后训练，Apple Silicon 建议
选择 `24 GB` 统一内存，`16 GB` 可完成当前 0.6B 教学实验但余量较小。

## 🎯 解码策略对比

| 策略 | 特点 | 适用场景 | 速度 |
|------|------|----------|------|
| Beam Search | 质量最高 | ⭐ 翻译（推荐） | 慢 |
| Top-P | 多样性强 | 创意写作 | 中等 |
| Greedy | 确定性 | 快速测试 | 快 |
| Top-K | 可控随机 | 对话生成 | 中等 |

## 🛠 开发计划

- [ ] 支持更多语言对（中英、法英等）
- [ ] 添加 BLEU/METEOR 自动评测
- [ ] 支持分布式训练（DDP）
- [ ] Web UI 界面
- [ ] 模型量化（INT8）和 ONNX 导出

## 📄 License

MIT License

## 🙏 致谢

- [IWSLT2017](https://wit3.fbk.eu/) - 翻译数据集
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Transformers](https://huggingface.co/transformers/) - Tokenizer
- [SentencePiece](https://github.com/google/sentencepiece) - 分词工具

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [On Layer Normalization in Transformers](https://arxiv.org/abs/2002.04745) - Pre-LN 架构
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化教程
