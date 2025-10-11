# Transformer 英中翻译与文本生成

基于 PyTorch 从零实现的 Transformer 模型，支持英中机器翻译和文本生成任务。

## 🌟 项目特点

- **双架构支持**
  - Encoder-Decoder：用于英中翻译（Pre-LN，6层编码器+6层解码器）
  - Decoder-Only：用于文本生成（Pre-LN，2层解码器）

- **智能分词方案**
  - 翻译任务：自训练 SentencePiece（16k词汇，中文高效）
  - 生成任务：GPT-2 Tokenizer（50k词汇，英文覆盖广）

- **生产级训练**
  - 混合精度训练（AMP）- 节省显存 40%，提速 1.5-2x
  - 梯度累积 - 模拟大 batch size
  - 学习率调度 - Warmup + Cosine Decay
  - TensorBoard 可视化 - 实时监控训练进度

- **多种解码策略**
  - Beam Search（推荐用于翻译）
  - Top-P/Top-K 采样（用于创造性生成）
  - Greedy Decoding（快速推理）

## 📦 环境要求

**Python**: 3.8+

**安装依赖**：
```bash
pip install -r requirements.txt
```

**推荐硬件**：
- GPU: NVIDIA RTX 3060 及以上（最低 8GB 显存）
- CPU: 可运行但速度慢

> **注意**: 如果云平台未预装 PyTorch，请在 `requirements.txt` 中取消注释 `torch>=2.0.0` 这一行。

## 🚀 快速开始

### 方案一：英中翻译（推荐）

#### 1. 准备数据

下载 IWSLT2017 英中翻译数据集：

```bash
python scripts/download_datasets.py --translation
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
python scripts/train_sentencepiece.py
```

生成 `tokenization/sentencepiece_enzh.model` 和 `.vocab` 文件。

#### 3. 预处理数据

```bash
python scripts/retokenize_dataset.py
```

将文本转换为 token IDs，保存到 `data/iwslt2017/*_ids_sp.pt`。

#### 4. 训练模型

```bash
python scripts/train_encoder_decoder.py
```

**默认配置**：
- d_model=128, 6层编码器+6层解码器, 4个注意力头
- Batch size=64, 梯度累积=6（有效batch=384）
- 学习率=2e-4（cosine调度+warmup）
- 自动混合精度（AMP）

**训练日志示例**：
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
python scripts/translate.py
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
python scripts/download_datasets.py --generation
```

数据将保存到 `data/wikitext2/`：
```
data/wikitext2/
├── train.txt
├── validation.txt
└── test.txt
```

#### 2. 训练模型

```bash
python scripts/train_decoder.py
```

#### 3. 生成文本

```bash
python scripts/generate.py
```

---

## 📊 训练监控

启动 TensorBoard：
```bash
tensorboard --logdir=/hy-tmp/Net/logs --port=6006
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
| 参数量 | ~7.5M | 包含 embedding |

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
# 显存节省 40-50%
# 速度提升 1.5-2x
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

## 📈 性能基准

**硬件**: RTX 4060 Ti 16GB

| 任务 | 时间/epoch | 显存占用 | 收敛epochs |
|------|-----------|---------|-----------|
| 翻译 | 10-15分钟 | 4-6 GB | 30-50 |
| 生成 | 5-10分钟 | 3-5 GB | 100 |

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
