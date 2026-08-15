# mini-llm-py

`mini-llm-py` 是一个从 Transformer 基础实现延伸到 LLM 后训练的学习型项目。
仓库同时包含：

- 数学、NumPy、Pandas 和 PyTorch 的可运行前置课程。
- 使用 PyTorch 从零实现的 Encoder-Decoder 和 Decoder-Only Transformer。
- 从位置编码、Attention 到 KV Cache、LoRA、GQA 和 MoE 的 12 个小实验。
- 英中翻译与文本生成的数据、训练、checkpoint、推理闭环。
- 使用 Apple MLX 和 Qwen3-0.6B 完成的 LoRA 工具路由后训练实验。

项目中的 Python 命令都从仓库根目录以 `python -m package.module` 运行。模型、
adapter、日志和评测结果与源代码分离，统一写入 Git 忽略的运行目录。

## 目录结构

```text
mini-llm-py/
├── docs/             所有专题文档
├── evaluation/       数据校验、模型评测和结果对比
├── finetuning/       MLX LoRA 短训练与长训练
├── foundations/      数学、NumPy、Pandas 与 PyTorch 前置课程
├── inference/        基座模型和 adapter 推理
├── labs/             12 个 Transformer / LLM 小实验
├── models/           PyTorch Transformer 模型实现
├── scripts/          数据准备、模型下载、训练和推理入口
├── tests/            pytest 测试
├── tokenization/     tokenizer 构建代码与 GPT-2 静态资源
├── utils/            mask、checkpoint、生成、调度器和共享路径
├── artifacts/        模型、adapter 和评测结果，本地生成
├── data/             下载或生成的数据，本地生成
├── requirements.txt
├── requirements-dev.txt
└── requirements-mlx.txt
```

源码依赖保持单向：

```text
scripts / labs / tests / finetuning / inference / evaluation
                              |
                              v
                         models + utils
```

`foundations/` 只依赖 Python、NumPy、Pandas 和 PyTorch，不导入 `models/` 或 `utils/`，
因此可以在阅读项目实现之前独立完成。

## 环境要求

项目保留三份依赖文件，它们不是三套互相重复的依赖：

| 文件 | 用途 | 应安装到哪里 |
| --- | --- | --- |
| `requirements.txt` | NumPy/Pandas 前置课程与 PyTorch 运行时 | `.venv` |
| `requirements-dev.txt` | 继承运行时依赖，再增加 pytest | `.venv` |
| `requirements-mlx.txt` | Apple Silicon 的 MLX 后训练 | `.venv-mlx` |

`requirements-dev.txt` 第一行通过 `-r requirements.txt` 复用运行依赖，因此日常开发
只需安装它。MLX 对平台和底层运行时有独立要求，不与 PyTorch 环境合并；否则非
Apple Silicon 用户也会被迫解析 MLX 包，两个框架的升级还会互相影响。

### PyTorch 环境

需要 Python 3.10 或更高版本，推荐 3.11 或 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

只运行项目、不执行测试时，可以安装 `requirements.txt`。

安装后检查环境：

```bash
python check_cloud_env.py
make check
```

### Apple MLX 环境

MLX 训练使用独立环境，避免与根 PyTorch 依赖互相约束：

```bash
python -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mlx.txt
```

该路径面向 Apple Silicon + Metal。已验证 MLX-LM 0.31.3 和
Hugging Face Hub 1.23.0。

## 学习路线

先完成不依赖数据集和 checkpoint 的前置课程：

```bash
python -m foundations.f00_math_basics
python -m foundations.f01_numpy_basics
python -m foundations.f02_embedding_geometry
python -m foundations.f03_pandas_basics
python -m foundations.f04_pytorch_tensors
python -m foundations.f05_pytorch_autograd
python -m foundations.f06_pytorch_training
```

也可以一次运行全部前置课程：

```bash
python -m foundations
```

然后运行 Transformer 实验：

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

一次运行全部自包含实验：

```bash
python -m labs
# 或同时运行 foundations 与 labs
make smoke-core
```

每个实验的核心问题、观察量和修改建议见 [Labs 使用指南](labs/README.md)。

完整学习顺序见
[00 学习路线、代码地图与工程数学基础](docs/00-learning-path-and-code-map.md)。

到第 11 章和 `lab09` 为止是 CPU/CUDA/MPS 均可完成的跨平台主线；第 12-14 章是 Apple
Silicon 上的 MLX 专项实践，不是完成 Transformer 与 LoRA 基础学习的前置条件。

## PyTorch 英中翻译

### 1. 下载数据

```bash
python -m scripts.download_datasets --translation
```

### 2. 训练 SentencePiece

```bash
python -m scripts.train_sentencepiece
```

### 3. 生成 token id 数据

```bash
python -m scripts.retokenize_dataset
```

### 4. 训练 Encoder-Decoder

```bash
python -m scripts.train_encoder_decoder
```

### 5. 交互式翻译

```bash
python -m scripts.translate
```

默认翻译模型使用 `d_model=128`、6 层 encoder、6 层 decoder、4 个注意力头和
SentencePiece 16k 词表，约 8.94M 参数。

## PyTorch 文本生成

### 1. 下载 WikiText-2

```bash
python -m scripts.download_datasets --generation
```

### 2. 预处理

```bash
python -m scripts.preprocess --generation
```

### 3. 训练 Decoder-Only

```bash
python -m scripts.train_decoder
```

### 4. 生成文本

```bash
python -m scripts.generate
```

默认生成模型使用 `d_model=128`、2 层 decoder、4 个注意力头和项目内 GPT-2
tokenizer，约 13.5M 参数。

## checkpoint 与训练监控

查看 checkpoint：

```bash
python -m scripts.resume_training \
  --checkpoint decoder_only_best.pt \
  --action info
```

继续训练：

```bash
python -m scripts.resume_training \
  --checkpoint decoder_only_interrupted.pt \
  --action resume \
  --model_type decoder \
  --epochs 10
```

这里的 `--epochs 10` 表示新增一个 10-epoch 训练阶段：恢复模型、optimizer 和 RNG 状态，
并从命令配置的学习率开始新的衰减周期，不会继续越过旧 scheduler 的终点。

启动 TensorBoard：

```bash
tensorboard --logdir=runs --port=6006
```

## MLX LoRA 工具路由

该实验使用 Qwen3-0.6B 和 50 条教学数据，演示模型下载、基座推理、数据构造、
Schema 校验、LoRA 短训练、LoRA 长训练、adapter 推理和固定测试集评测。

这套数据用于验证工程流程，不代表生产效果。

### 1. 下载模型

```bash
python -m scripts.download_mlx_model
```

首次下载会解析并记录当前 Hugging Face commit；之后默认复用该不可变 revision。要显式选择
版本可运行 `python -m scripts.download_mlx_model --revision <commit-or-tag>`，记录写入模型目录
的 `mini_llm_download_manifest.json`。

模型保存到：

```text
artifacts/models/Qwen3-0.6B/
```

### 2. 运行基座模型

```bash
python -m inference.base_model \
  --prompt "用一句话解释什么是监督微调" \
  --max-tokens 80
```

### 3. 准备并校验数据

```bash
python -m scripts.prepare_tool_router_data
python -m evaluation.validate_tool_router_data
```

数据保存到：

```text
data/tool_router/train.jsonl
data/tool_router/valid.jsonl
data/tool_router/test.jsonl
```

### 4. LoRA 短训练

```bash
python -m finetuning.train_lora_short --dry-run
python -m finetuning.train_lora_short
```

adapter 保存到 `artifacts/adapters/tool-router-short/`。

加载 adapter 推理：

```bash
python -m inference.tool_router \
  "查一下订单A1024到哪里了" \
  --adapter artifacts/adapters/tool-router-short
```

### 5. 评测基座与短训模型

```bash
python -m evaluation.tool_router \
  --label base \
  --output artifacts/results/tool-router/base.json

python -m evaluation.tool_router \
  --adapter artifacts/adapters/tool-router-short \
  --label short-lora \
  --output artifacts/results/tool-router/short-lora.json
```

### 6. LoRA 长训练

```bash
python -m finetuning.train_lora_long --dry-run
python -m finetuning.train_lora_long --iters 300 --num-layers 16
```

adapter 保存到 `artifacts/adapters/tool-router-long/`。长训练会保存最佳验证
checkpoint、最终 checkpoint、训练历史和过拟合分析。

### 7. 三模型对比

```bash
python -m evaluation.compare_tool_router_models
```

详细报告写入：

```text
artifacts/results/tool-router/comparison.md
```

完整流程见
[14 MLX LoRA 工具路由完整流程](docs/14-mlx-lora-tool-router-workflow.md)，
LoRA 原理、训练与评测分别见 11-13。

## 运行产物

以下目录不属于源码，均由命令生成并被 Git 忽略：

| 路径 | 内容 |
| --- | --- |
| `data/iwslt2017/` | 翻译数据 |
| `data/wikitext2/` | 文本生成数据 |
| `data/tool_router/` | MLX 工具路由数据 |
| `artifacts/models/` | 下载的基座模型 |
| `artifacts/adapters/` | LoRA adapter 与训练历史 |
| `artifacts/results/` | JSON 和 Markdown 评测报告 |
| `runs/`、`logs/` | TensorBoard 和训练日志 |
| `*.pt`、`*.pth`、`*.ckpt` | PyTorch checkpoint |

## 硬件建议

| 使用范围 | 最低配置 | 推荐配置 |
| --- | --- | --- |
| 文档、测试、大部分小实验 | 2 核 CPU、4 GB RAM | 4 核 CPU、8 GB RAM |
| PyTorch 微型训练 | 4 核 CPU、4 GB RAM | 8 GB RAM 或可用 CUDA/MPS |
| IWSLT / WikiText 完整训练 | 6 GB GPU 或 16 GB RAM | 8-12 GB CUDA GPU、16 GB RAM |
| MLX Qwen3-0.6B LoRA | 16 GB Apple 统一内存 | 16-24 GB Apple 统一内存 |

建议预留 15-20 GB 磁盘，用于环境、数据、模型、checkpoint、adapter 和报告。

## 开发与测试

```bash
make test-core       # 不需要 Transformers / MLX
make test-pytorch    # 完整 PyTorch 环境
make test-mlx        # Apple Silicon MLX 环境
make test
make check
```

`make check` 会编译所有 Python 源码并运行 pytest。测试应使用小模型、临时目录和
本地 tokenizer，不应依赖网络下载或已有 checkpoint。

新增代码时：

- 模型层放入 `models/`。
- 通用训练、生成和路径逻辑放入 `utils/`。
- 完整命令入口放入 `scripts/`。
- 后训练实现按职责放入 `finetuning/`、`inference/`、`evaluation/`。
- 数学、数据和框架前置课程放入 `foundations/`。
- 教学实验放入 `labs/`。
- 测试放入 `tests/`。
- 专题原理文档放入 `docs/`，课程包入口保留在 `foundations/README.md` 和 `labs/README.md`。

## 文档

- [Foundations 前置课程](foundations/README.md)
- [Transformer Labs 使用指南](labs/README.md)
- [00 学习路线、代码地图与工程数学基础](docs/00-learning-path-and-code-map.md)
- [01 Tokenizer、Embedding 与 Logits](docs/01-tokenization-embedding-and-logits.md)
- [02 位置编码与 RoPE](docs/02-positional-encoding-and-rope.md)
- [03 Scaled Attention 与 Mask](docs/03-scaled-attention-and-masks.md)
- [04 Multi-Head Attention 的 Shape](docs/04-multi-head-attention-shapes.md)
- [05 FFN、残差与 Pre-LN Block](docs/05-transformer-blocks-ffn-and-pre-ln.md)
- [06 Encoder-Decoder 与翻译训练](docs/06-encoder-decoder-translation-training.md)
- [07 Decoder-Only、Loss 与生成](docs/07-decoder-only-loss-and-generation.md)
- [08 KV Cache、MHA、MQA 与 GQA](docs/08-kv-cache-mqa-and-gqa.md)
- [09 RMSNorm、RoPE 与 SwiGLU](docs/09-rmsnorm-rope-and-swiglu.md)
- [10 MoE Router、Capacity 与专家](docs/10-moe-routing-capacity-and-experts.md)
- [11 LoRA 低秩适配原理](docs/11-lora-low-rank-adaptation.md)
- [12 LoRA 训练、Checkpoint 与过拟合](docs/12-lora-training-checkpoints-and-overfitting.md)
- [13 工具路由数据与评测](docs/13-tool-routing-data-and-evaluation.md)
- [14 MLX LoRA 工具路由完整流程](docs/14-mlx-lora-tool-router-workflow.md)
- [原始论文与官方资料](docs/primary-references.md)

## License

[MIT](LICENSE)
