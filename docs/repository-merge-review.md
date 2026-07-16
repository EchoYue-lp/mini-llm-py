# 两个学习仓库的审查与合并建议

审查对象：

- `mini-llm-py`：PyTorch 从零实现 Transformer、翻译和文本生成。
- `llm-posttrain-lab`：MLX + Qwen3 的 LoRA、工具路由数据、评测和过拟合实验。

## 1. 结论

合并到 `mini-llm-py` 是合理的，因为两者可以组成一条完整学习路径：

```text
Transformer 数学与从零实现
  -> 预训练目标和生成
  -> 使用真实预训练模型
  -> SFT / LoRA
  -> 数据校验和任务评测
  -> DPO / GRPO / MoE 等后续主题
```

但不建议把文件直接平铺到同一目录，也不建议合并成一个 `requirements.txt`。PyTorch
从零训练与 Apple MLX 后训练是两个运行时边界，应在同一个学习型 monorepo 中保留独立
子项目和独立环境。

## 2. 能互补的部分

| 维度 | `mini-llm-py` | `llm-posttrain-lab` |
| --- | --- | --- |
| 学习阶段 | 架构基础、从零训练 | 预训练模型后训练 |
| 框架 | PyTorch | MLX / MLX-LM |
| 模型 | 自定义小 Transformer | Qwen3-0.6B |
| 任务 | 翻译、语言模型 | 工具路由 SFT |
| 强项 | Attention、mask、Pre-LN 可见 | LoRA、prompt mask、checkpoint、评测闭环 |
| 硬件 | CPU/CUDA/MPS | Apple Silicon / Metal |

主题没有重复到需要二选一，反而正好连接“模型是怎样构成的”和“已有模型怎样被任务化”。

## 3. 推荐目录

第一阶段不移动现有 `mini-llm-py` 核心代码，只把后训练仓库放到明确子目录：

```text
mini-llm-py/
├── README.md
├── docs/
├── labs/
├── models/
├── scripts/
├── utils/
├── test/
└── posttraining/
    └── mlx_tool_router/
        ├── README.md
        ├── requirements.txt
        ├── train_lora_short.py
        ├── train_lora_long.py
        ├── evaluate.py
        ├── compare_models.py
        ├── tool_router.py
        ├── validate_data.py
        ├── scripts/
        └── docs/
```

后续若要重构包结构，再把从零实现移动到 `from_scratch/`。第一步就大规模移动会破坏现有
import、命令、checkpoint 路径和 Git 历史，收益不高。

## 4. 依赖必须分开

根项目：

```text
PyTorch + Transformers + SentencePiece + datasets + TensorBoard
```

MLX 后训练子项目：

```text
mlx-lm + huggingface-hub
```

MLX 对平台有明确要求。把它放入根依赖会让非 macOS 用户无法完成最基础的 Transformer
实验，也会让 CI 变得不必要地复杂。

## 5. 不应迁移的运行产物

以下目录应继续被忽略：

```text
.venv/
models/
data/
adapters/
results/
*.pt / *.safetensors
```

模型权重约 GB 级，不应进入普通 Git 历史。只迁移代码、配置、少量确定性示例和文档。

## 6. 正确性审查：`mini-llm-py`

### 已确认正确

- Scaled dot-product attention 的缩放维度正确。
- causal mask 和 key padding mask 的广播 shape 正确。
- Pre-LN 残差顺序和 final LayerNorm 正确。
- Encoder-Decoder cross-attention 使用 decoder query、encoder K/V。
- embedding scaling 在训练和 beam-search 手动 encoder/decoder 路径中一致。
- 位置编码支持奇数和偶数 `d_model`。
- 核心前向、padding、梯度流测试可通过。

### 审查中修复

- PyTorch 原本被注释为可选依赖，导致按 README 安装后无法运行。
- GPT-2 没有 PAD，旧代码把真实 token id `0` 当作 PAD。
- WikiText 预处理的 range 会漏掉最后一个完整 chunk。
- 语言模型语料没有 EOS 边界，模型几乎学不到停止信号。
- Decoder Top-P 没有包含第一个使累计概率越过 `p` 的 token。
- 翻译梯度累积按 micro-batch 计算 scheduler 总步数，cosine 曲线走不完整。
- 最后一个不足累积长度的 group 使用了错误的 loss 除数。
- 一个 beam-search 测试使用 1000 词模型配 50257 词 GPT-2 tokenizer，测试本身越界。
- TensorBoard 路径硬编码为云主机目录，现改为项目内 `runs/`。

### 仍需明确的限制

- Decoder-Only 复用了含 cross-attention 的 `DecoderLayer`，有未使用参数，但不影响输出。
- 训练结果依赖外部数据和较长训练，本次审查只验证小规模数值与代码路径。
- 现有翻译质量示例和硬件性能数字不是自动基准，不能当作可复现实验结论。

`resume_training.py` 已在合并后的后续提交中接入两个真实训练循环。恢复时会按 checkpoint
配置重建模型，并加载权重、优化器、scheduler、epoch 和可用的 AMP Scaler 状态；词表不一致
会直接报错。

## 7. 正确性审查：`llm-posttrain-lab`

### 已确认正确

- Qwen chat template 的 prompt token 确实是完整训练序列前缀。
- LoRA 使用 `A` 随机、`B` 为零，初始增量为零。
- 基座权重被冻结，只保存可训练 adapter 参数。
- `Delta-W` shape 与 MLX Linear 的 `[out, in]` 布局一致。
- adapter metadata、目标层路径和 MLX-LM 加载格式一致。
- train/valid/test 分离，validation 选择 best checkpoint，test 不参与训练决策。
- 数据生成得到 38/5/5，共 48 条；schema、action 和 split 重复校验通过。
- 仓库已有 base、短训、长训结果和过拟合分析，指标口径清楚。

### 审查中修复

变长 batch 的 loss mask 原本使用：

```text
position <= real_sequence_length
```

next-token shift 后，有效 target 位置应为 `1..L-1`，所以正确边界是：

```text
position < real_sequence_length
```

旧实现会把较短样本的第一个 padding token 计入 loss。默认 `batch_size=1` 时没有显现，
但文档允许扩大 batch，因此必须修复。

### Metal 回归

本次已在 Apple Silicon Metal 上使用 `batch_size=2`、最后 1 层 Q/V LoRA 完成 2-step
smoke training。验证 loss 从 `0.9983` 降到 `0.9832`，峰值记录约 `1.834GB`；生成的
adapter 可以被 `tool_router.py` 重新加载并完成一条推理。两步训练不用于判断任务质量，
但已覆盖变长 batch loss mask、梯度更新、验证、adapter 保存和重载链路。

## 8. 直接迁移执行记录

最终采用代码快照迁移：将 `llm-posttrain-lab` 当前已验证的代码直接放入
`posttraining/mlx_tool_router`，但不把源仓库提交历史接入 `mini-llm-py` 主线。这样目录就是
普通项目子目录，不需要专用同步命令、双向同步或两套发布流程。

后续改动只提交到 `mini-llm-py`。原 `llm-posttrain-lab` 不再维护或同步；是否删除其本地目录
应单独处理，避免误删被 `.gitignore` 排除的模型、adapter 和实验结果。

## 9. 合并验收清单

- [x] 根 README 有统一学习路线。
- [x] 两套依赖分别安装，不互相强制。
- [x] PyTorch 单元测试通过。
- [x] 12 个 `labs/` 实验能从根目录运行。
- [x] MLX 数据校验通过。
- [x] MLX 1 至 5 step smoke training 能保存并重新加载 adapter。
- [x] base 与 adapter 至少完成 1 条固定样本推理。
- [x] 根 `.gitignore` 不允许模型、adapter、数据和环境进入 Git。
- [x] 文档中的命令均以各自子项目根目录为 cwd。

以上验收项已经完成。合并提升了学习连贯性，同时仍以独立依赖环境隔离 PyTorch 与 MLX。
