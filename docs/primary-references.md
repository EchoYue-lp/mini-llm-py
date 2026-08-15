# 原始论文与官方资料

学习顺序仍以 00-14 为主。本页用于区分“项目中的教学实现”与论文、框架或模型的原始定义；
当实现存在不同变体时，先看对应章节的 shape 推导，再回到原始资料核对假设。

| 主题 | 原始论文或官方资料 |
| --- | --- |
| Transformer、scaled attention、multi-head attention | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| SentencePiece | [SentencePiece: A simple and language independent subword tokenizer and detokenizer](https://arxiv.org/abs/1808.06226) |
| RoPE | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) |
| Multi-Query Attention | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |
| Grouped-Query Attention | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) |
| RMSNorm | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) |
| SwiGLU / GLU variants | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) |
| Sparse MoE | [Switch Transformers](https://arxiv.org/abs/2101.03961) |
| Shared experts与现代 MoE | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| LoRA | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) |
| PyTorch API 语义 | [PyTorch documentation](https://pytorch.org/docs/stable/index.html) |
| MLX / MLX-LM | [MLX](https://github.com/ml-explore/mlx)、[MLX-LM](https://github.com/ml-explore/mlx-lm) |
| Qwen3-0.6B 模型定义 | [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) |

项目中的 `models/` 是为了暴露张量流和训练契约而写的最小实现，不应被理解为上述论文或
生产框架的逐行复刻。尤其是 mask 表示、位置编码实现、MoE capacity 和 LoRA 注入位置，必须
同时记录所采用的具体变体。
