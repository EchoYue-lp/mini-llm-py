# 学习文档索引

建议按以下顺序阅读和运行：

1. [Transformer 基础：从 token 到 logits](transformer-fundamentals.md)
2. `python labs/lab00_positional_encoding.py`
3. `python labs/lab01_attention_basics.py`
4. `python labs/lab02_multi_head_attention.py`
5. `python labs/lab03_pre_ln_block.py`
6. `python labs/lab04_tiny_copy_task.py --steps 400`
7. `python labs/lab05_tiny_language_model.py --steps 100`
8. [现代 LLM 组件与 MoE](modern-llm-and-moe.md)
9. `python labs/lab06_kv_cache.py` 到 `lab11_moe_variants.py`
10. [两个学习仓库的审查与合并建议](repository-merge-review.md)

## 主题与代码对照

| 主题 | 原理文档 | PyTorch / 手动实现 |
| --- | --- | --- |
| 位置编码 | `transformer-fundamentals.md`、`modern-llm-and-moe.md` | `lab00_positional_encoding.py`、`models/layers.py` |
| Self-Attention | `transformer-fundamentals.md` | `lab01_attention_basics.py` |
| Multi-Head Self-Attention | `transformer-fundamentals.md` | `lab02_multi_head_attention.py`、`models/layers.py` |
| Transformer Decoder / CausalLM | `transformer-fundamentals.md` | `lab05_tiny_language_model.py`、`models/transformer_models.py` |
| LoRA | `modern-llm-and-moe.md` | `lab09_lora_linear.py` |
| MHA / MQA / GQA | `modern-llm-and-moe.md` | `lab10_mha_mqa_gqa.py` |
| Dense / Sparse MoE | `modern-llm-and-moe.md` | `lab08_moe_routing.py`、`lab11_moe_variants.py` |
| Shared-expert Sparse MoE | `modern-llm-and-moe.md` | `lab11_moe_variants.py` |

补充主题包括 tokenizer/embedding、mask、残差、Pre-LN/RMSNorm、FFN/SwiGLU、
next-token loss、KV Cache、Top-K/Top-P/Beam Search、负载均衡和 expert capacity。

源码阅读顺序：

```text
models/layers.py
  -> models/decoder_encoder_layer.py
  -> models/transformer_models.py
  -> utils/mask_utils.py
  -> scripts/train_decoder.py
  -> scripts/train_encoder_decoder.py
  -> utils/generation_utils.py / translation_utils.py
```

`labs/` 是拆解后的最小实验，`models/` 和 `scripts/` 是组合后的完整训练实现。
