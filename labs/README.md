# Transformer 小实验

这些文件按“一个实验只解释一个核心问题”的原则组织。先运行前 3 个纯结构实验，
再运行两个最小训练任务，最后学习现代 LLM、MoE 和 LoRA。

硬件要求很低：结构实验使用 2 核 CPU 和 4GB RAM 即可；`lab04-05` 微型训练推荐
4 核以上 CPU 和 8GB RAM，CUDA/MPS 只是可选加速。完整训练的硬件分档见根目录
`README.md`。

| 顺序 | 文件 | 观察重点 |
| --- | --- | --- |
| 0 | `lab00_positional_encoding.py` | Sinusoidal、learned position、RoPE 的区别 |
| 1 | `lab01_attention_basics.py` | `QK^T / sqrt(d)`、softmax、因果 mask |
| 2 | `lab02_multi_head_attention.py` | `d_model` 如何拆成多个 head 再合并 |
| 3 | `lab03_pre_ln_block.py` | LayerNorm、残差路径和梯度 |
| 4 | `lab04_tiny_copy_task.py` | Encoder-Decoder、teacher forcing、shifted labels |
| 5 | `lab05_tiny_language_model.py` | Decoder-Only 和 next-token prediction |
| 6 | `lab06_kv_cache.py` | 全量因果注意力与逐 token cache 等价性 |
| 7 | `lab07_modern_blocks.py` | RMSNorm、RoPE、SwiGLU |
| 8 | `lab08_moe_routing.py` | Top-k router、expert dispatch、负载均衡 loss |
| 9 | `lab09_lora_linear.py` | 低秩增量、零初始化、adapter 融合 |
| 10 | `lab10_mha_mqa_gqa.py` | Query head 与 KV head 分离、cache 大小 |
| 11 | `lab11_moe_variants.py` | Dense、Sparse、shared-expert Sparse MoE |

从项目根目录运行：

```bash
python labs/lab00_positional_encoding.py
python labs/lab01_attention_basics.py
python labs/lab02_multi_head_attention.py
python labs/lab03_pre_ln_block.py
python labs/lab04_tiny_copy_task.py --steps 400
python labs/lab05_tiny_language_model.py --steps 100
python labs/lab06_kv_cache.py
python labs/lab07_modern_blocks.py
python labs/lab08_moe_routing.py
python labs/lab09_lora_linear.py
python labs/lab10_mha_mqa_gqa.py
python labs/lab11_moe_variants.py
```

建议每次只改一个变量，例如 head 数、hidden size、是否使用 mask、MoE 的 `top_k`
或 LoRA rank，然后记录 shape、loss、参数量和输出变化。
