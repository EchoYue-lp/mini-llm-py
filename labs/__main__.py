"""Run all self-contained labs in the recommended order."""

from labs.lab00_positional_encoding import run_demo as run_position
from labs.lab01_attention_basics import run_demo as run_attention
from labs.lab02_multi_head_attention import run_demo as run_multi_head
from labs.lab03_pre_ln_block import run_demo as run_pre_ln
from labs.lab04_tiny_copy_task import train_copy_task
from labs.lab05_tiny_language_model import train_language_model
from labs.lab06_kv_cache import compare_full_and_cached
from labs.lab07_modern_blocks import run_demo as run_modern_blocks
from labs.lab08_moe_routing import run_demo as run_moe
from labs.lab09_lora_linear import run_demo as run_lora
from labs.lab10_mha_mqa_gqa import run_demo as run_attention_variants
from labs.lab11_moe_variants import run_demo as run_moe_variants


def main() -> None:
    labs = [
        ("Lab 00 Positional encoding", run_position),
        ("Lab 01 Attention basics", run_attention),
        ("Lab 02 Multi-head attention", run_multi_head),
        ("Lab 03 Pre-LN block", run_pre_ln),
        ("Lab 04 Tiny copy task", train_copy_task),
        ("Lab 05 Tiny language model", train_language_model),
        ("Lab 06 KV cache", compare_full_and_cached),
        ("Lab 07 Modern blocks", run_modern_blocks),
        ("Lab 08 MoE routing", run_moe),
        ("Lab 09 LoRA linear", run_lora),
        ("Lab 10 MHA/MQA/GQA", run_attention_variants),
        ("Lab 11 MoE variants", run_moe_variants),
    ]
    for title, lab in labs:
        print(f"\n=== {title} ===")
        lab()


if __name__ == "__main__":
    main()
