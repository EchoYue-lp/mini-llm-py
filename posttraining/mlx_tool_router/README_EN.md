# LLM Post-training Lab

English | [中文](README.md)

> Migration status: this project's code has been copied directly into
> `mini-llm-py/posttraining/mlx_tool_router/`. That directory is the sole
> maintained entry point and is no longer synchronized with the original
> `llm-posttrain-lab` repository.

This beginner-oriented repository demonstrates an end-to-end LLM
post-training workflow. The first experiment uses `Qwen/Qwen3-0.6B` and Apple
MLX on a 16 GB M1 Pro Mac to cover model download, base inference, tool-routing
data, validation, short LoRA training, long LoRA training, and evaluation on
one fixed test split.

The repository is not tied to Qwen or LoRA. The same data and evaluation
pipeline can later support other model families, a quantized 4B model, GPU
LoRA, full fine-tuning, DPO, or GRPO.

For the LoRA equation, matrix dimensions, masking, training loop, checkpoints,
and parameter experiments, read [docs/lora-finetuning.md](docs/lora-finetuning.md).

> The current dataset contains only 48 examples and the test split contains
> only 5 examples. It demonstrates the workflow; it is not a production
> benchmark and must not be used to judge real model capability.

## Learning outcomes

1. Download a Hugging Face model into an explicit local directory.
2. Run the base model before training and establish a baseline.
3. Understand intents, slots, tool selection, clarification, and OOS data.
4. Validate schemas, labels, and split leakage before training.
5. Run a short LoRA experiment to verify the training pipeline.
6. Compare the base model and the short adapter.
7. Run a longer experiment and recognize overfitting.
8. Compare base, short, and long runs on the same test split.

## Repository layout

```text
posttraining/mlx_tool_router/
├── scripts/
│   ├── download_model.py       # Download the model
│   └── prepare_demo_data.py    # Generate deterministic demo data
├── inference.py                # Base-model inference
├── validate_data.py            # Data quality validation
├── train_lora_short.py         # 40-step LoRA run
├── train_lora_long.py          # 300-step LoRA run
├── tool_router.py              # Inference with a selected adapter
├── evaluate.py                 # Evaluate one model or adapter
├── compare_models.py           # Compare all three and write Markdown
├── requirements.txt
├── README.md
└── README_EN.md
```

Runtime directories `models/`, `data/`, `adapters/`, `results/`, and `.venv/`
are excluded from Git.

## 0. Environment setup

Verified environment:

- Apple M1 Pro with 16 GB unified memory
- macOS 26.5.2
- Python 3.12.4
- MLX-LM 0.31.3
- Hugging Face Hub 1.23.0

Hardware tiers:

| Scope | Validated configuration | Recommended | Notes |
| --- | --- | --- | --- |
| Data generation, validation, and report reading | M1 Pro with 16 GB unified memory | 16 GB unified memory | These steps do not load the model; no untested lower memory claim is made |
| Qwen3-0.6B inference and LoRA smoke tests | M1 Pro with 16 GB unified memory | 16 GB unified memory | The two-step Metal regression with `batch_size=2` peaked near 1.834 GB; 8 GB hardware was not tested |
| Complete 40/300-step teaching runs | M1 Pro with 16 GB unified memory | 16-24 GB unified memory | Recorded peaks are about 1.7-1.8 GB, but macOS, Python, and file caches also need memory |
| Future quantized 4B models, multiple adapters, or larger batches | Not validated | 24-32 GB unified memory or a CUDA workflow | Training memory cannot be inferred from weight-file size alone |

Reserve at least `5 GB` of disk space; `15-20 GB` is more practical for the
1.4 GB base model, virtual environment, data, checkpoints, adapters, and
evaluation reports. The validated path is Apple Silicon with Metal; Intel Macs,
CPU-only systems, and non-macOS platforms are outside the current test scope.

```bash
cd /Users/ls/MyWork/code/python/mini-llm-py/posttraining/mlx_tool_router
/opt/anaconda3/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Python 3.10 or newer. The macOS system Python 3.9 may resolve an older
MLX-LM release.

```bash
python --version
python -c "import importlib.metadata as m; print(m.version('mlx-lm'))"
python -c "import mlx.core as mx; print(mx.default_device())"
```

The last command should print something similar to `Device(gpu, 0)`.

## 1. Download the model

```bash
python scripts/download_model.py
```

The script uses `huggingface_hub.snapshot_download` to place the official
`Qwen/Qwen3-0.6B` snapshot in:

```text
models/Qwen3-0.6B/
```

Expected files include `config.json`, `model.safetensors`, `tokenizer.json`,
and `tokenizer_config.json`. The verified local directory is about 1.4 GB.
Do not commit `models/` to a normal Git repository.

## 2. Run the model before fine-tuning

```bash
python inference.py
```

Custom prompt:

```bash
python inference.py \
  --prompt "Explain supervised fine-tuning in one sentence." \
  --max-tokens 80
```

The script sets `enable_thinking=False`. A tool router should emit short,
machine-readable JSON rather than a visible reasoning trace.

This proves that the model and MLX environment work. The actual task baseline
is measured later with `evaluate.py` on the fixed test split.

## 3. Prepare and validate the data

```bash
python scripts/prepare_demo_data.py
```

It creates:

```text
data/train.jsonl   38 examples
data/valid.jsonl    5 examples
data/test.jsonl     5 examples
```

The assistant output contains five fields:

```json
{
  "action": "call_tool",
  "intent": "query_logistics",
  "tool": "logistics_query",
  "arguments": {"order_id": "A1024"},
  "missing_arguments": []
}
```

The dataset teaches three decisions:

| Behavior | action | Example |
| --- | --- | --- |
| Call a tool | `call_tool` | Query logistics with an order ID |
| Ask for input | `ask_clarification` | Weather request without a city |
| Do not call | `no_tool` | Chitchat, knowledge, and OOS requests |

Tool schemas:

| Tool | Arguments |
| --- | --- |
| `weather_query` | `city`, `date` |
| `logistics_query` | `order_id` |
| `order_cancel` | `order_id` |
| `refund_query` | `refund_id` |

Validate before training:

```bash
python validate_data.py
```

The validator checks JSONL parsing, chat roles, exact fields, allowed actions,
tool arguments, clarification consistency, `no_tool` consistency, and exact
duplicates across train, validation, and test splits. A real project should
also add semantic near-duplicate detection.

## 4. Short LoRA training

`train_lora_short.py` no longer delegates training to the `mlx_lm.lora` CLI.
It exposes an educational LoRA implementation directly:

1. `EducationalLoRALinear` implements `xW + (alpha/r) * xAB`.
2. `A` is random and `B` is zero, so the initial update is exactly zero.
3. `inject_lora` freezes the base model and replaces selected `q_proj/v_proj`.
4. `tokenize_chat_records` builds a prompt mask for assistant-only loss.
5. `causal_lm_loss` shows token shifting, cross entropy, and padding masks.
6. `value_and_grad`, gradient accumulation, and Adam updates are visible.
7. `delta_weight` and `fuse` show how LoRA merges into a normal linear layer.
8. Final, best, and periodic checkpoints are saved with loss and gradient logs.

Default equation and targets:

```text
output = xW + (alpha / rank) * dropout(x) @ A @ B
rank = 8, alpha = 16, scale = 2
targets = q_proj,v_proj
```

```bash
python train_lora_short.py
```

Inspect the generated MLX-LM command without training:

```bash
python train_lora_short.py --dry-run
```

Override parameters:

```bash
python train_lora_short.py \
  --iters 40 \
  --num-layers 8 \
  --targets q_proj,v_proj \
  --batch-size 1 \
  --grad-accumulation-steps 1 \
  --learning-rate 1e-4
```

The adapter is saved to `adapters/tool-router-short/`.

Observed result:

| Item | Value |
| --- | ---: |
| Steps | 40 |
| Trained layers | Last 8 |
| Trainable parameters | 327,680 / 596,049,920 (0.055%) |
| Adapted projections | 16 (8 blocks x q/v) |
| Peak unified memory | About 1.67 GB |
| Best validation loss | 0.1737 at step 20 |
| Final validation loss | 0.2710 at step 40 |
| Final test loss | 0.2289 |

The short run verifies data loading, backpropagation, adapter saving, and
adapter loading. It is not intended to produce a production model.

```bash
python tool_router.py \
  "Track order A1024" \
  --adapter adapters/tool-router-short
```

## 5. Compare base and short-trained models

```bash
python evaluate.py \
  --label base \
  --output results/base.json

python evaluate.py \
  --adapter adapters/tool-router-short \
  --label short-lora \
  --output results/short-lora.json
```

Metrics:

| Metric | Meaning |
| --- | --- |
| JSON validity | Whether output can be parsed |
| Exact match | All five fields are correct |
| Action accuracy | Call, clarify, or no-tool decision |
| Intent accuracy | Business intent label |
| Tool accuracy | Correct tool or correct non-call |
| Argument accuracy | All names and values are correct |
| Missing-argument accuracy | Exact missing-input list |

Observed teaching result:

| Model | JSON validity | Exact match | Intent accuracy |
| --- | ---: | ---: | ---: |
| Base | 100% | 0% | 0% |
| Short LoRA | 80% | 20% | 60% |

The short run improves task alignment but produces one invalid JSON output.
Intent accuracy alone would hide that failure.

## 6. Long LoRA training

```bash
python train_lora_long.py
```

```bash
python train_lora_long.py --dry-run
python train_lora_long.py --iters 300 --num-layers 16
```

The adapter is saved to `adapters/tool-router-long/`.

Observed result:

| Item | Value |
| --- | ---: |
| Steps | 300 |
| Trained layers | Last 16 |
| Trainable parameters | 655,360 / 596,049,920 (0.110%) |
| Adapted projections | 32 (16 blocks x q/v) |
| Peak unified memory | About 1.77 GB |
| Lowest validation loss | 0.0680 at step 110 |
| Validation loss at step 300 | 0.0797 |
| Training loss near step 300 | 0.0058 |
| Final test loss | 0.0642 |

Training loss approaches zero while validation loss rises after its minimum.
This is an overfitting signal. Real training should select the best validation
checkpoint instead of automatically using the last checkpoint.

## 7. Compare base, short, and long runs

```bash
python compare_models.py
```

The terminal prints a percentage summary. A beginner-friendly case-by-case
report is written to:

```text
results/comparison.md
```

Observed result:

| Model | JSON valid | Exact | Action | Intent | Tool | Args | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 100% | 0% | 40% | 0% | 60% | 80% | 40% |
| short-lora | 80% | 20% | 40% | 60% | 40% | 60% | 60% |
| long-lora | 100% | 80% | 80% | 100% | 80% | 80% | 80% |

`results/comparison.md` shows the input, expected JSON, all three outputs, and
exact correctness for every example. Read the case-level report before trusting
aggregate metrics, because aggregate metrics do not explain failure modes.

The base model does not know the custom labels, the short adapter begins to
learn the format but remains unstable, and the long adapter improves exact
match while still making missing-input errors. More steps cannot replace more
high-quality data.

## 8. Missing stages before a real experiment

1. Expand to 500-2,000 reviewed training examples.
2. Add intent-boundary, negation, colloquial, typo, and ASR-error examples.
3. Add OOS, no-tool, dangerous-tool, missing-input, and multi-intent examples.
4. Build a larger human-reviewed test set with 50-100 examples per intent.
5. Add semantic deduplication and dataset versioning.
6. Select the best validation checkpoint and implement early stopping.
7. Benchmark latency, peak memory, and tokens per second.
8. Use JSON Schema or constrained decoding for production output validity.
9. Evaluate a quantized Qwen3-4B on the same test set.
10. Reuse the same data and metrics for GPU LoRA and full fine-tuning.

## 9. LoRA and full fine-tuning

The repository starts with LoRA because a 16 GB M1 Mac can run the complete
learning loop quickly. LoRA is not the final scope.

When moving to full fine-tuning, keep the data schema, splits, independent test
set, metrics, error analysis, and random seeds unchanged. The training entry,
optimizer state, memory requirement, and checkpoint format will change.

Full fine-tuning a 4B model generally belongs on a large-memory GPU or
multi-GPU machine, not this 16 GB Mac.

## 10. Troubleshooting

### The environment breaks after renaming the repository

Python entrypoints contain absolute paths. Recreate the environment:

```bash
/opt/anaconda3/bin/python3 -m venv --clear .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Loss is low but predictions are still wrong

Common causes include too little data, distribution shift, ambiguous labels,
missing tool schemas in the prompt, and memorization without learning decision
boundaries. Inspect `results/comparison.md` instead of relying on training logs.
