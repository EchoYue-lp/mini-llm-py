#!/usr/bin/env python3
"""A from-first-principles LoRA training demo implemented with MLX.

Only model/tokenizer loading comes from MLX-LM. This file implements the LoRA
layer, target-module injection, chat tokenization, prompt masking, padded batch
creation, causal language-model loss, gradient accumulation, Adam updates,
validation, checkpointing, and training-history output.

The core LoRA equation is:

    y = xW + (alpha / rank) * dropout(x) @ A @ B

``W`` is frozen. ``A`` and ``B`` are the only trainable matrices. ``B`` starts
at zero, so the initial LoRA update is exactly zero and the wrapped model starts
with the same output as the base model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm import load
from mlx_lm.utils import get_total_parameters, save_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TARGETS = ("q_proj", "v_proj")


@dataclass
class ExperimentConfig:
    """Every value needed to reproduce one LoRA experiment."""

    name: str = "short-lora"
    model_path: str = "models/Qwen3-0.6B"
    data_path: str = "data"
    adapter_path: str = "adapters/tool-router-short"
    iterations: int = 40
    num_layers: int = 8
    target_modules: tuple[str, ...] = DEFAULT_TARGETS
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    max_seq_length: int = 512
    steps_per_report: int = 5
    steps_per_eval: int = 10
    steps_per_save: int = 20
    seed: int = 0

    @property
    def model_dir(self) -> Path:
        return PROJECT_ROOT / self.model_path

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / self.data_path

    @property
    def adapter_dir(self) -> Path:
        return PROJECT_ROOT / self.adapter_path

    @property
    def lora_scale(self) -> float:
        return self.lora_alpha / self.lora_rank

    @property
    def lora_parameters(self) -> dict[str, float | int | list[str]]:
        # ``keys`` is filled with full module paths when adapter metadata is saved.
        return {
            "rank": self.lora_rank,
            "scale": self.lora_scale,
            "dropout": self.lora_dropout,
        }


@dataclass
class TokenizedExample:
    token_ids: list[int]
    prompt_length: int


class EducationalLoRALinear(nn.Module):
    """Frozen linear projection plus a trainable low-rank residual branch."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        output_dims, input_dims = linear.weight.shape

        # Keep the original pretrained matrix W, but never update it.
        self.linear = linear
        self.linear.freeze()

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(p=dropout)

        # A uses a small random initialization. B starts at zero, therefore
        # A @ B == 0 at step 0 and the wrapped layer initially equals x @ W.
        bound = 1.0 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(input_dims, rank),
        )
        self.lora_b = mx.zeros((rank, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        base_output = self.linear(x)
        low_rank_update = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return base_output + (self.scale * low_rank_update).astype(x.dtype)

    def delta_weight(self) -> mx.array:
        """Return Delta-W in the same [out, in] layout as Linear.weight."""

        # Forward uses x @ A @ B, while Linear stores W as [out, in].
        return (self.scale * self.lora_b.T) @ self.lora_a.T

    def fuse(self) -> nn.Linear:
        """Merge W + Delta-W into a normal Linear layer for deployment."""

        output_dims, input_dims = self.linear.weight.shape
        has_bias = "bias" in self.linear
        fused = nn.Linear(input_dims, output_dims, bias=has_bias)
        fused.weight = self.linear.weight + self.delta_weight().astype(
            self.linear.weight.dtype
        )
        if has_bias:
            fused.bias = self.linear.bias
        return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter-path", default="adapters/tool-router-short")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without loading the model.",
    )
    return parser.parse_args()


def print_stage(title: str, explanation: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
    print(explanation)


def validate_config(config: ExperimentConfig) -> None:
    if not config.model_dir.exists():
        raise FileNotFoundError(
            f"Missing {config.model_dir}; run scripts/download_model.py first."
        )
    for split in ("train", "valid", "test"):
        path = config.data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; run scripts/prepare_demo_data.py first."
            )
    if config.lora_rank <= 0:
        raise ValueError("rank must be positive")
    if config.lora_alpha <= 0:
        raise ValueError("alpha must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.batch_size <= 0 or config.grad_accumulation_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    if config.iterations % config.grad_accumulation_steps != 0:
        raise ValueError(
            "iterations must be divisible by grad_accumulation_steps so every "
            "microbatch contributes to an optimizer update"
        )
    if not config.target_modules:
        raise ValueError("at least one LoRA target module is required")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def tokenize_chat_records(
    records: Sequence[dict[str, Any]],
    tokenizer,
    max_seq_length: int,
) -> list[TokenizedExample]:
    """Convert Chat records to tokens and record where answer loss begins."""

    examples = []
    for record in records:
        messages = record["messages"]
        all_tokens = tokenizer.apply_chat_template(
            messages,
            return_dict=False,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=False,
        )
        all_tokens = list(all_tokens[:max_seq_length])
        prompt_length = min(len(prompt_tokens), len(all_tokens))
        if prompt_length >= len(all_tokens):
            raise ValueError(
                "The assistant answer was fully truncated. Increase max_seq_length."
            )
        examples.append(TokenizedExample(all_tokens, prompt_length))
    return examples


def make_batch(
    examples: Sequence[TokenizedExample],
    indices: Sequence[int],
    pad_token_id: int,
) -> tuple[mx.array, mx.array]:
    """Pad variable-length examples and return [prompt_start, real_length]."""

    selected = [examples[index] for index in indices]
    max_length = max(len(example.token_ids) for example in selected)
    token_rows = []
    length_rows = []
    for example in selected:
        padding = [pad_token_id] * (max_length - len(example.token_ids))
        token_rows.append(example.token_ids + padding)
        length_rows.append([example.prompt_length, len(example.token_ids)])
    return mx.array(token_rows, dtype=mx.int32), mx.array(length_rows, dtype=mx.int32)


def infinite_train_batches(
    examples: Sequence[TokenizedExample],
    batch_size: int,
    pad_token_id: int,
    seed: int,
) -> Iterator[tuple[mx.array, mx.array]]:
    """Shuffle once per epoch and keep yielding minibatches."""

    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(examples))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if len(indices) == 0:
                continue
            yield make_batch(examples, indices, pad_token_id)


def finite_eval_batches(
    examples: Sequence[TokenizedExample],
    batch_size: int,
    pad_token_id: int,
) -> Iterator[tuple[mx.array, mx.array]]:
    for start in range(0, len(examples), batch_size):
        indices = range(start, min(start + batch_size, len(examples)))
        yield make_batch(examples, list(indices), pad_token_id)


def causal_lm_loss(model: nn.Module, batch: mx.array, lengths: mx.array):
    """Next-token cross entropy, masked to assistant-answer tokens only."""

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)

    # ``positions`` are indices in the original unshifted sequence.  For a
    # real sequence of length L, valid targets are at positions 1..L-1.  The
    # strict upper bound prevents the first padding token at position L from
    # contributing loss in a variable-length batch.
    positions = mx.arange(1, targets.shape[1] + 1)
    after_prompt = positions >= lengths[:, 0:1]
    before_padding = positions < lengths[:, 1:]
    mask = mx.logical_and(after_prompt, before_padding)

    per_token_loss = nn.losses.cross_entropy(logits, targets) * mask
    supervised_tokens = mask.sum()
    average_loss = per_token_loss.astype(mx.float32).sum() / supervised_tokens
    return average_loss, supervised_tokens


def inject_lora(model: nn.Module, config: ExperimentConfig) -> list[str]:
    """Freeze the model and replace selected projections in the last N blocks."""

    if config.num_layers > len(model.layers):
        raise ValueError(
            f"Requested {config.num_layers} blocks; model has {len(model.layers)}"
        )
    model.freeze()
    replaced = []
    first_block = len(model.layers) - config.num_layers

    for block_index in range(first_block, len(model.layers)):
        block = model.layers[block_index]
        updates = []
        for module_path, module in block.named_modules():
            leaf_name = module_path.rsplit(".", 1)[-1]
            if leaf_name not in config.target_modules:
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"Target {module_path} is {type(module).__name__}, expected Linear"
                )
            updates.append(
                (
                    module_path,
                    EducationalLoRALinear(
                        module,
                        rank=config.lora_rank,
                        alpha=config.lora_alpha,
                        dropout=config.lora_dropout,
                    ),
                )
            )
            replaced.append(f"layers.{block_index}.{module_path}")
        if updates:
            block.update_modules(tree_unflatten(updates))

    if not replaced:
        raise ValueError(
            f"No modules matched targets {config.target_modules}. "
            "Inspect model.layers[0].named_modules()."
        )
    return replaced


def trainable_parameter_stats(model: nn.Module) -> tuple[int, int]:
    total = get_total_parameters(model)
    trainable = sum(
        value.size for _, value in tree_flatten(model.trainable_parameters())
    )
    return total, trainable


def lora_update_l2_norm(model: nn.Module) -> float:
    """Measure the combined size of all learned Delta-W matrices."""

    squared_norms = []
    for _, module in model.named_modules():
        if isinstance(module, EducationalLoRALinear):
            delta = module.delta_weight().astype(mx.float32)
            squared_norms.append(mx.sum(delta**2))
    if not squared_norms:
        return 0.0
    norm = mx.sqrt(sum(squared_norms))
    mx.eval(norm)
    return float(norm.item())


def gradient_l2_norm(gradients) -> mx.array:
    squares = [mx.sum(value.astype(mx.float32) ** 2) for _, value in tree_flatten(gradients)]
    return mx.sqrt(sum(squares))


def evaluate_loss(
    model: nn.Module,
    examples: Sequence[TokenizedExample],
    batch_size: int,
    pad_token_id: int,
) -> float:
    """Compute token-weighted mean loss without changing any parameters."""

    model.eval()
    total_loss = mx.array(0.0)
    total_tokens = mx.array(0)
    for batch, lengths in finite_eval_batches(examples, batch_size, pad_token_id):
        loss, tokens = causal_lm_loss(model, batch, lengths)
        total_loss += loss * tokens
        total_tokens += tokens
        mx.eval(total_loss, total_tokens)
    return float((total_loss / total_tokens).item())


def save_trainable_weights(model: nn.Module, path: Path) -> None:
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), weights)


def save_adapter_config(
    config: ExperimentConfig,
    replaced_modules: Sequence[str],
) -> None:
    """Write metadata expected by MLX-LM when loading the saved adapter."""

    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    # MLX-LM applies these paths inside every selected transformer block when
    # reconstructing its compatible LoRALinear modules at inference time.
    relative_keys = sorted(
        {
            module_name.split(".", 2)[2]
            for module_name in replaced_modules
        }
    )
    lora_parameters = dict(config.lora_parameters)
    lora_parameters["keys"] = relative_keys
    metadata = {
        "model": config.model_path,
        "data": config.data_path,
        "adapter_path": config.adapter_path,
        "fine_tune_type": "lora",
        "num_layers": config.num_layers,
        "lora_parameters": lora_parameters,
        "mask_prompt": True,
        "batch_size": config.batch_size,
        "grad_accumulation_steps": config.grad_accumulation_steps,
        "iters": config.iterations,
        "learning_rate": config.learning_rate,
        "max_seq_length": config.max_seq_length,
        "seed": config.seed,
    }
    save_config(metadata, config.adapter_dir / "adapter_config.json")


def inspect_example(example: TokenizedExample, record: dict[str, Any]) -> None:
    target = json.loads(record["messages"][-1]["content"])
    print(f"Roles: {[message['role'] for message in record['messages']]}")
    print(f"Target JSON: {json.dumps(target, ensure_ascii=False)}")
    print(f"Total tokens: {len(example.token_ids)}")
    print(f"Prompt tokens with zero loss: {example.prompt_length}")
    print(f"Assistant tokens with loss: {len(example.token_ids) - example.prompt_length}")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    validate_config(config)
    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    print_stage(
        "1. Load base model and tokenizer",
        "No model parameter has been changed yet.",
    )
    model, tokenizer = load(str(config.model_dir))
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    print_stage(
        "2. Tokenize Chat data and build prompt masks",
        "Only assistant JSON tokens will contribute to supervised loss.",
    )
    raw_train = read_jsonl(config.data_dir / "train.jsonl")
    raw_valid = read_jsonl(config.data_dir / "valid.jsonl")
    raw_test = read_jsonl(config.data_dir / "test.jsonl")
    train_examples = tokenize_chat_records(raw_train, tokenizer, config.max_seq_length)
    valid_examples = tokenize_chat_records(raw_valid, tokenizer, config.max_seq_length)
    test_examples = tokenize_chat_records(raw_test, tokenizer, config.max_seq_length)
    print(
        f"Examples: train={len(train_examples)}, valid={len(valid_examples)}, "
        f"test={len(test_examples)}"
    )
    inspect_example(train_examples[0], raw_train[0])

    print_stage(
        "3. Freeze W and inject LoRA A/B matrices",
        "A is random, B is zero, scale=alpha/rank; only A and B are trainable.",
    )
    replaced = inject_lora(model, config)
    total_params, trainable_params = trainable_parameter_stats(model)
    print(f"Targets: {config.target_modules}")
    print(f"Adapted transformer blocks: last {config.num_layers}")
    print(f"Replaced linear projections: {len(replaced)}")
    for name in replaced[:8]:
        print(f"  - {name}")
    if len(replaced) > 8:
        print(f"  ... and {len(replaced) - 8} more")
    print(f"rank={config.lora_rank}, alpha={config.lora_alpha}, scale={config.lora_scale}")
    print(f"Trainable: {trainable_params:,}/{total_params:,} ({trainable_params / total_params:.3%})")
    initial_update_norm = lora_update_l2_norm(model)
    print(f"Initial combined Delta-W L2 norm: {initial_update_norm:.6f}")
    print("Initial norm is zero because every LoRA B matrix starts at zero.")

    print_stage(
        "4. Configure loss, gradients, and Adam",
        "value_and_grad differentiates causal_lm_loss only through trainable A/B matrices.",
    )
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    loss_and_grad = nn.value_and_grad(model, causal_lm_loss)
    save_adapter_config(config, replaced)
    history: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    best_val_loss = math.inf
    best_iteration = 0

    initial_val = evaluate_loss(
        model, valid_examples, config.batch_size, pad_token_id
    )
    history["validation"].append({"iteration": 0, "val_loss": initial_val})
    save_trainable_weights(model, config.adapter_dir / "best_adapters.safetensors")
    best_val_loss = initial_val
    print(f"Validation before training: {initial_val:.4f}")

    print_stage(
        "5. Training loop",
        "Each update performs forward -> masked CE -> backward -> Adam -> checkpoint.",
    )
    batches = infinite_train_batches(
        train_examples,
        config.batch_size,
        pad_token_id,
        config.seed,
    )
    model.train()
    accumulated_gradients = None
    report_losses = []
    report_tokens = 0
    trained_tokens = 0
    report_start = time.perf_counter()

    for iteration in range(1, config.iterations + 1):
        batch, lengths = next(batches)
        (loss, supervised_tokens), gradients = loss_and_grad(model, batch, lengths)

        if accumulated_gradients is None:
            accumulated_gradients = gradients
        else:
            accumulated_gradients = tree_map(
                lambda previous, current: previous + current,
                accumulated_gradients,
                gradients,
            )

        should_update = iteration % config.grad_accumulation_steps == 0
        grad_norm = gradient_l2_norm(gradients)
        if should_update:
            if config.grad_accumulation_steps > 1:
                accumulated_gradients = tree_map(
                    lambda value: value / config.grad_accumulation_steps,
                    accumulated_gradients,
                )
            optimizer.update(model, accumulated_gradients)
            accumulated_gradients = None

        mx.eval(model.parameters(), optimizer.state, loss, supervised_tokens, grad_norm)
        report_losses.append(float(loss.item()))
        tokens = int(supervised_tokens.item())
        report_tokens += tokens
        trained_tokens += tokens

        if iteration % config.steps_per_report == 0 or iteration == config.iterations:
            elapsed = time.perf_counter() - report_start
            train_loss = sum(report_losses) / len(report_losses)
            point = {
                "iteration": iteration,
                "train_loss": train_loss,
                "learning_rate": config.learning_rate,
                "gradient_l2_norm": float(grad_norm.item()),
                "trained_tokens": trained_tokens,
                "tokens_per_second": report_tokens / elapsed,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
            history["train"].append(point)
            print(
                f"iter={iteration:4d} train_loss={train_loss:.4f} "
                f"grad_norm={point['gradient_l2_norm']:.4f} "
                f"tokens/s={point['tokens_per_second']:.1f} "
                f"peak_mem={point['peak_memory_gb']:.3f}GB"
            )
            report_losses = []
            report_tokens = 0
            report_start = time.perf_counter()

        should_evaluate = (
            iteration % config.steps_per_eval == 0
            or iteration == config.iterations
        )
        if should_evaluate:
            val_loss = evaluate_loss(
                model, valid_examples, config.batch_size, pad_token_id
            )
            history["validation"].append(
                {"iteration": iteration, "val_loss": val_loss}
            )
            print(f"iter={iteration:4d} val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iteration
                save_trainable_weights(
                    model, config.adapter_dir / "best_adapters.safetensors"
                )
            model.train()

        if iteration % config.steps_per_save == 0:
            checkpoint = config.adapter_dir / f"{iteration:07d}_adapters.safetensors"
            save_trainable_weights(model, checkpoint)

    final_adapter = config.adapter_dir / "adapters.safetensors"
    save_trainable_weights(model, final_adapter)
    final_update_norm = lora_update_l2_norm(model)

    print_stage(
        "6. Final test evaluation",
        "The test split never contributed gradients or checkpoint selection.",
    )
    test_loss = evaluate_loss(model, test_examples, config.batch_size, pad_token_id)
    test_perplexity = math.exp(test_loss)
    print(f"test_loss={test_loss:.4f}, test_perplexity={test_perplexity:.4f}")
    print(f"best_val_loss={best_val_loss:.4f} at iteration={best_iteration}")
    print(f"Learned combined Delta-W L2 norm: {final_update_norm:.6f}")

    summary = {
        "experiment": asdict(config),
        "lora_formula": "xW + (alpha/rank) * dropout(x) @ A @ B",
        "replaced_modules": replaced,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_params / total_params,
        "initial_delta_weight_l2_norm": initial_update_norm,
        "final_delta_weight_l2_norm": final_update_norm,
        "best_validation_loss": best_val_loss,
        "best_validation_iteration": best_iteration,
        "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        "history": history,
    }
    (config.adapter_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (config.adapter_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Final adapter: {final_adapter}")
    print(f"Best adapter: {config.adapter_dir / 'best_adapters.safetensors'}")
    print(f"Training history: {config.adapter_dir / 'training_history.json'}")
    return summary


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    return ExperimentConfig(
        iterations=args.iters,
        num_layers=args.num_layers,
        target_modules=targets,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        adapter_path=args.adapter_path,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.dry_run:
        print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
        print(f"Derived LoRA scale alpha/rank: {config.lora_scale}")
        return
    run_experiment(config)


if __name__ == "__main__":
    main()
