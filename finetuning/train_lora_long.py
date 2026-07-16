#!/usr/bin/env python3
"""Long LoRA training with an explicit loop and overfitting analysis.

This file uses the same mathematical building blocks as
``train_lora_short.py`` but keeps the long-run experiment visible here:

- load and tokenize train/validation/test splits;
- freeze the base model and inject LoRA into selected projections;
- compute masked next-token cross entropy;
- run value_and_grad, gradient accumulation, and Adam updates;
- save periodic, best-validation, and final adapters;
- record loss, gradient norm, throughput, memory, and Delta-W norm;
- diagnose whether training continued beyond the best validation checkpoint.

The shared functions are imported instead of copied so the short and long
experiments cannot silently diverge in their LoRA math or data masking rules.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map
from mlx_lm import load

from finetuning.train_lora_short import (
    DEFAULT_TARGETS,
    ExperimentConfig,
    causal_lm_loss,
    evaluate_loss,
    gradient_l2_norm,
    infinite_train_batches,
    inject_lora,
    inspect_example,
    lora_update_l2_norm,
    print_stage,
    read_jsonl,
    save_adapter_config,
    save_trainable_weights,
    tokenize_chat_records,
    trainable_parameter_stats,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--adapter-path",
        default="artifacts/adapters/tool-router-long",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Stop after this many validation checks without improvement; "
            "zero disables early stopping."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and changed variables without training.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    return ExperimentConfig(
        name="long-lora",
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
        steps_per_report=5,
        steps_per_eval=10,
        steps_per_save=20,
        seed=args.seed,
    )


def nearest_train_loss(
    train_history: list[dict[str, Any]], iteration: int
) -> float | None:
    if not train_history:
        return None
    point = min(
        train_history,
        key=lambda item: abs(int(item["iteration"]) - iteration),
    )
    return float(point["train_loss"])


def analyze_learning_curve(summary: dict[str, Any]) -> dict[str, Any]:
    """Compare the best validation point with the final training state."""

    history = summary["history"]
    validation = history["validation"]
    training = history["train"]
    if not validation:
        raise ValueError("No validation points were recorded")

    best = min(validation, key=lambda item: float(item["val_loss"]))
    final = validation[-1]
    best_loss = float(best["val_loss"])
    final_loss = float(final["val_loss"])
    increase = (final_loss - best_loss) / best_loss if best_loss else 0.0
    final_train_loss = nearest_train_loss(training, int(final["iteration"]))

    if increase > 0.10:
        diagnosis = (
            "Validation loss rose more than 10% after its best point. "
            "This is a clear overfitting warning."
        )
    elif increase > 0:
        diagnosis = (
            "Validation loss rose slightly after its best point. "
            "Prefer the best checkpoint unless task metrics say otherwise."
        )
    else:
        diagnosis = "The final validation point is also the best observed point."

    return {
        "best_validation_iteration": int(best["iteration"]),
        "best_validation_loss": best_loss,
        "final_validation_iteration": int(final["iteration"]),
        "final_validation_loss": final_loss,
        "validation_loss_increase_after_best": increase,
        "nearest_final_train_loss": final_train_loss,
        "diagnosis": diagnosis,
        "checkpoint_guidance": (
            "best_adapters.safetensors is selected by validation loss; "
            "adapters.safetensors contains the final training weights."
        ),
    }


def print_analysis(analysis: dict[str, Any]) -> None:
    print_stage(
        "7. Analyze convergence and overfitting",
        "Training loss alone cannot determine which checkpoint generalizes best.",
    )
    print(
        "Best validation: "
        f"iteration={analysis['best_validation_iteration']}, "
        f"loss={analysis['best_validation_loss']:.4f}"
    )
    print(
        "Final validation: "
        f"iteration={analysis['final_validation_iteration']}, "
        f"loss={analysis['final_validation_loss']:.4f}"
    )
    print(
        "Validation increase after best: "
        f"{analysis['validation_loss_increase_after_best']:.2%}"
    )
    if analysis["nearest_final_train_loss"] is not None:
        print(f"Nearest final train loss: {analysis['nearest_final_train_loss']:.4f}")
    print(f"Diagnosis: {analysis['diagnosis']}")
    print(f"Checkpoint guidance: {analysis['checkpoint_guidance']}")


def run_long_experiment(
    config: ExperimentConfig,
    patience: int = 0,
) -> dict[str, Any]:
    """Run the long experiment while keeping every training stage explicit."""

    validate_config(config)
    if patience < 0:
        raise ValueError("patience cannot be negative")
    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    print_stage(
        "1. Load base model and tokenizer",
        "The same pretrained W is used as in the short experiment.",
    )
    model, tokenizer = load(str(config.model_dir))
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    print_stage(
        "2. Build train, validation, and test tensors",
        "Prompt tokens receive zero loss; only assistant JSON is supervised.",
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
        "3. Freeze W and inject LoRA",
        "Long training adapts more blocks, but still updates only low-rank A/B.",
    )
    replaced = inject_lora(model, config)
    total_params, trainable_params = trainable_parameter_stats(model)
    initial_delta_norm = lora_update_l2_norm(model)
    print(f"Targets: {config.target_modules}")
    print(f"Adapted blocks: last {config.num_layers}")
    print(f"Replaced projections: {len(replaced)}")
    print(
        f"rank={config.lora_rank}, alpha={config.lora_alpha}, "
        f"scale={config.lora_scale}"
    )
    print(
        f"Trainable: {trainable_params:,}/{total_params:,} "
        f"({trainable_params / total_params:.3%})"
    )
    print(f"Initial combined Delta-W L2 norm: {initial_delta_norm:.6f}")

    print_stage(
        "4. Configure gradients and optimizer",
        "value_and_grad computes gradients only for unfrozen LoRA parameters.",
    )
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    loss_and_grad = nn.value_and_grad(model, causal_lm_loss)
    save_adapter_config(config, replaced)
    history: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}

    initial_val_loss = evaluate_loss(
        model,
        valid_examples,
        config.batch_size,
        pad_token_id,
    )
    history["validation"].append(
        {"iteration": 0, "val_loss": initial_val_loss}
    )
    best_val_loss = initial_val_loss
    best_iteration = 0
    checks_without_improvement = 0
    save_trainable_weights(model, config.adapter_dir / "best_adapters.safetensors")
    print(f"Validation before training: {initial_val_loss:.4f}")

    print_stage(
        "5. Long training loop",
        "Watch the gap between falling train loss and validation loss over time.",
    )
    batches = infinite_train_batches(
        train_examples,
        config.batch_size,
        pad_token_id,
        config.seed,
    )
    model.train()
    accumulated_gradients = None
    report_losses: list[float] = []
    report_tokens = 0
    trained_tokens = 0
    report_start = time.perf_counter()
    completed_iterations = 0

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

        grad_norm = gradient_l2_norm(gradients)
        if iteration % config.grad_accumulation_steps == 0:
            if config.grad_accumulation_steps > 1:
                accumulated_gradients = tree_map(
                    lambda value: value / config.grad_accumulation_steps,
                    accumulated_gradients,
                )
            optimizer.update(model, accumulated_gradients)
            accumulated_gradients = None

        mx.eval(model.parameters(), optimizer.state, loss, supervised_tokens, grad_norm)
        completed_iterations = iteration
        current_loss = float(loss.item())
        current_tokens = int(supervised_tokens.item())
        report_losses.append(current_loss)
        report_tokens += current_tokens
        trained_tokens += current_tokens

        if iteration % config.steps_per_report == 0 or iteration == config.iterations:
            elapsed = time.perf_counter() - report_start
            point = {
                "iteration": iteration,
                "train_loss": sum(report_losses) / len(report_losses),
                "learning_rate": config.learning_rate,
                "gradient_l2_norm": float(grad_norm.item()),
                "trained_tokens": trained_tokens,
                "tokens_per_second": report_tokens / elapsed,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
            history["train"].append(point)
            print(
                f"iter={iteration:4d} train_loss={point['train_loss']:.4f} "
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
                model,
                valid_examples,
                config.batch_size,
                pad_token_id,
            )
            history["validation"].append(
                {"iteration": iteration, "val_loss": val_loss}
            )
            print(f"iter={iteration:4d} val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iteration
                checks_without_improvement = 0
                save_trainable_weights(
                    model,
                    config.adapter_dir / "best_adapters.safetensors",
                )
            else:
                checks_without_improvement += 1
            model.train()

            if patience and checks_without_improvement >= patience:
                print(
                    f"Early stopping at iteration {iteration}: "
                    f"no validation improvement for {patience} checks."
                )
                break

        if iteration % config.steps_per_save == 0:
            checkpoint = config.adapter_dir / f"{iteration:07d}_adapters.safetensors"
            save_trainable_weights(model, checkpoint)

    print_stage(
        "6. Save final state and evaluate the untouched test split",
        "Final and best adapters are deliberately kept as different files.",
    )
    final_adapter = config.adapter_dir / "adapters.safetensors"
    save_trainable_weights(model, final_adapter)
    final_delta_norm = lora_update_l2_norm(model)
    test_loss = evaluate_loss(model, test_examples, config.batch_size, pad_token_id)
    test_perplexity = math.exp(test_loss)
    print(f"Completed iterations: {completed_iterations}")
    print(f"test_loss={test_loss:.4f}, test_perplexity={test_perplexity:.4f}")
    print(f"best_val_loss={best_val_loss:.4f} at iteration={best_iteration}")
    print(f"Learned combined Delta-W L2 norm: {final_delta_norm:.6f}")

    summary = {
        "experiment": asdict(config),
        "completed_iterations": completed_iterations,
        "early_stopping_patience": patience,
        "lora_formula": "xW + (alpha/rank) * dropout(x) @ A @ B",
        "replaced_modules": replaced,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_params / total_params,
        "initial_delta_weight_l2_norm": initial_delta_norm,
        "final_delta_weight_l2_norm": final_delta_norm,
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
    return summary


def main() -> None:
    args = parse_args()
    config = build_config(args)
    print("Variables changed from the short experiment:")
    print("- iterations: 40 ->", config.iterations)
    print("- adapted transformer blocks: 8 ->", config.num_layers)
    print("- output adapter:", config.adapter_path)
    print("- early-stopping patience:", args.patience or "disabled")

    if args.dry_run:
        print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
        return

    summary = run_long_experiment(config, patience=args.patience)
    analysis = analyze_learning_curve(summary)
    print_analysis(analysis)
    output = config.adapter_dir / "overfitting_analysis.json"
    output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Analysis report: {output}")


if __name__ == "__main__":
    main()
