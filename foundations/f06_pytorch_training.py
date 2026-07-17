"""Foundation F06: a complete, deterministic PyTorch training loop.

Exercises after running this module:
1. Replace SGD with AdamW and compare the loss curve.
2. Remove ``zero_grad`` and explain the resulting optimization behavior.
3. Add Dropout, then compare repeated outputs in train and eval modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TinyClassifier(nn.Module):
    """A linear two-class model for a linearly separable toy dataset."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def make_dataset(num_samples: int = 256, seed: int = 7) -> TensorDataset:
    """Create deterministic 2D points separated by a straight boundary."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, 2, generator=generator)
    labels = (features[:, 0] - 0.75 * features[:, 1] > 0).long()
    return TensorDataset(features, labels)


def evaluate_accuracy(model: nn.Module, dataset: TensorDataset) -> float:
    """Evaluate without dropout or gradient graph construction."""
    model.eval()
    features, labels = dataset.tensors
    with torch.no_grad():
        predictions = model(features).argmax(dim=-1)
    return (predictions == labels).float().mean().item()


def train_classifier(
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 0.2,
) -> tuple[TinyClassifier, list[float], float]:
    """Train a tiny classifier and return model, epoch losses, and accuracy."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    torch.manual_seed(0)
    dataset = make_dataset()
    loader_generator = torch.Generator().manual_seed(11)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    model = TinyClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for features, labels in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            total_examples += features.size(0)
        losses.append(total_loss / total_examples)

    accuracy = evaluate_accuracy(model, dataset)
    return model, losses, accuracy


def run_demo() -> None:
    model, losses, accuracy = train_classifier()
    assert losses[-1] < losses[0]
    assert accuracy > 0.9

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("trainable parameters:", parameter_count)
    print("first loss:", round(losses[0], 4))
    print("final loss:", round(losses[-1], 4))
    print("training-set accuracy:", round(accuracy, 4))
    print("loop: zero_grad -> forward -> loss -> backward -> step")


if __name__ == "__main__":
    run_demo()
