import math

import numpy as np
import pandas as pd
import torch

from foundations.f00_math_basics import (
    cross_entropy_from_logits,
    finite_difference,
    matrix_multiply,
    stable_softmax as math_softmax,
)
from foundations.f01_numpy_basics import (
    merge_heads as merge_numpy_heads,
    split_heads as split_numpy_heads,
    stable_softmax as numpy_softmax,
)
from foundations.f03_pandas_basics import prepare_dataset, summarize_splits
from foundations.f05_pytorch_autograd import residual_gradient
from foundations.f04_pytorch_tensors import (
    masked_softmax,
    merge_heads as merge_torch_heads,
    split_heads as split_torch_heads,
)
from foundations.f06_pytorch_training import train_classifier


def test_math_foundations_are_numerically_consistent():
    assert matrix_multiply([[1, 2]], [[3], [4]]) == [[11.0]]
    probabilities = math_softmax([1000.0, 999.0, 998.0])
    assert math.isclose(sum(probabilities), 1.0)
    assert math.isclose(
        cross_entropy_from_logits([2.0, 1.0, 0.0], 0),
        -math.log(math_softmax([2.0, 1.0, 0.0])[0]),
    )
    assert math.isclose(
        finite_difference(lambda x: x**3, 2.0),
        12.0,
        rel_tol=1e-5,
    )


def test_numpy_shapes_softmax_and_head_round_trip():
    hidden = np.arange(2 * 3 * 8).reshape(2, 3, 8)
    heads = split_numpy_heads(hidden, num_heads=4)
    assert heads.shape == (2, 4, 3, 2)
    assert np.array_equal(merge_numpy_heads(heads), hidden)

    logits = np.array([[1000.0, 999.0, 998.0]])
    probabilities = numpy_softmax(logits)
    assert np.allclose(probabilities.sum(-1), np.ones(1))


def test_pandas_cleaning_and_split_summary():
    raw = pd.DataFrame(
        {
            "text": ["one two", " ", None, "three tokens here"],
            "split": ["train", "train", "validation", "validation"],
            "label": ["a", None, "b", "c"],
        }
    )
    cleaned = prepare_dataset(raw)
    summary = summarize_splits(cleaned)

    assert cleaned["text"].tolist() == ["one two", "three tokens here"]
    assert cleaned["word_count"].tolist() == [2, 3]
    assert summary["examples"].sum() == 2


def test_pytorch_masks_heads_and_residual_gradient():
    hidden = torch.randn(2, 3, 8)
    heads = split_torch_heads(hidden, num_heads=4)
    assert heads.shape == (2, 4, 3, 2)
    assert torch.equal(merge_torch_heads(heads), hidden)

    scores = torch.randn(1, 2, 3, 3)
    causal = torch.tril(torch.ones(3, 3, dtype=torch.bool)).view(1, 1, 3, 3)
    probabilities = masked_softmax(scores, causal)
    assert torch.allclose(probabilities.sum(-1), torch.ones(1, 2, 3))
    assert torch.count_nonzero(probabilities.triu(1)) == 0

    values = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.allclose(residual_gradient(values), 1 + 2 * values)


def test_tiny_training_loop_reduces_loss():
    _, losses, accuracy = train_classifier(epochs=25)
    assert losses[-1] < losses[0]
    assert accuracy > 0.9
