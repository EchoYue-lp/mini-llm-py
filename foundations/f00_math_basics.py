"""Foundation F00: math used directly by tensor and Transformer code.

Exercises after running this module:
1. Change the vectors in ``run_demo`` and calculate the dot product by hand.
2. Increase one logit by 10 and observe the softmax concentration.
3. Replace ``x**2`` in the finite-difference example with ``x**3``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of two equally sized vectors."""
    if len(left) != len(right):
        raise ValueError("dot product requires vectors with the same length")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def matrix_multiply(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Multiply two small rectangular matrices represented by Python lists."""
    if not left or not right or not left[0] or not right[0]:
        raise ValueError("matrices must be non-empty")

    left_width = len(left[0])
    right_width = len(right[0])
    if any(len(row) != left_width for row in left):
        raise ValueError("left matrix must be rectangular")
    if any(len(row) != right_width for row in right):
        raise ValueError("right matrix must be rectangular")
    if left_width != len(right):
        raise ValueError("inner matrix dimensions must match")

    right_columns = list(zip(*right))
    return [[dot(row, column) for column in right_columns] for row in left]


def population_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def population_variance(values: Sequence[float]) -> float:
    """Return population variance, matching the default used in many norms."""
    mean = population_mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def stable_softmax(logits: Sequence[float]) -> list[float]:
    """Compute softmax after subtracting the largest logit."""
    if not logits:
        raise ValueError("softmax requires at least one logit")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def cross_entropy_from_logits(logits: Sequence[float], target: int) -> float:
    """Return one-label cross entropy using a stable log-sum-exp formula."""
    if not 0 <= target < len(logits):
        raise IndexError("target index is outside the logits vector")
    maximum = max(logits)
    log_sum_exp = maximum + math.log(
        sum(math.exp(value - maximum) for value in logits)
    )
    return log_sum_exp - logits[target]


def scaled_dot_product(query: Sequence[float], key: Sequence[float]) -> float:
    """Scale a dot product by the square root of its feature dimension."""
    if not query:
        raise ValueError("query and key must be non-empty")
    return dot(query, key) / math.sqrt(len(query))


def finite_difference(
    function: Callable[[float], float],
    value: float,
    epsilon: float = 1e-5,
) -> float:
    """Approximate a scalar derivative with a centered finite difference."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (function(value + epsilon) - function(value - epsilon)) / (2 * epsilon)


def run_demo() -> None:
    vector_a = [1.0, 2.0, 3.0]
    vector_b = [4.0, 5.0, 6.0]
    product = dot(vector_a, vector_b)
    assert product == 32.0

    matrix = matrix_multiply(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    assert matrix == [[4.0, 5.0], [10.0, 11.0]]

    probabilities = stable_softmax([1000.0, 999.0, 998.0])
    assert math.isclose(sum(probabilities), 1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]

    loss = cross_entropy_from_logits([2.0, 1.0, 0.0], target=0)
    derivative = finite_difference(lambda x: x**2, value=3.0)
    assert math.isclose(loss, -math.log(stable_softmax([2.0, 1.0, 0.0])[0]))
    assert math.isclose(derivative, 6.0, rel_tol=1e-5)

    variance = population_variance([1.0, 2.0, 3.0])
    score = scaled_dot_product(vector_a, vector_b)

    print("dot product:", product)
    print("matrix product shape: [2,3] @ [3,2] -> [2,2]", matrix)
    print("stable softmax:", [round(value, 4) for value in probabilities])
    print("cross entropy for target 0:", round(loss, 4))
    print("population variance:", round(variance, 4))
    print("scaled dot product:", round(score, 4))
    print("finite-difference derivative of x^2 at x=3:", round(derivative, 4))


if __name__ == "__main__":
    run_demo()
