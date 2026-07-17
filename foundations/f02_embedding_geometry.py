"""Foundation F02: from one-hot identities to low-dimensional geometry.

This lesson uses a tiny co-occurrence matrix and truncated SVD to make the
high-dimensional-to-low-dimensional idea observable. Modern language models
learn embeddings end to end instead of running this exact SVD pipeline.

Exercises:
1. Add more fruit and physics sentences and inspect nearest neighbors.
2. Change the embedding dimension from 3 to 1 and compare reconstruction.
3. Replace mean-context mixing with attention from ``labs.lab01``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def one_hot_lookup_equivalence(
    token_id: int,
    embedding_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conceptual one-hot multiplication and direct table lookup."""
    vocab_size = embedding_table.shape[0]
    if not 0 <= token_id < vocab_size:
        raise IndexError("token id is outside the embedding table")
    one_hot = np.zeros(vocab_size, dtype=embedding_table.dtype)
    one_hot[token_id] = 1
    return one_hot @ embedding_table, embedding_table[token_id]


def build_vocabulary(sentences: Sequence[Sequence[str]]) -> dict[str, int]:
    tokens = sorted({token for sentence in sentences for token in sentence})
    return {token: index for index, token in enumerate(tokens)}


def cooccurrence_matrix(
    sentences: Sequence[Sequence[str]],
    vocabulary: dict[str, int],
    window: int = 2,
) -> np.ndarray:
    """Count neighboring tokens in a symmetric context window."""
    matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=np.float64)
    for sentence in sentences:
        ids = [vocabulary[token] for token in sentence]
        for center, center_id in enumerate(ids):
            start = max(0, center - window)
            stop = min(len(ids), center + window + 1)
            for context in range(start, stop):
                if context != center:
                    matrix[center_id, ids[context]] += 1
    return matrix


def positive_pmi(counts: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Convert counts to positive pointwise mutual information."""
    total = counts.sum()
    row_probability = counts.sum(axis=1, keepdims=True) / total
    column_probability = counts.sum(axis=0, keepdims=True) / total
    joint_probability = counts / total
    pmi = np.log(
        (joint_probability + epsilon)
        / (row_probability @ column_probability + epsilon)
    )
    return np.maximum(pmi, 0.0)


def truncated_svd_embeddings(
    matrix: np.ndarray,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low-dimensional rows and their rank-D reconstruction."""
    if not 1 <= dimensions <= min(matrix.shape):
        raise ValueError("dimensions must fit inside the matrix rank bound")
    left, singular_values, right_transpose = np.linalg.svd(
        matrix,
        full_matrices=False,
    )
    root = np.sqrt(singular_values[:dimensions])
    embeddings = left[:, :dimensions] * root
    context = root[:, None] * right_transpose[:dimensions, :]
    reconstruction = embeddings @ context
    return embeddings, reconstruction


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(left @ right / denominator)


def contextualize(
    token_embedding: np.ndarray,
    context_embeddings: np.ndarray,
) -> np.ndarray:
    """A tiny context mixer, not a replacement for Transformer attention."""
    return token_embedding + context_embeddings.mean(axis=0)


def run_demo() -> None:
    table = np.random.default_rng(0).normal(size=(8, 3))
    multiplied, looked_up = one_hot_lookup_equivalence(5, table)
    assert np.array_equal(multiplied, looked_up)
    assert np.linalg.matrix_rank(table) <= table.shape[1]

    sentences = [
        ["apple", "is", "sweet", "fruit"],
        ["banana", "is", "sweet", "fruit"],
        ["apple", "is", "fresh", "fruit"],
        ["banana", "is", "fresh", "fruit"],
        ["quantum", "uses", "physics", "equations"],
        ["electron", "uses", "physics", "equations"],
    ]
    vocabulary = build_vocabulary(sentences)
    counts = cooccurrence_matrix(sentences, vocabulary)
    ppmi = positive_pmi(counts)
    embeddings, reconstruction = truncated_svd_embeddings(ppmi, dimensions=3)

    apple = embeddings[vocabulary["apple"]]
    banana = embeddings[vocabulary["banana"]]
    quantum = embeddings[vocabulary["quantum"]]
    fruit_similarity = cosine_similarity(apple, banana)
    unrelated_similarity = cosine_similarity(apple, quantum)
    assert fruit_similarity > unrelated_similarity

    bank = np.array([0.2, -0.1, 0.5])
    finance_context = np.stack((apple, banana))
    physics_context = np.stack(
        (embeddings[vocabulary["quantum"]], embeddings[vocabulary["physics"]])
    )
    finance_state = contextualize(bank, finance_context)
    physics_state = contextualize(bank, physics_context)
    assert not np.allclose(finance_state, physics_state)

    relative_error = np.linalg.norm(ppmi - reconstruction) / np.linalg.norm(ppmi)
    print("one-hot multiplication equals lookup:", np.array_equal(multiplied, looked_up))
    print("table shape/rank:", table.shape, np.linalg.matrix_rank(table))
    print("co-occurrence shape:", counts.shape)
    print("low-dimensional embedding shape:", embeddings.shape)
    print("rank-3 reconstruction relative error:", round(float(relative_error), 4))
    print("cosine(apple, banana):", round(fruit_similarity, 4))
    print("cosine(apple, quantum):", round(unrelated_similarity, 4))
    print("same static token, different contextual states:")
    print(" finance:", finance_state.round(4).tolist())
    print(" physics:", physics_state.round(4).tolist())


if __name__ == "__main__":
    run_demo()
