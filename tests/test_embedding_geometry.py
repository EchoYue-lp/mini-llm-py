import numpy as np

from foundations.f02_embedding_geometry import (
    build_vocabulary,
    cooccurrence_matrix,
    cosine_similarity,
    one_hot_lookup_equivalence,
    positive_pmi,
    truncated_svd_embeddings,
)


def test_one_hot_multiplication_matches_embedding_lookup():
    table = np.arange(20, dtype=np.float64).reshape(5, 4)
    multiplied, looked_up = one_hot_lookup_equivalence(3, table)
    assert np.array_equal(multiplied, looked_up)


def test_shared_context_produces_closer_low_dimensional_embeddings():
    sentences = [
        ["apple", "sweet", "fruit"],
        ["banana", "sweet", "fruit"],
        ["apple", "fresh", "fruit"],
        ["banana", "fresh", "fruit"],
        ["quantum", "physics", "equation"],
    ]
    vocabulary = build_vocabulary(sentences)
    matrix = positive_pmi(cooccurrence_matrix(sentences, vocabulary))
    embeddings, reconstruction = truncated_svd_embeddings(matrix, dimensions=2)

    apple = embeddings[vocabulary["apple"]]
    banana = embeddings[vocabulary["banana"]]
    quantum = embeddings[vocabulary["quantum"]]
    assert cosine_similarity(apple, banana) > cosine_similarity(apple, quantum)
    assert reconstruction.shape == matrix.shape
