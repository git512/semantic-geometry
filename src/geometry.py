from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class DomainShape:
    name: str
    terms: list[str]
    embeddings: np.ndarray
    cosine_similarity: np.ndarray
    cosine_distance: np.ndarray
    euclidean_distance: np.ndarray


@dataclass(frozen=True)
class ShapeComparison:
    pearson: float
    spearman: float
    frobenius_difference: float
    procrustes_disparity: float


def build_domain_shape(name: str, terms: list[str], embeddings: np.ndarray) -> DomainShape:
    normalized = normalize(embeddings)
    cos_sim = cosine_similarity(normalized)
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    euclidean_dist = squareform(pdist(normalized, metric="euclidean"))

    return DomainShape(
        name=name,
        terms=terms,
        embeddings=embeddings,
        cosine_similarity=cos_sim,
        cosine_distance=cos_dist,
        euclidean_distance=euclidean_dist,
    )


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices_from(matrix, k=1)
    return matrix[indices]


def normalized_matrix(matrix: np.ndarray) -> np.ndarray:
    values = upper_triangle_values(matrix)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0.0:
        return matrix - mean
    return (matrix - mean) / std


def compare_shapes(shape_a: DomainShape, shape_b: DomainShape) -> ShapeComparison:
    if len(shape_a.terms) != len(shape_b.terms):
        raise ValueError(
            f"Cannot compare {shape_a.name} ({len(shape_a.terms)} terms) with "
            f"{shape_b.name} ({len(shape_b.terms)} terms)."
        )

    pearson = shape_similarity_pearson(shape_a, shape_b)
    dist_a = shape_a.cosine_distance
    dist_b = shape_b.cosine_distance
    spearman = safe_corr(
        spearmanr,
        upper_triangle_values(dist_a),
        upper_triangle_values(dist_b),
    )
    frobenius = float(np.linalg.norm(normalized_matrix(dist_a) - normalized_matrix(dist_b), ord="fro"))
    disparity = procrustes_disparity(shape_a.embeddings, shape_b.embeddings)

    return ShapeComparison(
        pearson=pearson,
        spearman=spearman,
        frobenius_difference=frobenius,
        procrustes_disparity=disparity,
    )


def shape_similarity_pearson(shape_a: DomainShape, shape_b: DomainShape) -> float:
    tri_a = upper_triangle_values(shape_a.cosine_distance)
    tri_b = upper_triangle_values(shape_b.cosine_distance)
    return safe_corr(pearsonr, tri_a, tri_b)


def safe_corr(corr_func, values_a: np.ndarray, values_b: np.ndarray) -> float:
    if np.std(values_a) == 0.0 or np.std(values_b) == 0.0:
        return float("nan")
    result = corr_func(values_a, values_b)
    return float(result.statistic)


def procrustes_disparity(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    try:
        _, _, disparity = procrustes(embeddings_a, embeddings_b)
    except Exception:
        return float("nan")
    return float(disparity)


def all_domain_pairs(domain_names: list[str]) -> list[tuple[str, str]]:
    return list(combinations(domain_names, 2))


def random_similarity_distribution(
    shapes: dict[str, DomainShape],
    actual_pair: tuple[str, str],
    trials: int,
    rng: np.random.Generator,
    intended_pairs: set[tuple[str, str]],
) -> np.ndarray:
    names = list(shapes)
    n_terms = len(shapes[actual_pair[0]].terms)
    candidates = [
        pair
        for pair in all_domain_pairs(names)
        if pair_key(pair) not in intended_pairs
        and len(shapes[pair[0]].terms) == n_terms
        and len(shapes[pair[1]].terms) == n_terms
    ]
    if not candidates:
        return np.array([], dtype=float)

    sample_indices = rng.integers(0, len(candidates), size=trials)
    values = [
        shape_similarity_pearson(shapes[candidates[index][0]], shapes[candidates[index][1]])
        for index in sample_indices
    ]
    return np.asarray(values, dtype=float)


def pair_key(pair: tuple[str, str] | list[str]) -> tuple[str, str]:
    return tuple(sorted((pair[0], pair[1])))
