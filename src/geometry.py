from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    description: str = ""
    role: str = "unknown"

    def embedding_text(self, mode: str) -> str:
        if mode == "label_only" or not self.description:
            return self.label
        if mode == "label_plus_description":
            return f"{self.label}: {self.description}"
        raise ValueError(f"Unknown embed-text mode: {mode}")


@dataclass(frozen=True)
class DomainShape:
    name: str
    concepts: list[Concept]
    embeddings: np.ndarray
    cosine_similarity: np.ndarray
    cosine_distance: np.ndarray
    euclidean_distance: np.ndarray

    @property
    def terms(self) -> list[str]:
        return [concept.id for concept in self.concepts]

    @property
    def labels(self) -> list[str]:
        return [concept.label for concept in self.concepts]


@dataclass(frozen=True)
class ShapeComparison:
    pearson: float
    spearman: float
    frobenius_difference: float
    procrustes_disparity: float


@dataclass(frozen=True)
class AlignmentResult:
    comparison: ShapeComparison
    permutation: tuple[int, ...]
    method_used: str


def build_domain_shape(name: str, concepts: list[Concept], embeddings: np.ndarray) -> DomainShape:
    normalized = normalize(embeddings)
    cos_sim = cosine_similarity(normalized)
    cos_dist = 1.0 - cos_sim
    np.fill_diagonal(cos_dist, 0.0)
    euclidean_dist = squareform(pdist(normalized, metric="euclidean"))

    return DomainShape(
        name=name,
        concepts=concepts,
        embeddings=embeddings,
        cosine_similarity=cos_sim,
        cosine_distance=cos_dist,
        euclidean_distance=euclidean_dist,
    )


def permute_shape(shape: DomainShape, permutation: tuple[int, ...]) -> DomainShape:
    indices = np.asarray(permutation, dtype=int)
    return DomainShape(
        name=shape.name,
        concepts=[shape.concepts[index] for index in permutation],
        embeddings=shape.embeddings[indices],
        cosine_similarity=shape.cosine_similarity[np.ix_(indices, indices)],
        cosine_distance=shape.cosine_distance[np.ix_(indices, indices)],
        euclidean_distance=shape.euclidean_distance[np.ix_(indices, indices)],
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


def compare_with_permutation(
    shape_a: DomainShape,
    shape_b: DomainShape,
    permutation: tuple[int, ...],
) -> ShapeComparison:
    return compare_shapes(shape_a, permute_shape(shape_b, permutation))


def find_optimal_alignment(
    shape_a: DomainShape,
    shape_b: DomainShape,
    rng: np.random.Generator,
    max_bruteforce_n: int = 8,
    restarts: int = 200,
    steps: int = 1000,
    progress_callback=None,
) -> AlignmentResult:
    n_terms = len(shape_a.terms)
    if n_terms != len(shape_b.terms):
        raise ValueError(
            f"Cannot align {shape_a.name} ({n_terms} terms) with "
            f"{shape_b.name} ({len(shape_b.terms)} terms)."
        )

    if n_terms <= max_bruteforce_n:
        return brute_force_alignment(shape_a, shape_b, progress_callback)
    return heuristic_alignment(shape_a, shape_b, rng, restarts, steps, progress_callback)


def brute_force_alignment(shape_a: DomainShape, shape_b: DomainShape, progress_callback=None) -> AlignmentResult:
    best_score = -np.inf
    best_permutation = tuple(range(len(shape_a.terms)))
    checked = 0

    for permutation in permutations(range(len(shape_a.terms))):
        score = shape_similarity_for_permutation(shape_a, shape_b, permutation)
        checked += 1
        if score > best_score:
            best_score = score
            best_permutation = tuple(permutation)
        if progress_callback is not None:
            progress_callback(checked, float(best_score))

    return AlignmentResult(
        comparison=compare_with_permutation(shape_a, shape_b, best_permutation),
        permutation=best_permutation,
        method_used="brute_force",
    )


def heuristic_alignment(
    shape_a: DomainShape,
    shape_b: DomainShape,
    rng: np.random.Generator,
    restarts: int,
    steps: int,
    progress_callback=None,
) -> AlignmentResult:
    n_terms = len(shape_a.terms)
    best_score = -np.inf
    best_permutation = tuple(range(n_terms))
    checked = 0

    for restart in range(restarts):
        current = rng.permutation(n_terms)
        current_score = shape_similarity_for_permutation(shape_a, shape_b, current)

        for _ in range(steps):
            i, j = rng.choice(n_terms, size=2, replace=False)
            candidate = current.copy()
            candidate[i], candidate[j] = candidate[j], candidate[i]
            candidate_score = shape_similarity_for_permutation(shape_a, shape_b, candidate)
            checked += 1

            if candidate_score >= current_score:
                current = candidate
                current_score = candidate_score

            if current_score > best_score:
                best_score = current_score
                best_permutation = tuple(int(index) for index in current)

            if progress_callback is not None:
                progress_callback(checked, float(best_score), restart + 1)

    return AlignmentResult(
        comparison=compare_with_permutation(shape_a, shape_b, best_permutation),
        permutation=best_permutation,
        method_used="heuristic",
    )


def shape_similarity_pearson(shape_a: DomainShape, shape_b: DomainShape) -> float:
    tri_a = upper_triangle_values(shape_a.cosine_distance)
    tri_b = upper_triangle_values(shape_b.cosine_distance)
    return pearson_arrays(tri_a, tri_b)


def shape_similarity_for_permutation(
    shape_a: DomainShape,
    shape_b: DomainShape,
    permutation: tuple[int, ...] | np.ndarray,
) -> float:
    indices_a, indices_b = np.triu_indices(len(shape_a.terms), k=1)
    permutation_array = np.asarray(permutation, dtype=int)
    values_a = shape_a.cosine_distance[indices_a, indices_b]
    values_b = shape_b.cosine_distance[permutation_array[indices_a], permutation_array[indices_b]]
    return pearson_arrays(values_a, values_b)


def pearson_from_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    return pearson_arrays(upper_triangle_values(matrix_a), upper_triangle_values(matrix_b))


def bootstrap_similarity_ci(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values_a = upper_triangle_values(matrix_a)
    values_b = upper_triangle_values(matrix_b)
    if samples <= 0:
        return (float("nan"), float("nan"))

    sample_size = len(values_a)
    bootstrapped = []
    for _ in range(samples):
        indices = rng.integers(0, sample_size, size=sample_size)
        bootstrapped.append(pearson_arrays(values_a[indices], values_b[indices]))

    return (
        float(np.nanpercentile(bootstrapped, 100.0 * alpha / 2.0)),
        float(np.nanpercentile(bootstrapped, 100.0 * (1.0 - alpha / 2.0))),
    )


def safe_corr(corr_func, values_a: np.ndarray, values_b: np.ndarray) -> float:
    if np.std(values_a) == 0.0 or np.std(values_b) == 0.0:
        return float("nan")
    result = corr_func(values_a, values_b)
    return float(result.statistic)


def pearson_arrays(values_a: np.ndarray, values_b: np.ndarray) -> float:
    centered_a = values_a - np.mean(values_a)
    centered_b = values_b - np.mean(values_b)
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(centered_a, centered_b) / denominator)


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
    progress_callback=None,
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
    values = []
    for trial_number, index in enumerate(sample_indices, start=1):
        values.append(shape_similarity_pearson(shapes[candidates[index][0]], shapes[candidates[index][1]]))
        if progress_callback is not None:
            progress_callback(trial_number, values[-1])
    return np.asarray(values, dtype=float)


def random_permutation_distribution(
    shape_a: DomainShape,
    shape_b: DomainShape,
    trials: int,
    rng: np.random.Generator,
    progress_callback=None,
) -> np.ndarray:
    n_terms = len(shape_a.terms)
    values = []
    for trial_number in range(1, trials + 1):
        permutation = tuple(int(index) for index in rng.permutation(n_terms))
        values.append(shape_similarity_for_permutation(shape_a, shape_b, permutation))
        if progress_callback is not None:
            progress_callback(trial_number, values[-1])
    return np.asarray(values, dtype=float)


def pair_key(pair: tuple[str, str] | list[str]) -> tuple[str, str]:
    return tuple(sorted((pair[0], pair[1])))
