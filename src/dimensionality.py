from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from .geometry import DomainShape, pearson_arrays


@dataclass(frozen=True)
class DomainDimensionality:
    domain: str
    n_terms: int
    dim_50: int
    dim_75: int
    dim_90: int
    dim_95: int
    participation_ratio: float
    spectral_entropy: float


def centered(embeddings: np.ndarray) -> np.ndarray:
    return embeddings - np.mean(embeddings, axis=0, keepdims=True)


def pca_coordinates(embeddings: np.ndarray, max_components: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n_terms = embeddings.shape[0]
    n_components = min(max_components or n_terms - 1, n_terms - 1, embeddings.shape[1])
    model = PCA(n_components=n_components)
    coords = model.fit_transform(centered(embeddings))
    return coords, model.explained_variance_ratio_


def mds_coordinates(distance_matrix: np.ndarray, max_components: int) -> np.ndarray:
    coords_by_k = []
    for k in range(1, max_components + 1):
        model = MDS(
            n_components=k,
            metric="precomputed",
            random_state=42,
            normalized_stress="auto",
            init="random",
            n_init=4,
            max_iter=300,
        )
        coords_by_k.append(model.fit_transform(distance_matrix))
    return coords_by_k


def pca_spectrum(shape: DomainShape) -> tuple[list[dict], DomainDimensionality]:
    _, ratios = pca_coordinates(shape.embeddings)
    cumulative = np.cumsum(ratios)
    rows = [
        {
            "domain": shape.name,
            "component": index + 1,
            "explained_variance_ratio": float(ratio),
            "cumulative_explained_variance_ratio": float(cumulative[index]),
        }
        for index, ratio in enumerate(ratios)
    ]
    summary = DomainDimensionality(
        domain=shape.name,
        n_terms=len(shape.terms),
        dim_50=threshold_dim(cumulative, 0.50),
        dim_75=threshold_dim(cumulative, 0.75),
        dim_90=threshold_dim(cumulative, 0.90),
        dim_95=threshold_dim(cumulative, 0.95),
        participation_ratio=participation_ratio(ratios),
        spectral_entropy=spectral_entropy_dim(ratios),
    )
    return rows, summary


def threshold_dim(cumulative: np.ndarray, threshold: float) -> int:
    indices = np.where(cumulative >= threshold)[0]
    if len(indices) == 0:
        return int(len(cumulative))
    return int(indices[0] + 1)


def participation_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    denominator = float(np.sum(values**2))
    if total <= 0 or denominator <= 0:
        return float("nan")
    return float((total**2) / denominator)


def spectral_entropy_dim(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    if total <= 0:
        return float("nan")
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def orthogonal_procrustes_metrics(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[float, float, np.ndarray]:
    a = centered(coords_a)
    b = centered(coords_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan"), float("nan"), b
    a = a / norm_a
    b = b / norm_b
    u, _, vt = np.linalg.svd(b.T @ a, full_matrices=False)
    rotation = u @ vt
    b_aligned = b @ rotation
    disparity = float(np.sum((a - b_aligned) ** 2))
    orientation_similarity = float(max(0.0, min(1.0, 1.0 - disparity)))
    return disparity, orientation_similarity, b_aligned


def distance_similarity(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    dist_a = squareform(pdist(coords_a, metric="euclidean"))
    dist_b = squareform(pdist(coords_b, metric="euclidean"))
    indices = np.triu_indices_from(dist_a, k=1)
    return pearson_arrays(dist_a[indices], dist_b[indices])


def dimensionality_curves(
    shape_a: DomainShape,
    aligned_shape_b: DomainShape,
    full_similarity: float,
    reduction_method: str,
) -> tuple[list[dict], list[dict], dict]:
    n_terms = len(shape_a.terms)
    max_k = n_terms - 1
    a_full_pca, a_ratios = pca_coordinates(shape_a.embeddings, max_k)
    b_full_pca, b_ratios = pca_coordinates(aligned_shape_b.embeddings, max_k)

    if reduction_method == "pca":
        coords_a_by_k = [a_full_pca[:, :k] for k in range(1, max_k + 1)]
        coords_b_by_k = [b_full_pca[:, :k] for k in range(1, max_k + 1)]
    elif reduction_method == "mds":
        coords_a_by_k = mds_coordinates(shape_a.cosine_distance, max_k)
        coords_b_by_k = mds_coordinates(aligned_shape_b.cosine_distance, max_k)
    else:
        raise ValueError(f"Unknown reduction method: {reduction_method}")

    curve_rows = []
    component_rows = []
    best_orientation = -np.inf
    best_orientation_k = 1
    best_disparity = float("nan")
    best_shape_similarity = -np.inf
    best_shape_k = 1

    for k, (coords_a, coords_b) in enumerate(zip(coords_a_by_k, coords_b_by_k, strict=True), start=1):
        disparity, orientation_similarity, aligned_b = orthogonal_procrustes_metrics(coords_a, coords_b)
        reconstructed_similarity = distance_similarity(coords_a, coords_b)
        fraction_recovered = safe_fraction(reconstructed_similarity, full_similarity)
        a_variance = float(np.sum(a_ratios[:k]))
        b_variance = float(np.sum(b_ratios[:k]))

        if orientation_similarity > best_orientation:
            best_orientation = orientation_similarity
            best_orientation_k = k
            best_disparity = disparity
        if fraction_recovered > best_shape_similarity:
            best_shape_similarity = fraction_recovered
            best_shape_k = k

        curve_rows.append(
            {
                "domain_a": shape_a.name,
                "domain_b": aligned_shape_b.name,
                "k": k,
                "reduction_method": reduction_method,
                "a_variance_explained": a_variance,
                "b_variance_explained": b_variance,
                "mean_variance_explained": float((a_variance + b_variance) / 2.0),
                "procrustes_disparity": disparity,
                "orientation_similarity": orientation_similarity,
                "reconstructed_distance_similarity": reconstructed_similarity,
                "fraction_of_full_similarity_recovered": fraction_recovered,
            }
        )

        if reduction_method == "pca":
            for component in range(k):
                component_rows.append(
                    {
                        "domain_a": shape_a.name,
                        "domain_b": aligned_shape_b.name,
                        "component": component + 1,
                        "component_similarity": component_similarity(
                            coords_a[:, component],
                            aligned_b[:, component],
                        ),
                        "a_explained_variance_ratio": float(a_ratios[component]),
                        "b_explained_variance_ratio": float(b_ratios[component]),
                    }
                )

    summary = {
        "best_k_by_orientation": best_orientation_k,
        "best_orientation_similarity": float(best_orientation),
        "best_procrustes_disparity": float(best_disparity),
        "k_90_similarity": recovery_k(curve_rows, "orientation_similarity", 0.90 * best_orientation),
        "k_95_similarity": recovery_k(curve_rows, "orientation_similarity", 0.95 * best_orientation),
        "k_80_shape_recovery": recovery_k(curve_rows, "fraction_of_full_similarity_recovered", 0.80),
        "k_90_shape_recovery": recovery_k(curve_rows, "fraction_of_full_similarity_recovered", 0.90),
        "k_95_shape_recovery": recovery_k(curve_rows, "fraction_of_full_similarity_recovered", 0.95),
        "best_k_by_shape_recovery": best_shape_k,
    }
    return curve_rows, dedupe_component_rows(component_rows), summary


def dedupe_component_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        key = (row["domain_a"], row["domain_b"], row["component"])
        if key not in seen:
            deduped.append(row)
            seen.add(key)
    return deduped


def component_similarity(values_a: np.ndarray, values_b: np.ndarray) -> float:
    similarity = pearson_arrays(values_a, values_b)
    return float(abs(similarity)) if not np.isnan(similarity) else float("nan")


def safe_fraction(value: float, denominator: float) -> float:
    if denominator == 0.0 or np.isnan(denominator):
        return float("nan")
    return float(value / denominator)


def recovery_k(rows: list[dict], key: str, threshold: float) -> int:
    if np.isnan(threshold):
        return 0
    for row in rows:
        value = row[key]
        if not np.isnan(value) and value >= threshold:
            return int(row["k"])
    return int(rows[-1]["k"]) if rows else 0
