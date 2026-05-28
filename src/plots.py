from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry import DomainShape


def plot_actual_vs_random(random_values: np.ndarray, actual_mean: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(random_values, bins=40, color="#547AA5", alpha=0.78, edgecolor="white")
    ax.axvline(actual_mean, color="#C44900", linewidth=2.5, label="Mean actual similarity")
    ax.set_title("Actual Metaphor Similarity vs Random Controls")
    ax.set_xlabel("Shape similarity (Pearson r)")
    ax.set_ylabel("Random comparison count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pair_z_scores(results: pd.DataFrame, path: Path) -> None:
    labels = results["domain_a"] + " / " + results["domain_b"]
    fig_height = max(5, 0.42 * len(results) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(labels, results["z_score"], color="#6A994E")
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_title("Metaphor Pair Z-Scores")
    ax.set_xlabel("Z-score vs random controls")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_distance_matrix_examples(
    shapes: dict[str, DomainShape],
    domain_a: str,
    domain_b: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    matrices = [shapes[domain_a].cosine_distance, shapes[domain_b].cosine_distance]
    vmax = max(float(np.max(matrix)) for matrix in matrices)

    for ax, domain, matrix in zip(axes, [domain_a, domain_b], matrices, strict=True):
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(domain)
        ax.set_xticks(range(len(shapes[domain].terms)))
        ax.set_yticks(range(len(shapes[domain].terms)))
        ax.set_xticklabels(shapes[domain].terms, rotation=45, ha="right")
        ax.set_yticklabels(shapes[domain].terms)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="Cosine distance")
    fig.suptitle("Cosine-Distance Matrix Examples")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shape_similarity_heatmap(
    domain_names: list[str],
    similarity_matrix: np.ndarray,
    path: Path,
) -> None:
    fig_size = max(7, 0.58 * len(domain_names) + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(similarity_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Shape Similarity Between Domains")
    ax.set_xticks(range(len(domain_names)))
    ax.set_yticks(range(len(domain_names)))
    ax.set_xticklabels(domain_names, rotation=45, ha="right")
    ax.set_yticklabels(domain_names)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
