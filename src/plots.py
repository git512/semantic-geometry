from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import DomainShape


def get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def plot_actual_vs_random(
    random_values: np.ndarray,
    ordered_mean: float,
    optimal_mean: float,
    path: Path,
) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(random_values, bins=40, color="#547AA5", alpha=0.78, edgecolor="white")
    ax.axvline(ordered_mean, color="#C44900", linewidth=2.2, label="Mean ordered similarity")
    ax.axvline(optimal_mean, color="#2F9C95", linewidth=2.2, label="Mean optimal similarity")
    ax.set_title("Actual Metaphor Similarity vs Random Controls")
    ax.set_xlabel("Shape similarity (Pearson r)")
    ax.set_ylabel("Random comparison count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pair_z_scores(results: pd.DataFrame, path: Path) -> None:
    plt = get_pyplot()
    labels = results["domain_a"] + " / " + results["domain_b"]
    fig_height = max(5, 0.42 * len(results) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = np.arange(len(results))
    height = 0.36
    ax.barh(y - height / 2, results["random_domain_z_score"], height=height, color="#6A994E", label="Domain controls")
    ax.barh(
        y + height / 2,
        results["random_permutation_z_score"],
        height=height,
        color="#BC6C25",
        label="Permutation controls",
    )
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_title("Metaphor Pair Z-Scores")
    ax.set_xlabel("Z-score vs random controls")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_distance_matrix_examples(
    shapes: dict[str, DomainShape],
    domain_a: str,
    domain_b: str,
    optimal_shape_b: DomainShape | None,
    path: Path,
) -> None:
    plt = get_pyplot()
    columns = 3 if optimal_shape_b is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(5.5 * columns, 5))
    matrices = [shapes[domain_a].cosine_distance, shapes[domain_b].cosine_distance]
    domains = [domain_a, domain_b]
    labels = [shapes[domain_a].labels, shapes[domain_b].labels]
    titles = [domain_a, f"{domain_b} ordered"]
    if optimal_shape_b is not None:
        matrices.append(optimal_shape_b.cosine_distance)
        domains.append(domain_b)
        labels.append(optimal_shape_b.labels)
        titles.append(f"{domain_b} optimal order")
    vmax = max(float(np.max(matrix)) for matrix in matrices)

    for ax, title, matrix, tick_labels in zip(axes, titles, matrices, labels, strict=True):
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(tick_labels)))
        ax.set_yticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="Cosine distance")
    fig.suptitle("Cosine-Distance Matrix Examples")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shape_similarity_heatmap(
    domain_names: list[str],
    similarity_matrix: np.ndarray,
    path: Path,
) -> None:
    plt = get_pyplot()
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


def plot_ordered_vs_optimal(results: pd.DataFrame, path: Path) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(results["ordered_pearson"], results["optimal_pearson"], color="#386FA4", s=54)
    min_value = float(min(results["ordered_pearson"].min(), results["optimal_pearson"].min()))
    max_value = float(max(results["ordered_pearson"].max(), results["optimal_pearson"].max()))
    pad = 0.05
    ax.plot([min_value - pad, max_value + pad], [min_value - pad, max_value + pad], color="#333333", linewidth=1)
    ax.set_title("Ordered vs Optimal Shape Similarity")
    ax.set_xlabel("Ordered Pearson similarity")
    ax.set_ylabel("Optimal Pearson similarity")
    ax.set_xlim(min_value - pad, max_value + pad)
    ax.set_ylim(min_value - pad, max_value + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_alignment_improvements(results: pd.DataFrame, path: Path) -> None:
    plt = get_pyplot()
    labels = results["domain_a"] + " / " + results["domain_b"]
    fig_height = max(5, 0.42 * len(results) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(labels, results["alignment_improvement"], color="#2F9C95")
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_title("Optimal Alignment Improvement")
    ax.set_xlabel("Optimal Pearson - ordered Pearson")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_permutation_control_example(
    random_values: np.ndarray,
    optimal_score: float,
    pair_label: str,
    path: Path,
) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(random_values, bins=40, color="#9E768F", alpha=0.78, edgecolor="white")
    ax.axvline(optimal_score, color="#1B998B", linewidth=2.5, label="Optimal alignment")
    ax.set_title(f"Permutation Controls: {pair_label}")
    ax.set_xlabel("Shape similarity (Pearson r)")
    ax.set_ylabel("Random permutation count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dimensionality_recovery_curves(curves: pd.DataFrame, path: Path, individual_dir: Path) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    individual_dir.mkdir(parents=True, exist_ok=True)
    for (domain_a, domain_b), group in curves.groupby(["domain_a", "domain_b"]):
        label = f"{domain_a} / {domain_b}"
        ax.plot(group["k"], group["fraction_of_full_similarity_recovered"], marker="o", linewidth=1.4, label=label)

        pair_fig, pair_ax = plt.subplots(figsize=(7, 4.5))
        pair_ax.plot(group["k"], group["fraction_of_full_similarity_recovered"], marker="o", color="#386FA4")
        pair_ax.axhline(0.9, color="#333333", linewidth=1, linestyle="--")
        pair_ax.set_title(label)
        pair_ax.set_xlabel("Dimensions")
        pair_ax.set_ylabel("Fraction of full similarity recovered")
        pair_fig.tight_layout()
        pair_fig.savefig(individual_dir / f"{domain_a}__{domain_b}.png", dpi=180)
        plt.close(pair_fig)

    ax.axhline(0.9, color="#333333", linewidth=1, linestyle="--")
    ax.set_title("Dimensionality Recovery Curves")
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Fraction of full similarity recovered")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_procrustes_orientation_curves(curves: pd.DataFrame, path: Path) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (domain_a, domain_b), group in curves.groupby(["domain_a", "domain_b"]):
        ax.plot(group["k"], group["orientation_similarity"], marker="o", linewidth=1.4, label=f"{domain_a} / {domain_b}")
    ax.set_title("Procrustes Orientation Curves")
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Orientation similarity")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_domain_pca_spectra(spectra: pd.DataFrame, path: Path) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for domain, group in spectra.groupby("domain"):
        ax.plot(group["component"], group["cumulative_explained_variance_ratio"], marker="o", linewidth=1.4, label=domain)
    ax.axhline(0.9, color="#333333", linewidth=1, linestyle="--")
    ax.set_title("Domain PCA Cumulative Variance")
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative explained variance")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_effective_dimensionality(summary: pd.DataFrame, path: Path) -> None:
    plt = get_pyplot()
    labels = summary["domain"]
    y = np.arange(len(summary))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(summary) + 1.5)))
    ax.barh(y - height / 2, summary["participation_ratio"], height=height, label="Participation ratio", color="#386FA4")
    ax.barh(y + height / 2, summary["spectral_entropy"], height=height, label="Spectral entropy dim", color="#BC6C25")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Effective Dimensionality")
    ax.set_xlabel("Effective dimensions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_k90_shape_recovery(results: pd.DataFrame, path: Path) -> None:
    plt = get_pyplot()
    labels = results["domain_a"] + " / " + results["domain_b"]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(results) + 1.5)))
    ax.barh(labels, results["k_90_shape_recovery"], color="#2F9C95")
    ax.invert_yaxis()
    ax.set_title("k90 Shape Recovery")
    ax.set_xlabel("Smallest k recovering 90% of full shape similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_plausibility_vs_geometry(results: pd.DataFrame, path: Path) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(results["average_plausibility_score"], results["optimal_pearson"], color="#6A994E", s=56)
    ax.set_title("Plausibility vs Geometry")
    ax.set_xlabel("Average plausibility score")
    ax.set_ylabel("Optimal Pearson geometry score")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_component_similarity_examples(component_rows: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    plt = get_pyplot()
    if component_rows.empty or "domain_a" not in component_rows.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Component similarity is available for PCA runs.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return
    strongest = results.loc[results["optimal_pearson"].idxmax()]
    weakest = results.loc[results["optimal_pearson"].idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, row, title in zip(axes, [strongest, weakest], ["Strongest", "Weakest"], strict=True):
        group = component_rows[
            (component_rows["domain_a"] == row["domain_a"]) & (component_rows["domain_b"] == row["domain_b"])
        ]
        ax.bar(group["component"], group["component_similarity"], color="#547AA5")
        ax.set_title(f"{title}: {row['domain_a']} / {row['domain_b']}")
        ax.set_xlabel("Component")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Component similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
