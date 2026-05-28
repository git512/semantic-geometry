from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .embedder import Embedder
from .geometry import (
    build_domain_shape,
    compare_shapes,
    pair_key,
    random_similarity_distribution,
)
from .plots import (
    plot_actual_vs_random,
    plot_distance_matrix_examples,
    plot_pair_z_scores,
    plot_shape_similarity_heatmap,
)


DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether metaphor-paired domains preserve relational geometry in embeddings."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name or local path.")
    parser.add_argument("--data", default="data/domain_sets.yaml", help="Path to domain set YAML.")
    parser.add_argument("--outdir", default="outputs", help="Directory for CSV, JSON, and plots.")
    parser.add_argument("--random-trials", type=int, default=1000, help="Random control comparisons per pair.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    return parser.parse_args()


def load_domain_data(path: Path) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    if not path.exists():
        raise ValueError(f"Data file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Data file is not valid YAML: {path}") from exc

    if not isinstance(raw, dict) or "domains" not in raw or "metaphor_pairs" not in raw:
        raise ValueError("Data file must contain top-level 'domains' and 'metaphor_pairs' keys.")

    domains = {}
    for name, config in raw["domains"].items():
        terms = config.get("terms") if isinstance(config, dict) else None
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            raise ValueError(f"Domain '{name}' must define a string list under 'terms'.")
        if len(terms) < 2:
            raise ValueError(f"Domain '{name}' needs at least two terms.")
        domains[name] = terms

    pairs = []
    for item in raw["metaphor_pairs"]:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid metaphor pair entry: {item!r}")
        domain_a, domain_b = item
        if domain_a not in domains or domain_b not in domains:
            raise ValueError(f"Metaphor pair references unknown domain: {item!r}")
        pairs.append((domain_a, domain_b))

    return domains, pairs


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        domains, metaphor_pairs = load_domain_data(Path(args.data))
        embedder = Embedder(args.model)
        shapes = {
            name: build_domain_shape(name, terms, embedder.embed_terms(terms))
            for name, terms in domains.items()
        }
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    intended_pair_keys = {pair_key(pair) for pair in metaphor_pairs}
    rows = []
    all_random_values = []

    for domain_a, domain_b in metaphor_pairs:
        shape_a = shapes[domain_a]
        shape_b = shapes[domain_b]
        if len(shape_a.terms) != len(shape_b.terms):
            print(
                f"Warning: skipping {domain_a} / {domain_b}; unequal term counts "
                f"({len(shape_a.terms)} vs {len(shape_b.terms)}).",
                file=sys.stderr,
            )
            continue

        actual = compare_shapes(shape_a, shape_b)
        random_values = random_similarity_distribution(
            shapes,
            (domain_a, domain_b),
            args.random_trials,
            rng,
            intended_pair_keys,
        )
        if random_values.size == 0:
            print(f"Warning: no valid random controls for {domain_a} / {domain_b}.", file=sys.stderr)
            continue

        all_random_values.extend(random_values.tolist())
        random_mean = float(np.mean(random_values))
        random_std = float(np.std(random_values, ddof=0))
        z_score = float((actual.pearson - random_mean) / random_std) if random_std > 0 else float("nan")
        percentile = float(100.0 * np.mean(random_values <= actual.pearson))
        p_estimate = float(np.mean(random_values >= actual.pearson))

        rows.append(
            {
                "domain_a": domain_a,
                "domain_b": domain_b,
                "n_terms": len(shape_a.terms),
                "shape_similarity_pearson": actual.pearson,
                "shape_similarity_spearman": actual.spearman,
                "frobenius_difference": actual.frobenius_difference,
                "procrustes_disparity": actual.procrustes_disparity,
                "random_mean": random_mean,
                "random_std": random_std,
                "z_score": z_score,
                "percentile": percentile,
                "p_estimate": p_estimate,
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        print("Error: no valid metaphor pairs were evaluated.", file=sys.stderr)
        return 1

    results.to_csv(outdir / "results.csv", index=False)

    domain_names = list(domains)
    heatmap = build_similarity_heatmap(domain_names, shapes)
    random_array = np.asarray(all_random_values, dtype=float)

    plot_actual_vs_random(
        random_array,
        float(results["shape_similarity_pearson"].mean()),
        plots_dir / "actual_vs_random_similarity.png",
    )
    plot_pair_z_scores(results, plots_dir / "pair_z_scores.png")
    plot_distance_matrix_examples(shapes, "electrical_flow", "fluid_flow", plots_dir / "distance_matrix_examples.png")
    plot_shape_similarity_heatmap(domain_names, heatmap, plots_dir / "shape_similarity_heatmap.png")

    summary = {
        "model": args.model,
        "seed": args.seed,
        "number_of_domains": len(domains),
        "number_of_metaphor_pairs": int(len(results)),
        "number_of_random_trials": args.random_trials,
        "mean_actual_similarity": float(results["shape_similarity_pearson"].mean()),
        "mean_random_similarity": float(np.mean(random_array)),
        "mean_z_score": float(results["z_score"].mean()),
        "count_above_90th_percentile": int((results["percentile"] >= 90.0).sum()),
        "count_above_95th_percentile": int((results["percentile"] >= 95.0).sum()),
        "count_above_99th_percentile": int((results["percentile"] >= 99.0).sum()),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {outdir / 'results.csv'}")
    print(f"Wrote {outdir / 'summary.json'}")
    print(f"Wrote plots to {plots_dir}")
    return 0


def build_similarity_heatmap(domain_names: list[str], shapes: dict) -> np.ndarray:
    matrix = np.ones((len(domain_names), len(domain_names)), dtype=float)
    for i, domain_a in enumerate(domain_names):
        for j, domain_b in enumerate(domain_names):
            if i == j:
                continue
            if len(shapes[domain_a].terms) != len(shapes[domain_b].terms):
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = compare_shapes(shapes[domain_a], shapes[domain_b]).pearson
    return matrix


if __name__ == "__main__":
    raise SystemExit(main())
