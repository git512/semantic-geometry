from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

DEFAULT_CPU_THREADS = os.environ.get("SEMANTIC_GEOMETRY_CPU_THREADS", "38")
for thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_env_var, DEFAULT_CPU_THREADS)

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .embedder import Embedder
from .geometry import (
    Concept,
    build_domain_shape,
    bootstrap_similarity_ci,
    compare_shapes,
    find_optimal_alignment,
    pair_key,
    permute_shape,
    random_permutation_distribution,
    random_similarity_distribution,
)
from .plots import (
    plot_actual_vs_random,
    plot_alignment_improvements,
    plot_distance_matrix_examples,
    plot_ordered_vs_optimal,
    plot_pair_z_scores,
    plot_permutation_control_example,
    plot_shape_similarity_heatmap,
)


DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
VALID_EMBED_TEXT = {"label_only", "label_plus_description"}
VALID_ALIGNMENT = {"ordered", "optimal", "both"}
console = Console()
error_console = Console(stderr=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether metaphor-paired domains preserve relational geometry in embeddings."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name or local path.")
    parser.add_argument("--data", default="data/domain_sets.yaml", help="Path to domain set YAML.")
    parser.add_argument("--outdir", default="outputs", help="Directory for CSV, JSON, and plots.")
    parser.add_argument("--random-trials", type=int, default=2000, help="Random domain controls per pair.")
    parser.add_argument("--permutation-trials", type=int, default=2000, help="Random permutation controls per pair.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap samples for 95%% CIs.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--embed-text", choices=sorted(VALID_EMBED_TEXT), default="label_only")
    parser.add_argument("--alignment", choices=sorted(VALID_ALIGNMENT), default="both")
    parser.add_argument("--max-bruteforce-n", type=int, default=8)
    parser.add_argument("--alignment-restarts", type=int, default=200)
    parser.add_argument("--alignment-steps", type=int, default=1000)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=int(DEFAULT_CPU_THREADS),
        help="Cap CPU worker threads used by BLAS/OpenMP/PyTorch libraries. Default: 38.",
    )
    parser.add_argument("--quick", action="store_true", help="Use faster exploratory settings.")
    parser.add_argument("--deep", action="store_true", help="Use slower high-sample settings.")
    parser.add_argument("--live-plot", action="store_true", help="Show an optional live matplotlib histogram.")
    return parser.parse_args()


def apply_run_mode(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick and args.deep:
        raise ValueError("Use only one of --quick or --deep.")
    if args.quick:
        args.random_trials = 200
        args.permutation_trials = 200
        args.bootstrap_samples = 200
        args.alignment_restarts = 25
        args.alignment_steps = 200
    elif args.deep:
        args.random_trials = 10000
        args.permutation_trials = 10000
        args.bootstrap_samples = 5000
        args.alignment_restarts = 1000
        args.alignment_steps = 5000
    return args


def load_domain_data(path: Path) -> tuple[dict[str, list[Concept]], list[tuple[str, str]]]:
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
        if not isinstance(config, dict):
            raise ValueError(f"Domain '{name}' must be a mapping.")
        raw_concepts = config.get("concepts")
        if raw_concepts is None:
            raw_concepts = config.get("terms")
        domains[name] = parse_concepts(name, raw_concepts)

    pairs = []
    for item in raw["metaphor_pairs"]:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid metaphor pair entry: {item!r}")
        domain_a, domain_b = item
        if domain_a not in domains or domain_b not in domains:
            raise ValueError(f"Metaphor pair references unknown domain: {item!r}")
        pairs.append((domain_a, domain_b))

    return domains, pairs


def parse_concepts(domain_name: str, raw_concepts) -> list[Concept]:
    if not isinstance(raw_concepts, list):
        raise ValueError(f"Domain '{domain_name}' must define 'concepts' or V1-style 'terms'.")
    if len(raw_concepts) < 2:
        raise ValueError(f"Domain '{domain_name}' needs at least two concepts.")

    concepts = []
    for item in raw_concepts:
        if isinstance(item, str):
            concepts.append(Concept(id=slug_id(item), label=item, description=""))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Invalid concept in '{domain_name}': {item!r}")
        concept_id = item.get("id")
        label = item.get("label")
        description = item.get("description", "")
        if not isinstance(concept_id, str) or not isinstance(label, str):
            raise ValueError(f"Concepts in '{domain_name}' need string 'id' and 'label'.")
        if not isinstance(description, str):
            raise ValueError(f"Concept '{concept_id}' in '{domain_name}' has a non-string description.")
        concepts.append(Concept(id=concept_id, label=label, description=description))

    return concepts


def slug_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def main() -> int:
    try:
        args = apply_run_mode(parse_args())
    except Exception as exc:
        error_console.print(f"[red]Error:[/red] {exc}")
        return 1

    thread_controller = configure_cpu_threads(args.cpu_threads)
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            load_task = progress.add_task("Loading data", total=1)
            domains, metaphor_pairs = load_domain_data(Path(args.data))
            progress.advance(load_task)

            model_task = progress.add_task("Loading embedding model", total=1)
            embedder = Embedder(args.model)
            progress.advance(model_task)

            embed_task = progress.add_task("Embedding concepts", total=len(domains))
            shapes = {}
            for name, concepts in domains.items():
                texts = [concept.embedding_text(args.embed_text) for concept in concepts]
                shapes[name] = build_domain_shape(name, concepts, embedder.embed_terms(texts))
                progress.advance(embed_task)

            compare_task = progress.add_task("Evaluating metaphor pairs", total=len(metaphor_pairs))
            rows, alignments, all_random_domain_values, permutation_examples = evaluate_pairs(
                args=args,
                shapes=shapes,
                metaphor_pairs=metaphor_pairs,
                rng=rng,
                progress=progress,
            )
            progress.advance(compare_task, len(metaphor_pairs))

            output_task = progress.add_task("Writing outputs and rendering plots", total=1)
            write_outputs(
                args=args,
                outdir=outdir,
                plots_dir=plots_dir,
                domains=domains,
                shapes=shapes,
                rows=rows,
                alignments=alignments,
                all_random_domain_values=all_random_domain_values,
                permutation_examples=permutation_examples,
            )
            progress.advance(output_task)
    except Exception as exc:
        error_console.print(f"[red]Error:[/red] {exc}")
        return 1

    print_final_table(pd.DataFrame(rows))
    if thread_controller is not None:
        thread_controller.restore_original_limits()
    return 0


def configure_cpu_threads(cpu_threads: int):
    if cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1.")

    for thread_env_var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[thread_env_var] = str(cpu_threads)

    controller = None
    try:
        from threadpoolctl import threadpool_limits

        controller = threadpool_limits(limits=cpu_threads)
    except Exception:
        controller = None

    try:
        import torch

        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(cpu_threads, 4)))
    except Exception:
        pass

    console.print(f"[dim]CPU thread cap: {cpu_threads}[/dim]")
    return controller


def evaluate_pairs(args, shapes, metaphor_pairs, rng, progress) -> tuple[list[dict], list[dict], list[float], dict]:
    intended_pair_keys = {pair_key(pair) for pair in metaphor_pairs}
    rows = []
    alignments = []
    all_random_domain_values = []
    permutation_examples = {}
    live_plot = LiveHistogram() if args.live_plot else None

    for domain_a, domain_b in metaphor_pairs:
        shape_a = shapes[domain_a]
        shape_b = shapes[domain_b]
        if len(shape_a.terms) != len(shape_b.terms):
            console.print(
                f"[yellow]Warning:[/yellow] skipping {domain_a} / {domain_b}; unequal term counts "
                f"({len(shape_a.terms)} vs {len(shape_b.terms)})."
            )
            continue

        ordered = compare_shapes(shape_a, shape_b)
        if args.alignment == "ordered":
            optimal = ordered
            permutation = tuple(range(len(shape_b.terms)))
            method_used = "ordered"
        else:
            total = math.factorial(len(shape_b.terms)) if len(shape_b.terms) <= args.max_bruteforce_n else (
                args.alignment_restarts * args.alignment_steps
            )
            align_task = progress.add_task(f"Optimal alignment: {domain_a} / {domain_b}", total=total)

            def alignment_progress(checked, best_score, restart=None):
                if checked % 100 == 0 or checked == total:
                    description = f"Optimal alignment: {domain_a} / {domain_b} best={best_score:.3f}"
                    if restart is not None:
                        description += f" restart={restart}"
                    progress.update(align_task, completed=min(checked, total), description=description)

            alignment = find_optimal_alignment(
                shape_a,
                shape_b,
                rng,
                max_bruteforce_n=args.max_bruteforce_n,
                restarts=args.alignment_restarts,
                steps=args.alignment_steps,
                progress_callback=alignment_progress,
            )
            progress.update(align_task, completed=total)
            optimal = alignment.comparison
            permutation = alignment.permutation
            method_used = alignment.method_used

        optimal_shape_b = permute_shape(shape_b, permutation)

        domain_task = progress.add_task(
            f"Random domain controls: {domain_a} / {domain_b}",
            total=args.random_trials,
        )

        def domain_progress(trial_number, value):
            if trial_number % 50 == 0 or trial_number == args.random_trials:
                progress.update(domain_task, completed=trial_number)

        random_domain_values = random_similarity_distribution(
            shapes,
            (domain_a, domain_b),
            args.random_trials,
            rng,
            intended_pair_keys,
            progress_callback=domain_progress,
        )
        if random_domain_values.size == 0:
            console.print(f"[yellow]Warning:[/yellow] no valid random domain controls for {domain_a} / {domain_b}.")
            continue
        all_random_domain_values.extend(random_domain_values.tolist())

        perm_task = progress.add_task(
            f"Random permutation controls: {domain_a} / {domain_b}",
            total=args.permutation_trials,
        )

        permutation_so_far = []

        def permutation_progress(trial_number, value):
            permutation_so_far.append(value)
            if trial_number % 50 == 0 or trial_number == args.permutation_trials:
                progress.update(perm_task, completed=trial_number)
                if live_plot is not None:
                    live_plot.update(
                        title=f"{domain_a} / {domain_b}",
                        values=permutation_so_far,
                        marker=optimal.pearson,
                    )

        random_permutation_values = random_permutation_distribution(
            shape_a,
            shape_b,
            args.permutation_trials,
            rng,
            progress_callback=permutation_progress,
        )
        permutation_examples[f"{domain_a}/{domain_b}"] = random_permutation_values

        ci_task = progress.add_task(f"Bootstrap CIs: {domain_a} / {domain_b}", total=1)
        ordered_ci_low, ordered_ci_high = bootstrap_similarity_ci(
            shape_a.cosine_distance,
            shape_b.cosine_distance,
            args.bootstrap_samples,
            rng,
        )
        optimal_ci_low, optimal_ci_high = bootstrap_similarity_ci(
            shape_a.cosine_distance,
            optimal_shape_b.cosine_distance,
            args.bootstrap_samples,
            rng,
        )
        progress.advance(ci_task)

        domain_stats = control_stats(optimal.pearson, random_domain_values)
        permutation_stats = control_stats(optimal.pearson, random_permutation_values)
        improvement = float(optimal.pearson - ordered.pearson)
        mapping = {
            shape_a.concepts[index].id: shape_b.concepts[permutation[index]].id
            for index in range(len(shape_a.concepts))
        }

        rows.append(
            {
                "domain_a": domain_a,
                "domain_b": domain_b,
                "n_terms": len(shape_a.terms),
                "ordered_pearson": ordered.pearson,
                "ordered_spearman": ordered.spearman,
                "ordered_frobenius": ordered.frobenius_difference,
                "optimal_pearson": optimal.pearson,
                "optimal_spearman": optimal.spearman,
                "optimal_frobenius": optimal.frobenius_difference,
                "alignment_improvement": improvement,
                "alignment_method": method_used,
                "random_domain_mean": domain_stats["mean"],
                "random_domain_std": domain_stats["std"],
                "random_domain_z_score": domain_stats["z_score"],
                "random_domain_percentile": domain_stats["percentile"],
                "random_domain_p_estimate": domain_stats["p_estimate"],
                "random_permutation_mean": permutation_stats["mean"],
                "random_permutation_std": permutation_stats["std"],
                "random_permutation_z_score": permutation_stats["z_score"],
                "random_permutation_percentile": permutation_stats["percentile"],
                "random_permutation_p_estimate": permutation_stats["p_estimate"],
                "ordered_similarity_ci_low": ordered_ci_low,
                "ordered_similarity_ci_high": ordered_ci_high,
                "optimal_similarity_ci_low": optimal_ci_low,
                "optimal_similarity_ci_high": optimal_ci_high,
            }
        )
        alignments.append(
            {
                "domain_a": domain_a,
                "domain_b": domain_b,
                "ordered_similarity": ordered.pearson,
                "optimal_similarity": optimal.pearson,
                "improvement": improvement,
                "mapping": mapping,
                "domain_a_labels": shape_a.labels,
                "domain_b_labels_ordered": shape_b.labels,
                "domain_b_labels_optimal": optimal_shape_b.labels,
                "method_used": method_used,
            }
        )

    if live_plot is not None:
        live_plot.pause()
    return rows, alignments, all_random_domain_values, permutation_examples


def control_stats(actual: float, random_values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(random_values))
    std = float(np.std(random_values, ddof=0))
    return {
        "mean": mean,
        "std": std,
        "z_score": float((actual - mean) / std) if std > 0 else float("nan"),
        "percentile": float(100.0 * np.mean(random_values <= actual)),
        "p_estimate": float(np.mean(random_values >= actual)),
    }


class LiveHistogram:
    def __init__(self) -> None:
        try:
            import matplotlib

            if matplotlib.get_backend().lower() == "agg":
                for backend in ("QtAgg", "TkAgg", "GTK3Agg", "WXAgg"):
                    try:
                        matplotlib.use(backend, force=True)
                        break
                    except Exception:
                        continue
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(f"Could not enable --live-plot: {exc}") from exc

        self.plt = plt
        if self.plt.get_backend().lower() == "agg":
            raise RuntimeError(
                "Could not enable --live-plot because matplotlib is using the non-interactive Agg backend. "
                "Install a GUI backend such as PyQt/PySide or Tk, or set MPLBACKEND=QtAgg/TkAgg."
            )
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots(figsize=(8, 5))

    def update(self, title: str, values: list[float], marker: float) -> None:
        if len(values) < 2:
            return
        self.ax.clear()
        self.ax.hist(values, bins=min(35, max(8, int(math.sqrt(len(values))))), color="#9E768F", alpha=0.78)
        self.ax.axvline(marker, color="#1B998B", linewidth=2.2, label="Optimal alignment")
        self.ax.set_title(f"Live permutation controls: {title}")
        self.ax.set_xlabel("Shape similarity (Pearson r)")
        self.ax.set_ylabel("Random permutation count")
        self.ax.legend()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def pause(self) -> None:
        self.plt.pause(0.5)


def write_outputs(
    args,
    outdir: Path,
    plots_dir: Path,
    domains,
    shapes,
    rows,
    alignments,
    all_random_domain_values,
    permutation_examples,
) -> None:
    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No valid metaphor pairs were evaluated.")

    results.to_csv(outdir / "results.csv", index=False)
    with (outdir / "alignments.json").open("w", encoding="utf-8") as handle:
        json.dump(alignments, handle, indent=2)
        handle.write("\n")

    domain_names = list(domains)
    heatmap = build_similarity_heatmap(domain_names, shapes)
    random_domain_array = np.asarray(all_random_domain_values, dtype=float)
    strongest_index = results["optimal_pearson"].idxmax()
    strongest_pair = f"{results.loc[strongest_index, 'domain_a']}/{results.loc[strongest_index, 'domain_b']}"

    best_alignment = alignments[int(strongest_index)]
    optimal_example_shape = permute_shape(
        shapes[best_alignment["domain_b"]],
        tuple(shapes[best_alignment["domain_b"]].terms.index(target) for target in best_alignment["mapping"].values()),
    )

    plot_actual_vs_random(
        random_domain_array,
        float(results["ordered_pearson"].mean()),
        float(results["optimal_pearson"].mean()),
        plots_dir / "actual_vs_random_similarity.png",
    )
    plot_pair_z_scores(results, plots_dir / "pair_z_scores.png")
    plot_distance_matrix_examples(
        shapes,
        best_alignment["domain_a"],
        best_alignment["domain_b"],
        optimal_example_shape,
        plots_dir / "distance_matrix_examples.png",
    )
    plot_shape_similarity_heatmap(domain_names, heatmap, plots_dir / "shape_similarity_heatmap.png")
    plot_ordered_vs_optimal(results, plots_dir / "ordered_vs_optimal_similarity.png")
    plot_alignment_improvements(results, plots_dir / "alignment_improvements.png")
    plot_permutation_control_example(
        permutation_examples[strongest_pair],
        float(results.loc[strongest_index, "optimal_pearson"]),
        strongest_pair,
        plots_dir / "permutation_control_examples.png",
    )

    summary = build_summary(args, domains, results, random_domain_array)
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def build_summary(args, domains, results: pd.DataFrame, random_domain_array: np.ndarray) -> dict:
    strongest_optimal_index = results["optimal_pearson"].idxmax()
    strongest_z_index = results["random_domain_z_score"].idxmax()
    weakest_index = results["optimal_pearson"].idxmin()

    def pair_record(index) -> dict:
        row = results.loc[index]
        return {
            "domain_a": row["domain_a"],
            "domain_b": row["domain_b"],
            "optimal_pearson": float(row["optimal_pearson"]),
            "random_domain_z_score": float(row["random_domain_z_score"]),
            "random_permutation_z_score": float(row["random_permutation_z_score"]),
        }

    return {
        "version": 2,
        "model": args.model,
        "embed_text": args.embed_text,
        "seed": args.seed,
        "number_of_domains": len(domains),
        "number_of_metaphor_pairs": int(len(results)),
        "number_of_random_domain_trials": args.random_trials,
        "number_of_random_permutation_trials": args.permutation_trials,
        "bootstrap_samples": args.bootstrap_samples,
        "mean_ordered_similarity": float(results["ordered_pearson"].mean()),
        "mean_optimal_similarity": float(results["optimal_pearson"].mean()),
        "mean_alignment_improvement": float(results["alignment_improvement"].mean()),
        "mean_random_domain_similarity": float(np.mean(random_domain_array)),
        "mean_random_permutation_similarity": float(results["random_permutation_mean"].mean()),
        "count_optimal_above_90th_percentile_domain_controls": int((results["random_domain_percentile"] >= 90).sum()),
        "count_optimal_above_95th_percentile_domain_controls": int((results["random_domain_percentile"] >= 95).sum()),
        "count_optimal_above_99th_percentile_domain_controls": int((results["random_domain_percentile"] >= 99).sum()),
        "count_optimal_above_90th_percentile_permutation_controls": int(
            (results["random_permutation_percentile"] >= 90).sum()
        ),
        "count_optimal_above_95th_percentile_permutation_controls": int(
            (results["random_permutation_percentile"] >= 95).sum()
        ),
        "count_optimal_above_99th_percentile_permutation_controls": int(
            (results["random_permutation_percentile"] >= 99).sum()
        ),
        "strongest_pair_by_optimal_similarity": pair_record(strongest_optimal_index),
        "strongest_pair_by_z_score": pair_record(strongest_z_index),
        "weakest_pair": pair_record(weakest_index),
    }


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


def print_final_table(results: pd.DataFrame) -> None:
    table = Table(title="V2 Metaphor Geometry Results")
    table.add_column("Pair")
    table.add_column("Ordered", justify="right")
    table.add_column("Optimal", justify="right")
    table.add_column("Improve", justify="right")
    table.add_column("Domain z", justify="right")
    table.add_column("Perm z", justify="right")
    table.add_column("Perm %", justify="right")

    for _, row in results.iterrows():
        table.add_row(
            f"{row['domain_a']} / {row['domain_b']}",
            f"{row['ordered_pearson']:.3f}",
            f"{row['optimal_pearson']:.3f}",
            f"{row['alignment_improvement']:.3f}",
            f"{row['random_domain_z_score']:.2f}",
            f"{row['random_permutation_z_score']:.2f}",
            f"{row['random_permutation_percentile']:.1f}",
        )
    console.print(table)

    strongest = results.loc[results["optimal_pearson"].idxmax()]
    weakest = results.loc[results["optimal_pearson"].idxmin()]
    above_95 = int((results["random_permutation_percentile"] >= 95).sum())
    console.print(
        Panel(
            "\n".join(
                [
                    f"Strongest optimal pair: {strongest['domain_a']} / {strongest['domain_b']} "
                    f"({strongest['optimal_pearson']:.3f})",
                    f"Weakest optimal pair: {weakest['domain_a']} / {weakest['domain_b']} "
                    f"({weakest['optimal_pearson']:.3f})",
                    f"Mean alignment improvement: {results['alignment_improvement'].mean():.3f}",
                    f"Pairs above 95th percentile permutation controls: {above_95}/{len(results)}",
                ]
            ),
            title="Interpretation Snapshot",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
