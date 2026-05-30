# Recursive Resonance Semantic Geometry Lab

This project tests whether metaphorically compatible concept domains preserve similar relational geometry in embedding space.

V3 extends the original ordered-distance experiment with phrase-based concepts, optimal alignment search, stronger controls, bootstrap confidence intervals, dimensionality analysis, Procrustes orientation alignment, semantic plausibility scoring, richer plots, and rich terminal progress output.

## What It Tests

Each domain is represented as a small list of concepts. Each concept has a stable `id`, an embedding `label`, and an optional `description`. The program embeds those concepts, computes pairwise distance matrices inside each domain, and compares one domain's relational shape with another's.

The V3 hypothesis is narrower than a full theory: metaphorically compatible domains may preserve a recoverable multidimensional pattern of relational orientations, and the resulting alignments should remain at least partly semantically plausible.

V2 asks whether two domains can be rearranged into similar relational distance-shapes. V3 asks how many dimensions carry that shared shape, whether orientation can be aligned, and whether the resulting concept mappings remain semantically plausible.

## Install

Use Python 3.11 or newer.

With `uv`:

```bash
uv venv --python /usr/bin/python3.11 .venv
uv pip install -r requirements.txt
```

Or with pip:

```bash
pip install -r requirements.txt
```

The default model is `sentence-transformers/all-mpnet-base-v2`. Runtime does not require internet if the selected model is already cached locally by Sentence Transformers.

## Run

From the project root:

```bash
source .venv/bin/activate
python -m src.main --quick
```

Default run:

```bash
python -m src.main
```

Deeper run:

```bash
python -m src.main --deep
```

V3 deep run:

```bash
python -m src.main --v3-deep
```

Useful options:

```bash
python -m src.main --model sentence-transformers/all-MiniLM-L6-v2
python -m src.main --embed-text label_plus_description
python -m src.main --alignment both --random-trials 5000 --permutation-trials 5000 --seed 42
python -m src.main --data data/domain_sets.yaml --outdir outputs
python -m src.main --quick --live-plot
python -m src.main --cpu-threads 4
python -m src.main --reduction pca
python -m src.main --reduction mds
```

The default CPU thread cap is `38`, leaving some headroom on a 48-thread workstation. Lower it if you want the desktop to stay more relaxed during a run.

If your environment does not provide a `python` alias, use `python3 -m src.main`.

## Alignment

The ordered comparison preserves the V1 baseline: concept 1 in domain A is compared to concept 1 in domain B, concept 2 to concept 2, and so on.

The optimal comparison searches for a permutation of domain B that maximizes Pearson similarity between the two cosine-distance matrices. For domains with 8 concepts by default, V2 brute-forces every permutation. For larger domains, it switches to a restart-based greedy swap heuristic.

`outputs/alignments.json` records the readable concept mapping for each metaphor pair.

## V3 Dimensionality

Distance-shape similarity compares pairwise distances among concepts. Orientation-shape asks whether the aligned point clouds themselves can be rotated into similar positions. V3 measures this with orthogonal Procrustes alignment after PCA or classical MDS reduction.

Dimensionality curves show how much of the full optimal shape similarity is recovered using only `k` dimensions. `k90 shape recovery` is the smallest `k` that recovers at least 90% of the full distance-shape similarity. Low k90 suggests a compact structural metaphor; high k90 suggests the shared pattern is more distributed.

Participation ratio and spectral entropy dimension estimate how many PCA components effectively carry the domain's variance. They are soft effective-dimensionality measures, not literal ontology.

Semantic plausibility scoring was added because a mathematically strong permutation can still pair concepts in ways that look suspect. V3 combines label similarity, description similarity, and role matching. Optional `--llm-judge` can call a local OpenAI-compatible endpoint, but no LLM judge is required.

## Controls

V2 uses two control families:

- Domain controls compare the actual pair against random mismatched domain pairs.
- Permutation controls keep the actual pair fixed but randomly shuffle domain B many times.

Domain controls ask whether the metaphor pair is stronger than unrelated domain pairings. Permutation controls ask whether the optimal alignment is stronger than arbitrary alignments of the same two domains.

Z-scores measure how far the actual score is from the control mean in control standard deviations. Percentiles show how much of the control distribution the actual score beats. `p_estimate` is the fraction of control scores greater than or equal to the actual score.

## Outputs

The run writes:

- `outputs/results.csv`: ordered metrics, optimal metrics, control statistics, and bootstrap confidence intervals.
- `outputs/alignments.json`: best concept mappings and aligned label orders.
- `outputs/summary.json`: V2 aggregate fields and strongest/weakest pair records.
- `outputs/dimensionality_curves.csv`
- `outputs/domain_pca_spectra.csv`
- `outputs/domain_dimensionality_summary.csv`
- `outputs/component_similarity.csv`
- `outputs/alignment_plausibility.csv`
- `outputs/plots/actual_vs_random_similarity.png`
- `outputs/plots/pair_z_scores.png`
- `outputs/plots/distance_matrix_examples.png`
- `outputs/plots/shape_similarity_heatmap.png`
- `outputs/plots/ordered_vs_optimal_similarity.png`
- `outputs/plots/alignment_improvements.png`
- `outputs/plots/permutation_control_examples.png`
- `outputs/plots/dimensionality_recovery_curves.png`
- `outputs/plots/procrustes_orientation_curves.png`
- `outputs/plots/domain_pca_spectra.png`
- `outputs/plots/effective_dimensionality.png`
- `outputs/plots/k90_shape_recovery.png`
- `outputs/plots/plausibility_vs_geometry.png`
- `outputs/plots/component_similarity_examples.png`

## Plot Guide

`actual_vs_random_similarity.png` shows the random domain-control distribution with ordered and optimal mean metaphor similarities marked.

`ordered_vs_optimal_similarity.png` shows whether alignment search improves pair scores above the diagonal.

`alignment_improvements.png` ranks pairs by optimal-minus-ordered similarity.

`permutation_control_examples.png` shows the random permutation distribution for the strongest pair, with the optimal score marked.

`distance_matrix_examples.png` lets you visually compare a source domain, the paired domain in listed order, and the paired domain under the optimal alignment.

`dimensionality_recovery_curves.png` and the per-pair files under `outputs/plots/dimensionality_curves/` show how quickly reduced dimensions recover full shape similarity.

`procrustes_orientation_curves.png` shows whether reduced point clouds can be rotated into similar orientations.

`plausibility_vs_geometry.png` shows whether high geometry scores also correspond to plausible concept mappings.

## Interpretation

High optimal similarity with high control percentiles supports the limited claim that the selected metaphor pair preserves more embedding-space relational structure than the selected controls.

Low or mixed similarity is also useful. It may mean the embedding model does not encode the relevant relation, the chosen concepts are weak, the domain ordering or alignment is wrong, metaphor depends on richer context than static embeddings capture, or the hypothesis is weaker than expected.

Positive results do not prove that metaphor is fundamentally geometric, nor do they prove Recursive Resonance or metaphysical claims. They only show that selected metaphor-paired domains preserve relational shape in this embedding model better than selected controls.

## Limitations

The concept sets are curated and small. Optimal permutation search can overfit small matrices, which is why permutation controls and plausibility scoring matter. Results are model-dependent and should be treated as exploratory measurements rather than broad conclusions about language, cognition, or reality.

Old V1-style YAML files with `terms:` are still accepted and automatically converted into concept objects.
