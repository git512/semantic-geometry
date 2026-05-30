# Codex Build Spec: Semantic Geometry Lab V3

## Project Context

This is V3 of `recursive-resonance-semantic-geometry-lab`.

V1 tested ordered relational-shape similarity between metaphor-paired domains.

V2 added phrase-based concepts, optimal permutation alignment, stronger random controls, bootstrap confidence intervals, and rich terminal progress.

V2 deep run produced strong results:

* Mean ordered similarity around 0.206
* Mean optimal similarity around 0.768
* Mean random domain similarity around 0.068
* Mean random permutation similarity near zero
* All 10 metaphor pairs above 99th percentile against both random-domain and random-permutation controls

However, V2 mainly tested distance-matrix shape. It showed that metaphor-paired domains can be permuted into similar relational geometries, but it did not answer:

1. How many latent dimensions carry the shared shape?
2. Whether the domains share actual orientation structure after rotation/alignment, not merely pairwise distances.
3. Whether optimized alignments are semantically plausible or merely mathematically convenient.
4. Whether the effect survives alternative embedding models and embedding text modes.

V3 should answer those.

## Main Goal

Extend the experiment to measure the dimensional anatomy of metaphorical shape similarity.

Core V3 hypothesis:

> Metaphorically compatible domains preserve a recoverable multidimensional pattern of relational orientations. When properly aligned, their shared structure should concentrate in a measurable number of latent dimensions and outperform random controls.

V3 should preserve all V2 functionality and add:

1. PCA / MDS dimensional analysis
2. Procrustes orientation alignment
3. Shape preservation curves over dimension count
4. Intrinsic dimensionality estimates
5. Component-wise similarity measures
6. Semantic plausibility scoring for alignments
7. Optional multi-model comparison
8. Stronger reports and plots

## Required New Features

### 1. Preserve V2 Completely

Do not remove V2 features.

Keep:

* phrase-based concept YAML support
* ordered comparison
* optimal permutation alignment
* random domain controls
* random permutation controls
* bootstrap CIs
* rich progress output
* all V2 plots
* backward compatibility with V1 YAML

V3 should build on V2, not replace it.

Add `version: 3` to summary output.

## 2. Dimensional Shape Analysis

For each intended metaphor pair, after the best V2 optimal alignment is found:

1. Extract aligned embedding matrices:

   * A: domain A embeddings
   * B: domain B embeddings reordered according to optimal mapping

2. Center both point clouds independently.

3. Analyze their geometry across dimension counts:

For k from 1 to `n_terms - 1`:

* Reduce A and B to k dimensions.
* Use PCA by default.
* Optionally support classical MDS.

Add CLI argument:

`--reduction pca`

or:

`--reduction mds`

Default:

`pca`

For each k, compute:

* reconstruction quality for A
* reconstruction quality for B
* Procrustes disparity between A_k and B_k
* orientation similarity score
* distance-shape similarity reconstructed at k dimensions
* fraction of full optimal similarity recovered at k

Output this as:

`outputs/dimensionality_curves.csv`

Columns:

* domain_a
* domain_b
* k
* reduction_method
* a_variance_explained
* b_variance_explained
* mean_variance_explained
* procrustes_disparity
* orientation_similarity
* reconstructed_distance_similarity
* fraction_of_full_similarity_recovered

## 3. PCA Variance Spectrum

For each domain independently, compute PCA variance spectrum.

Output:

`outputs/domain_pca_spectra.csv`

Columns:

* domain
* component
* explained_variance_ratio
* cumulative_explained_variance_ratio

Also compute effective dimensionality thresholds:

* dimensions needed for 50% variance
* dimensions needed for 75% variance
* dimensions needed for 90% variance
* dimensions needed for 95% variance

Add to:

`outputs/domain_dimensionality_summary.csv`

Columns:

* domain
* n_terms
* dim_50
* dim_75
* dim_90
* dim_95
* participation_ratio
* spectral_entropy

## 4. Intrinsic Dimensionality Estimate

For each domain and each aligned metaphor pair, estimate intrinsic dimensionality.

Use at least:

1. Participation ratio from PCA eigenvalues:

`PR = (sum(lambda)^2) / sum(lambda^2)`

2. Spectral entropy effective dimension:

`exp(entropy(normalized eigenvalues))`

Optional if easy:

3. Two-nearest-neighbor intrinsic dimensionality estimator.

Output these in domain and pair summaries.

For each metaphor pair, include:

* domain_a_participation_ratio
* domain_b_participation_ratio
* pair_mean_participation_ratio
* domain_a_spectral_entropy_dim
* domain_b_spectral_entropy_dim
* pair_mean_spectral_entropy_dim

Add these columns to `results.csv`.

## 5. Procrustes Orientation Alignment

For each metaphor pair after optimal concept permutation:

Run Procrustes alignment for k = 1 to n_terms - 1.

Use scipy if practical:

`scipy.spatial.procrustes`

Or implement manually:

1. Center both matrices.
2. Scale to unit Frobenius norm.
3. Solve orthogonal Procrustes with SVD.
4. Compute residual error/disparity.

Define:

`orientation_similarity = 1 - normalized_procrustes_disparity`

Clamp or normalize reasonably so higher is better.

Add to `results.csv`:

* best_k_by_orientation
* best_orientation_similarity
* best_procrustes_disparity
* k_90_similarity
* k_95_similarity

Where:

* `best_k_by_orientation` is the k with best orientation similarity.
* `k_90_similarity` is the smallest k recovering at least 90% of best orientation similarity.
* `k_95_similarity` is the smallest k recovering at least 95% of best orientation similarity.

## 6. Distance-Shape Recovery Curve

For each k:

1. Reduce A and B to k dims.
2. Recompute pairwise distance matrices in reduced space.
3. Compare those reduced distance matrices.
4. Compare to full-space optimal similarity.

This tells how many dimensions are needed to recover the V2 shape similarity.

Add to `results.csv`:

* k_80_shape_recovery
* k_90_shape_recovery
* k_95_shape_recovery
* best_k_by_shape_recovery

Interpretation:

If k_90_shape_recovery is low, the metaphor is structurally low-dimensional.

If k_90_shape_recovery is high, the metaphor depends on richer distributed geometry.

## 7. Component-Wise Similarity

After PCA reduction and Procrustes alignment, compute similarity by component.

For each metaphor pair and k:

* compare aligned component axes
* output component correlation / cosine similarity where meaningful

Write:

`outputs/component_similarity.csv`

Columns:

* domain_a
* domain_b
* component
* component_similarity
* a_explained_variance_ratio
* b_explained_variance_ratio

This should help answer whether the metaphor is carried by:

* one dominant axis
* a few strong axes
* many weak axes

## 8. Semantic Plausibility Scoring for Alignments

V2 optimal alignment sometimes finds mathematically strong but semantically questionable mappings.

V3 should add a second scoring layer to estimate whether each concept mapping is semantically plausible.

For each mapped concept pair:

Example:

`electrical resistance -> hydraulic work rate`

Compute plausibility using multiple cheap signals:

### Signal A: Label embedding similarity

Cosine similarity between the two concept label embeddings.

### Signal B: Description embedding similarity

If descriptions exist, cosine similarity between descriptions.

### Signal C: Role-word overlap / role category match

Add optional concept field in YAML:

```yaml
role: flow
```

Possible roles:

* flow
* gradient
* resistance
* power
* path
* reference
* storage
* instability
* context
* boundary
* transformation
* signal
* noise
* memory
* resolution
* tension
* unknown

If both concepts have the same role, add a plausibility boost.

If roles conflict sharply, penalize.

Update `data/domain_sets.yaml` to add `role` for every concept.

### Signal D: Optional LLM judge mode

Add optional CLI flag:

`--llm-judge`

If enabled, call an OpenAI-compatible local endpoint to score each mapping from 0 to 1.

Default endpoint:

`http://localhost:19090/v1/chat/completions`

Allow CLI args:

* `--llm-endpoint`
* `--llm-model`
* `--llm-api-key`

The LLM judge prompt should be short and deterministic.

It should ask:

> Does concept A play a structurally similar role to concept B in their respective domains? Score 0.0 to 1.0. Return JSON only.

Default: `--llm-judge` off.

V3 must not require an LLM judge to run.

## 9. Combined Alignment Score

Compute:

* geometry_score = optimal Pearson similarity
* plausibility_score = average mapping plausibility
* combined_score = weighted combination

Default:

`combined_score = 0.75 * geometry_score + 0.25 * plausibility_score`

Add CLI args:

* `--geometry-weight 0.75`
* `--plausibility-weight 0.25`

In V3, do not replace the V2 optimal alignment yet. Instead, report both:

1. geometry-optimal alignment
2. plausibility-adjusted alignment if implemented

If feasible, add a second alignment search objective:

`score = geometry_weight * shape_similarity + plausibility_weight * mapping_plausibility`

CLI:

`--alignment-objective geometry`

or:

`--alignment-objective combined`

Default:

`geometry`

If implementing combined search is too much, compute plausibility after the geometry-optimal alignment only.

## 10. Alignment Plausibility Output

Create:

`outputs/alignment_plausibility.csv`

Columns:

* domain_a
* domain_b
* concept_a_id
* concept_a_label
* concept_a_role
* concept_b_id
* concept_b_label
* concept_b_role
* label_similarity
* description_similarity
* role_match_score
* llm_judge_score
* final_plausibility_score

Update `outputs/alignments.json` to include:

* average_plausibility_score
* individual mapping plausibility details
* combined_score

## 11. Multi-Model Comparison

Add optional support for running multiple embedding models.

CLI:

`--models sentence-transformers/all-mpnet-base-v2 sentence-transformers/all-MiniLM-L6-v2`

If multiple models are provided:

* run the full experiment per model
* save outputs under:

`outputs/by_model/<safe_model_name>/`

* create aggregate comparison files:

`outputs/model_comparison.csv`
`outputs/model_comparison_summary.json`

At minimum compare:

* mean ordered similarity
* mean optimal similarity
* mean random domain similarity
* mean random permutation similarity
* mean k_90_shape_recovery
* mean best_orientation_similarity
* mean plausibility_score

Default behavior with single `--model` should remain simple.

## 12. Label Plus Description Mode

V2 supported:

* label_only
* label_plus_description

V3 should actively compare both if requested.

Add CLI:

`--compare-embed-text-modes`

If enabled, run:

1. label_only
2. label_plus_description

Output:

`outputs/embed_text_mode_comparison.csv`

Columns:

* embed_text_mode
* mean_ordered_similarity
* mean_optimal_similarity
* mean_alignment_improvement
* mean_plausibility_score
* mean_k_90_shape_recovery
* mean_best_orientation_similarity

## 13. New V3 Plots

Keep all V2 plots.

Add the following:

### A. Dimensionality Recovery Curves

`outputs/plots/dimensionality_recovery_curves.png`

For each metaphor pair, plot:

* x-axis: k dimensions
* y-axis: fraction of full similarity recovered

If 10 lines are too cluttered, also create individual plots:

`outputs/plots/dimensionality_curves/<domain_a>__<domain_b>.png`

### B. Procrustes Orientation Curves

`outputs/plots/procrustes_orientation_curves.png`

x-axis: k dimensions
y-axis: orientation similarity

### C. PCA Spectra Plot

`outputs/plots/domain_pca_spectra.png`

Plot cumulative explained variance per domain.

### D. Effective Dimensionality Bar Chart

`outputs/plots/effective_dimensionality.png`

Bar chart of participation ratio and spectral entropy dimension for each domain.

### E. k90 Shape Recovery Bar Chart

`outputs/plots/k90_shape_recovery.png`

For each metaphor pair, show the smallest k needed to recover 90% of full shape similarity.

### F. Plausibility vs Geometry Scatter

`outputs/plots/plausibility_vs_geometry.png`

x-axis: average plausibility score
y-axis: optimal Pearson geometry score

This reveals whether the strongest geometric mappings are also semantically plausible.

### G. Component Similarity Plot

`outputs/plots/component_similarity_examples.png`

For strongest pair and weakest pair, show component similarity by component.

## 14. Rich Terminal Output

Enhance rich progress output.

During V3, show phases:

1. Loading data
2. Loading embedding model(s)
3. Embedding concepts
4. Computing V2 shape metrics
5. Optimal alignment search
6. Dimensionality analysis
7. Procrustes orientation analysis
8. Intrinsic dimensionality estimation
9. Plausibility scoring
10. Random controls
11. Bootstrap CIs
12. Writing outputs
13. Rendering plots

At the end, print rich tables:

### Table 1: V2 Geometry Summary

* pair
* ordered similarity
* optimal similarity
* improvement
* z-score vs domain controls
* z-score vs permutation controls

### Table 2: V3 Dimensionality Summary

* pair
* best orientation similarity
* best k
* k90 shape recovery
* k95 shape recovery
* participation ratio mean
* spectral entropy dim mean

### Table 3: Plausibility Summary

* pair
* geometry score
* plausibility score
* combined score
* most suspicious mapping

“Most suspicious mapping” = lowest plausibility concept pair.

## 15. CLI Defaults

Default run should remain usable.

Defaults:

* `--random-trials 2000`
* `--permutation-trials 2000`
* `--bootstrap-samples 1000`
* `--max-bruteforce-n 8`
* `--alignment both`
* `--alignment-objective geometry`
* `--embed-text label_only`
* `--reduction pca`
* `--seed 42`
* `--llm-judge` off

Quick mode:

`--quick`

* random trials = 200
* permutation trials = 200
* bootstrap samples = 200
* skip multi-model unless explicitly requested
* skip LLM judge unless explicitly requested

Deep mode:

`--deep`

* random trials = 10000
* permutation trials = 10000
* bootstrap samples = 5000

Add:

`--v3-deep`

If set:

* deep mode
* compare label_only vs label_plus_description
* run dimensionality analysis
* run plausibility scoring
* do not enable LLM judge unless explicitly requested

## 16. Data Update: Add Roles

Update `data/domain_sets.yaml`.

Every concept should have:

* id
* label
* description
* role

Example:

```yaml
domains:
  electrical_flow:
    concepts:
      - id: current
        label: electrical current
        description: movement of electric charge through a conductor
        role: flow
      - id: voltage
        label: voltage difference
        description: electrical potential difference that drives current
        role: gradient
      - id: resistance
        label: electrical resistance
        description: opposition to current flow
        role: resistance
      - id: power
        label: electrical power
        description: rate of energy transfer in an electrical circuit
        role: power
      - id: circuit_path
        label: circuit path
        description: conductive route through which current can flow
        role: path
      - id: ground_reference
        label: electrical ground reference
        description: reference potential or return path in a circuit
        role: reference
      - id: capacitance_storage
        label: electrical capacitance
        description: ability to store charge and energy in an electric field
        role: storage
      - id: impedance_dynamic
        label: electrical impedance
        description: frequency-dependent opposition to alternating current
        role: resistance
```

Do this for all domains.

## 17. Updated Summary JSON

`outputs/summary.json` should include:

* version: 3
* all V2 summary fields
* reduction method
* mean_best_orientation_similarity
* mean_best_procrustes_disparity
* mean_k_80_shape_recovery
* mean_k_90_shape_recovery
* mean_k_95_shape_recovery
* mean_participation_ratio
* mean_spectral_entropy_dim
* mean_alignment_plausibility
* mean_combined_score
* strongest_pair_by_orientation
* strongest_pair_by_dimensional_efficiency
* richest_pair_by_dimensionality
* most_plausible_alignment
* least_plausible_alignment

Definitions:

* strongest_pair_by_orientation = highest best_orientation_similarity
* strongest_pair_by_dimensional_efficiency = lowest k_90_shape_recovery with high optimal similarity
* richest_pair_by_dimensionality = highest k_90_shape_recovery or highest participation ratio
* most_plausible_alignment = highest average plausibility
* least_plausible_alignment = lowest average plausibility

## 18. README Updates

Update README with a V3 section explaining:

1. Difference between distance-shape and orientation-shape
2. What Procrustes alignment measures
3. What PCA/MDS dimensionality curves mean
4. What k90 shape recovery means
5. What participation ratio and spectral entropy dimension mean
6. Why semantic plausibility scoring was added
7. Limits of optimal permutation search
8. Why positive results still do not prove Recursive Resonance or metaphysical claims
9. How to run quick/default/deep/V3-deep modes

Include this conceptual explanation:

> V2 asks whether two domains can be rearranged into similar relational distance-shapes. V3 asks how many dimensions carry that shared shape, whether orientation can be aligned, and whether the resulting concept mappings remain semantically plausible.

## 19. Definition of Done

V3 is complete when:

1. `python -m src.main --quick` runs successfully.
2. `python -m src.main` runs successfully.
3. `python -m src.main --v3-deep` runs successfully.
4. All V2 outputs are still produced.
5. New V3 outputs are produced:

   * `dimensionality_curves.csv`
   * `domain_pca_spectra.csv`
   * `domain_dimensionality_summary.csv`
   * `component_similarity.csv`
   * `alignment_plausibility.csv`
6. `results.csv` includes V3 dimensionality and plausibility columns.
7. `summary.json` includes V3 summary fields.
8. V3 plots are generated.
9. Rich terminal output shows V3 progress and final tables.
10. README is updated.
11. V1 YAML backward compatibility still works.
12. Results are reproducible with the same seed.

## 20. Implementation Priority

Priority order:

1. Preserve V2 behavior.
2. Add roles to data.
3. Add PCA spectra and intrinsic dimensionality.
4. Add Procrustes orientation curves.
5. Add k-dimensional shape recovery curves.
6. Add dimensionality CSV outputs.
7. Add semantic plausibility scoring without LLM judge.
8. Add V3 plots.
9. Add rich final summary tables.
10. Add optional LLM judge.
11. Add multi-model comparison.
12. Add embed-text-mode comparison.

Do not let optional features break the core experiment. If time runs short, prioritize dimensionality and orientation analysis over LLM judging or multi-model support.

