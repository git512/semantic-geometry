# Codex Build Spec: Semantic Geometry Lab V2

## Project Context

This is V2 of the `recursive-resonance-semantic-geometry-lab`.

V1 successfully tested whether metaphorically compatible domains preserve similar relational geometry in embedding space by comparing ordered cosine-distance matrices.

V1 worked, but the results showed three likely limitations:

1. Single-word embeddings are too ambiguous.
2. Ordered term correspondence is too fragile.
3. The random-control baseline needs to be stronger.
4. The run needs better live visual/progress feedback.

V2 should preserve the V1 workflow while adding phrase-based concepts, optimal alignment/permutation search, better statistics, and richer live progress output.

## Main Goal

Improve the metaphor-geometry experiment so that it can test whether two domains preserve relational shape even when the initially listed term order is imperfect.

Core V2 hypothesis:

> Metaphorically compatible domains may preserve relational geometry even if their concept-to-concept alignment is not initially ordered correctly. Therefore, the experiment should search for the best shape-preserving alignment between domains and compare that against stronger random controls.

## Required New Features

### 1. Phrase-Based Domain Terms

Update `data/domain_sets.yaml` so each domain term can be a short phrase, not just a single word.

Each concept entry should support:

```yaml
- id: current
  label: electrical current
  description: movement of electric charge through a conductor
```

Required fields:

* `id`: short stable identifier
* `label`: phrase to embed
* `description`: optional text description

The embedding text should be configurable:

* `label_only`
* `label_plus_description`

Default should be `label_only`.

Add CLI argument:

`--embed-text label_only`

or:

`--embed-text label_plus_description`

If using `label_plus_description`, embed:

`"{label}: {description}"`

If description is missing, fall back to label.

### 2. Preserve V1 Ordered Comparison

Keep the V1 ordered comparison as a baseline.

For each intended metaphor pair, compute:

* ordered Pearson similarity
* ordered Spearman similarity
* ordered Frobenius difference

These should remain in `results.csv`.

### 3. Add Optimal Alignment / Permutation Search

For each metaphor pair, add an optimal alignment mode.

The goal is to find the concept permutation of domain B that maximizes shape similarity against domain A.

For small domains, brute force is acceptable.

If both domains have `n <= 9`, brute force all permutations of B.

If `n > 9`, use a heuristic search:

* start with random permutations
* use greedy swaps / hill climbing
* run multiple restarts
* keep the best permutation

Add CLI args:

* `--alignment ordered`
* `--alignment optimal`
* `--alignment both`

Default:

`--alignment both`

Add CLI args:

* `--max-bruteforce-n 9`
* `--alignment-restarts 200`
* `--alignment-steps 1000`

For each pair, output:

* best permutation
* best Pearson similarity
* best Spearman similarity
* best Frobenius difference
* improvement over ordered similarity

The best permutation should be written as readable mappings, for example:

```json
{
  "current": "flow_rate",
  "voltage": "pressure_difference",
  "resistance": "restriction",
  "power": "work_rate"
}
```

### 4. Add Alignment Output File

Create:

`outputs/alignments.json`

This should contain one object per metaphor pair.

Each object should include:

* domain_a
* domain_b
* ordered_similarity
* optimal_similarity
* improvement
* mapping
* domain_a_labels
* domain_b_labels_ordered
* domain_b_labels_optimal
* method_used: brute_force or heuristic

### 5. Stronger Random Controls

V1 random controls compared random domain pairs.

V2 should add two control types:

#### Control A: Random Domain Pair Controls

Same as V1.

#### Control B: Random Permutation Controls

For the actual metaphor pair:

* Keep domain A fixed.
* Randomly permute domain B many times.
* Compute shape similarity for each random permutation.
* Compare the optimal alignment against these random permutations.

This tests whether the optimal alignment is meaningfully better than arbitrary alignments of the same two domains.

Add output metrics:

* random_domain_mean

* random_domain_std

* random_domain_z_score

* random_domain_percentile

* random_domain_p_estimate

* random_permutation_mean

* random_permutation_std

* random_permutation_z_score

* random_permutation_percentile

* random_permutation_p_estimate

### 6. Bootstrap Confidence Intervals

Add bootstrapped confidence intervals for actual metaphor-pair similarity.

For each pair:

* Resample upper-triangle distance entries with replacement.
* Compute Pearson correlation for each bootstrap sample.
* Use 1,000 bootstrap samples by default.
* Report 95% CI.

Add CLI argument:

`--bootstrap-samples 1000`

Output:

* optimal_similarity_ci_low
* optimal_similarity_ci_high
* ordered_similarity_ci_low
* ordered_similarity_ci_high

### 7. More Robust `results.csv`

Update `outputs/results.csv` columns to include:

* domain_a
* domain_b
* n_terms
* ordered_pearson
* ordered_spearman
* ordered_frobenius
* optimal_pearson
* optimal_spearman
* optimal_frobenius
* alignment_improvement
* alignment_method
* random_domain_mean
* random_domain_std
* random_domain_z_score
* random_domain_percentile
* random_domain_p_estimate
* random_permutation_mean
* random_permutation_std
* random_permutation_z_score
* random_permutation_percentile
* random_permutation_p_estimate
* ordered_similarity_ci_low
* ordered_similarity_ci_high
* optimal_similarity_ci_low
* optimal_similarity_ci_high

### 8. Better Summary File

Update `outputs/summary.json`.

Include:

* version: 2
* model used
* embed_text mode
* seed
* number_of_domains
* number_of_metaphor_pairs
* number_of_random_domain_trials
* number_of_random_permutation_trials
* bootstrap_samples
* mean_ordered_similarity
* mean_optimal_similarity
* mean_alignment_improvement
* mean_random_domain_similarity
* mean_random_permutation_similarity
* count_optimal_above_90th_percentile_domain_controls
* count_optimal_above_95th_percentile_domain_controls
* count_optimal_above_99th_percentile_domain_controls
* count_optimal_above_90th_percentile_permutation_controls
* count_optimal_above_95th_percentile_permutation_controls
* count_optimal_above_99th_percentile_permutation_controls
* strongest_pair_by_optimal_similarity
* strongest_pair_by_z_score
* weakest_pair

### 9. Live Progress / Visual Feedback

The user specifically wants visible output during execution.

Add rich terminal progress using `rich`.

Add to `requirements.txt`:

`rich`

Use:

* progress bars
* spinners
* live status panels
* tables printed during/after major phases

Runtime should show phases like:

1. Loading data
2. Loading embedding model
3. Embedding concepts
4. Computing domain distance matrices
5. Ordered metaphor comparisons
6. Optimal alignment search
7. Random domain controls
8. Random permutation controls
9. Bootstrap confidence intervals
10. Writing outputs
11. Rendering plots

Use `rich.progress.Progress` with nested tasks where practical.

During optimal alignment, show:

* current pair
* method: brute force or heuristic
* best score so far
* percent complete if brute force
* restart number if heuristic

During random controls, show trial progress.

At the end, print a rich table with:

* pair
* ordered similarity
* optimal similarity
* improvement
* domain-control z-score
* permutation-control z-score
* percentile

Also print a short interpretation block:

* strongest pair
* weakest pair
* mean improvement
* how many pairs beat 95th percentile controls

### 10. Optional Live Plot Mode

Add optional live visual mode using matplotlib interactive updates if easy.

CLI argument:

`--live-plot`

If enabled, during random controls, show a live-updating histogram for current random similarity distribution with a vertical line for the actual/optimal score.

This feature is optional. Do not let it destabilize the core CLI. If implementing live plotting is messy, skip it and focus on rich terminal progress.

### 11. New Plots

Keep all V1 plots and add V2-specific plots.

Required plots:

1. `actual_vs_random_similarity.png`

   * Updated to show ordered and optimal mean actual similarities as separate vertical lines.

2. `pair_z_scores.png`

   * Show both domain-control and permutation-control z-scores if practical.

3. `distance_matrix_examples.png`

   * Same as V1, but include optimal-aligned version for the paired domain if possible.

4. `shape_similarity_heatmap.png`

   * Same as V1.

5. `ordered_vs_optimal_similarity.png`

   * Scatter plot:

     * x-axis: ordered similarity
     * y-axis: optimal similarity
     * diagonal line y=x

6. `alignment_improvements.png`

   * Bar chart of optimal minus ordered similarity by pair.

7. `permutation_control_examples.png`

   * For the strongest pair, histogram of random permutation similarities with optimal score marked.

Use matplotlib only.

Do not use seaborn.

### 12. Updated Data

Revise domain terms to use phrase labels and descriptions.

Keep the 12 V1 domains, but convert the terms into structured concept objects.

Use 6–8 concepts per domain.

Important: for paired domains, try to preserve an intended order, but V2 should no longer depend entirely on that order.

Example:

```yaml
domains:
  electrical_flow:
    concepts:
      - id: current
        label: electrical current
        description: movement of electric charge through a conductor
      - id: voltage
        label: voltage difference
        description: electrical potential difference that drives current
      - id: resistance
        label: electrical resistance
        description: opposition to current flow
      - id: power
        label: electrical power
        description: rate of energy transfer in an electrical circuit
      - id: circuit_path
        label: circuit path
        description: conductive route through which current can flow
      - id: ground_reference
        label: electrical ground reference
        description: reference potential or return path in a circuit
      - id: capacitance_storage
        label: electrical capacitance
        description: ability to store charge and energy in an electric field
      - id: impedance_dynamic
        label: electrical impedance
        description: frequency-dependent opposition to alternating current
```

Create similarly specific phrase-based concepts for:

* fluid_flow
* social_pressure
* mechanical_force
* music_harmony
* color_harmony
* emotional_valence
* thermal_flow
* economic_exchange
* ecological_balance
* signal_processing
* linguistic_meaning

### 13. CLI Defaults

Default run should be reasonable and not too slow.

Defaults:

* `--random-trials 2000`
* `--permutation-trials 2000`
* `--bootstrap-samples 1000`
* `--max-bruteforce-n 8`
* `--alignment both`
* `--embed-text label_only`
* `--seed 42`

Add `--quick` mode:

If `--quick` is set:

* random trials = 200
* permutation trials = 200
* bootstrap samples = 200
* heuristic restarts = 25
* heuristic steps = 200

Add `--deep` mode:

If `--deep` is set:

* random trials = 10000
* permutation trials = 10000
* bootstrap samples = 5000
* heuristic restarts = 1000
* heuristic steps = 5000

### 14. Backward Compatibility

If the old V1 YAML format is encountered:

```yaml
terms:
  - current
  - voltage
```

Automatically convert each term to:

```yaml
id: current
label: current
description: ""
```

This lets old files still run.

### 15. README Updates

Update README with:

1. V2 explanation
2. What optimal alignment means
3. Why phrase-based embeddings matter
4. Difference between domain controls and permutation controls
5. How to interpret z-scores and percentiles
6. How to run quick/default/deep modes
7. How to read the output plots
8. Limitations

Explicitly include this warning:

> Positive results do not prove that metaphor is fundamentally geometric, nor do they prove Recursive Resonance. They only show that selected metaphor-paired domains preserve relational shape in this embedding model better than selected controls.

### 16. Definition of Done

V2 is complete when:

1. `python -m src.main --quick` runs successfully.
2. `python -m src.main` runs successfully.
3. Rich progress bars/status panels are visible during execution.
4. `outputs/results.csv` includes ordered and optimal metrics.
5. `outputs/alignments.json` is created.
6. `outputs/summary.json` includes V2 summary fields.
7. All V2 plots are created.
8. README explains the experiment clearly.
9. Old V1-style YAML still works.
10. Results are reproducible with the same seed.

## Implementation Priority

Priority order:

1. Structured phrase data support
2. Ordered comparison preserved
3. Optimal alignment search
4. Random permutation controls
5. Rich terminal progress
6. Updated CSV/JSON outputs
7. Updated plots
8. Bootstrap confidence intervals
9. Optional live plot mode

Do not overcomplicate the architecture. Keep code readable and experiment-focused.

