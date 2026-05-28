# Codex Build Spec: Metaphor Geometry / Relational Shape Similarity Experiment

## Project Name

`recursive-resonance-semantic-geometry-lab`

## Goal

Build a Python experiment that tests whether metaphorically compatible domains preserve similar relational geometry in embedding space.

The central hypothesis:

> If two domains can meaningfully serve as metaphors for one another, then the internal relationship-shape among their concepts should be more similar than random unrelated concept groups.

Example:

* electrical domain: current, voltage, resistance, power
* fluid domain: flow, pressure, restriction, work

These are not the same words, but their relational structures may have similar shape.

The experiment should quantify that.

## Core Idea

Each conceptual domain is represented as a small ordered list of terms.

For each domain:

1. Embed each term or phrase into a vector using a local embedding model.
2. Compute the pairwise distance matrix between all terms.
3. Treat that distance matrix as the “relational shape” of the domain.
4. Compare the shape of one domain against another using matrix similarity metrics.
5. Compare metaphor-paired domains against random mismatched domains.

If metaphor-paired domains show consistently higher shape similarity than random pairs, that is evidence that metaphor is partly structure-preserving in embedding geometry.

## Requirements

Use Python 3.11+.

Use a normal project layout:

* `README.md`
* `requirements.txt`
* `data/domain_sets.yaml`
* `src/main.py`
* `src/embedder.py`
* `src/geometry.py`
* `src/plots.py`
* `outputs/`
* `outputs/results.csv`
* `outputs/summary.json`
* `outputs/plots/`

The program should be runnable from the project root with:

`python -m src.main`

Do not require internet at runtime if the embedding model is already cached locally.

## Preferred Libraries

Use:

* `sentence-transformers`
* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `pyyaml`

Default embedding model:

`sentence-transformers/all-mpnet-base-v2`

Allow model override by command-line argument.

Example:

`python -m src.main --model sentence-transformers/all-MiniLM-L6-v2`

## Data File

Create `data/domain_sets.yaml`.

It should contain manually curated domain groups and intended metaphor pairings.

Use this structure:

```yaml
domains:
  electrical_flow:
    terms:
      - current
      - voltage
      - resistance
      - power
      - circuit
      - ground
      - capacitance
      - impedance

  fluid_flow:
    terms:
      - flow
      - pressure
      - restriction
      - work
      - pipe
      - reservoir
      - storage
      - turbulence

metaphor_pairs:
  - [electrical_flow, fluid_flow]
```

Include at least these initial domain groups:

1. electrical_flow
2. fluid_flow
3. social_pressure
4. mechanical_force
5. music_harmony
6. color_harmony
7. emotional_valence
8. thermal_flow
9. economic_exchange
10. ecological_balance
11. signal_processing
12. linguistic_meaning

Each domain should have 6–10 terms.

Include at least 10 intended metaphor pairs, such as:

* electrical_flow ↔ fluid_flow
* social_pressure ↔ mechanical_force
* music_harmony ↔ color_harmony
* thermal_flow ↔ economic_exchange
* ecological_balance ↔ economic_exchange
* signal_processing ↔ linguistic_meaning
* emotional_valence ↔ color_harmony
* mechanical_force ↔ social_pressure
* fluid_flow ↔ economic_exchange
* signal_processing ↔ electrical_flow

The exact terms can be adjusted, but keep the term count per paired domain equal, because ordered correspondence matters.

## Important Detail: Ordered Correspondence

For this first version, assume each paired domain has terms in corresponding order.

Example:

```yaml
electrical_flow:
  terms:
    - current
    - voltage
    - resistance
    - power

fluid_flow:
  terms:
    - flow
    - pressure
    - restriction
    - work
```

That means:

* current corresponds to flow
* voltage corresponds to pressure
* resistance corresponds to restriction
* power corresponds to work

This lets us compare shape directly.

Later we can add optimal alignment / permutation search, but not in version 1 unless easy.

## Geometry Metrics

For each domain, compute:

1. Cosine similarity matrix
2. Cosine distance matrix, where distance = 1 - cosine_similarity
3. Euclidean distance matrix on normalized embeddings

For comparing two domain shapes, compute:

1. Pearson correlation between flattened upper triangles of the distance matrices
2. Spearman correlation between flattened upper triangles
3. Frobenius norm difference after normalizing each distance matrix
4. Procrustes disparity if practical

The main score should be:

`shape_similarity = Pearson correlation of upper-triangle cosine-distance matrices`

Higher is better.

## Controls

For every intended metaphor pair:

1. Compute actual pair shape similarity.
2. Compare against random mismatched domain pairs.
3. Run at least 1,000 random comparisons by default.
4. Compute percentile rank of the actual metaphor pair against random controls.
5. Compute z-score against random controls.

Example output:

```text
electrical_flow ↔ fluid_flow
shape_similarity: 0.82
random_mean: 0.31
random_std: 0.18
z_score: 2.83
percentile: 97.4
```

## Outputs

Write `outputs/results.csv` with one row per metaphor pair.

Columns:

* domain_a
* domain_b
* n_terms
* shape_similarity_pearson
* shape_similarity_spearman
* frobenius_difference
* random_mean
* random_std
* z_score
* percentile
* p_estimate

Where:

`p_estimate = fraction of random pairs with similarity >= actual similarity`

Write `outputs/summary.json` with:

* model used
* number of domains
* number of metaphor pairs
* number of random trials
* mean actual similarity
* mean random similarity
* mean z_score
* count of metaphor pairs above 90th percentile
* count above 95th percentile
* count above 99th percentile

## Plots

Create plots in `outputs/plots/`.

Required plots:

1. `actual_vs_random_similarity.png`

   * Histogram of random similarities
   * Vertical line for mean actual metaphor-pair similarity

2. `pair_z_scores.png`

   * Bar chart of z-scores for each metaphor pair

3. `distance_matrix_examples.png`

   * Side-by-side heatmaps for at least two paired domains, preferably electrical_flow and fluid_flow

4. `shape_similarity_heatmap.png`

   * Heatmap of shape similarity between all domains

Use matplotlib only.

Do not use seaborn.

## CLI Arguments

Support:

* `--model`
* `--data`
* `--outdir`
* `--random-trials`
* `--seed`

Example:

`python -m src.main --random-trials 5000 --seed 42`

## Reproducibility

Use deterministic random seeds.

Store the random seed in `summary.json`.

Cache embeddings during a run so the same domain terms are not embedded repeatedly.

## Error Handling

If paired domains have unequal term counts, print a clear warning and skip that pair for ordered comparison.

If a model fails to load, print a useful error explaining what happened.

If the data file is missing or malformed, fail clearly.

## README Requirements

Write a README explaining:

1. What the experiment tests
2. How to install dependencies
3. How to run the experiment
4. How to interpret results
5. Limitations

The README should explicitly state that positive results do not prove metaphysical claims. They only show that metaphorically paired domains preserve more embedding-space relational structure than random controls.

## Interpretation Notes

The experiment is successful if it can produce:

* numeric shape similarity scores
* random-control comparisons
* visual plots
* a clean results table

Even a negative or mixed result is useful.

Possible interpretations:

* High metaphor-pair similarity supports the idea that metaphors preserve relational geometry.
* Low similarity may mean:

  * the embedding model does not encode the relevant relation,
  * the chosen terms are bad,
  * the domain ordering is wrong,
  * metaphor works through higher-order context not captured by static embeddings,
  * or the hypothesis is weaker than expected.

## Future Extensions

Do not implement these unless the basic version is complete and clean:

1. Permutation search to find best concept alignment between two domains.
2. Phrase-based concepts instead of single words.
3. Multiple embedding models for comparison.
4. LLM-generated domain term sets.
5. Hidden-state trajectory comparison from llama.cpp or transformers.
6. Tensor-style comparison using triples or higher-order relation objects.
7. Graph-based comparison instead of matrix comparison.
8. Interactive dashboard.

## Definition of Done

The project is done when:

1. `python -m src.main` runs successfully.
2. `outputs/results.csv` is created.
3. `outputs/summary.json` is created.
4. All required plots are created.
5. README explains the experiment and limitations.
6. Results are reproducible with the same seed.
7. Code is organized and readable enough for further experiments.
