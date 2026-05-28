# Recursive Resonance Semantic Geometry Lab

This project tests whether metaphorically compatible concept domains preserve similar relational geometry in embedding space.

The central idea is simple: represent each domain as an ordered list of terms, embed those terms, compute the pairwise distance matrix inside each domain, and compare one domain's relational shape with another's. If metaphor-paired domains have higher shape similarity than random mismatched domains, that is evidence that the chosen embedding model encodes some structure-preserving aspect of metaphor.

## Install

Use Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

With `uv`:

```bash
uv venv --python /usr/bin/python3.11 .venv
uv pip install -r requirements.txt
```

The default model is `sentence-transformers/all-mpnet-base-v2`. Runtime does not require internet if the selected model is already cached locally by Sentence Transformers.

## Run

From the project root:

```bash
python -m src.main
```

If your environment does not provide a `python` alias, use `python3 -m src.main`.

Useful options:

```bash
python -m src.main --model sentence-transformers/all-MiniLM-L6-v2
python -m src.main --random-trials 5000 --seed 42
python -m src.main --data data/domain_sets.yaml --outdir outputs
```

## Outputs

The run writes:

- `outputs/results.csv`: one row per evaluated metaphor pair.
- `outputs/summary.json`: model, seed, aggregate scores, and percentile counts.
- `outputs/plots/actual_vs_random_similarity.png`
- `outputs/plots/pair_z_scores.png`
- `outputs/plots/distance_matrix_examples.png`
- `outputs/plots/shape_similarity_heatmap.png`

The main score is `shape_similarity_pearson`, the Pearson correlation between the upper triangles of two cosine-distance matrices. Higher values mean the two domains have more similar internal distance structure under the selected embedding model.

`percentile` reports where the actual metaphor pair lands against random controls. `p_estimate` is the fraction of random similarities greater than or equal to the actual similarity.

## Interpretation

High metaphor-pair similarity supports the limited claim that the paired domains preserve more embedding-space relational structure than random controls.

Low or mixed similarity is also informative. It may mean the embedding model does not encode the relevant relation, the chosen terms are weak, the ordered correspondence is wrong, metaphor depends on richer context than static embeddings capture, or the hypothesis is weaker than expected.

Positive results do not prove metaphysical claims. They only show that, for these curated terms and this embedding model, metaphorically paired domains preserve more relational geometry than the sampled random controls.

## Limitations

Version 1 assumes ordered correspondence: term 1 in one domain corresponds to term 1 in the paired domain, and so on. It does not search for the best permutation.

The domain sets are manually curated, small, and model-dependent. Results should be treated as exploratory measurements rather than broad conclusions about language, cognition, or reality.
