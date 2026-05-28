from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Embedder:
    """Small cached wrapper around SentenceTransformer.encode."""

    model_name: str
    _model: object | None = field(default=None, init=False, repr=False)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}'. If you need "
                "offline runtime, make sure the model is already cached locally or "
                "choose a cached model with `--model`."
            ) from exc

        return self._model

    def embed_terms(self, terms: list[str]) -> np.ndarray:
        missing = [term for term in terms if term not in self._cache]
        if missing:
            model = self._load_model()
            vectors = model.encode(
                missing,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            for term, vector in zip(missing, vectors, strict=True):
                self._cache[term] = np.asarray(vector, dtype=float)

        return np.vstack([self._cache[term] for term in terms])
