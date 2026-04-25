"""Vector embeddings module for GitMem0.

Provides semantic similarity search via sentence-transformers with
graceful fallback when the library is not installed.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Suppress transformers/sentence-transformers noise BEFORE any imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_DIM = 384

_sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer

    _sentence_transformers_available = True
except ImportError:
    logger.warning(
        "sentence-transformers not installed. "
        "Embeddings will return zero vectors. "
        "Install with: pip install sentence-transformers"
    )


class EmbeddingEngine:
    """Manages text-to-vector encoding with lazy model loading.

    When sentence-transformers is unavailable, all methods degrade to
    returning zero vectors of the expected dimension.
    """

    def __init__(self, model_name: Optional[str] = None, dimension: int = _DEFAULT_DIM) -> None:
        self._model_name = model_name or _DEFAULT_MODEL
        self._dimension = dimension
        self._model: Optional[SentenceTransformer] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazily load the sentence-transformers model on first use."""
        if self._loaded:
            return
        self._loaded = True  # mark early to avoid repeated failed attempts
        if not _sentence_transformers_available:
            return
        try:
            logger.info("Loading embedding model: %s", self._model_name)
            import os
            import sys
            # Suppress ALL output during model loading
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
            _devnull = open(os.devnull, "w")
            sys.stdout = _devnull
            sys.stderr = _devnull
            try:
                self._model = SentenceTransformer(self._model_name)
            finally:
                sys.stdout = _orig_stdout
                sys.stderr = _orig_stderr
                _devnull.close()
            try:
                self._dimension = self._model.get_embedding_dimension()
            except AttributeError:
                self._dimension = self._model.get_sentence_embedding_dimension()
        except Exception:
            logger.exception("Failed to load embedding model %s", self._model_name)
            self._model = None

    def is_available(self) -> bool:
        """Return True if the real sentence-transformers model is loaded."""
        self._ensure_loaded()
        return self._model is not None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _zero(self) -> np.ndarray:
        return np.zeros(self._dimension, dtype=np.float32)

    def embed(self, text: str) -> list[float]:
        """Encode a single text into a vector embedding."""
        self._ensure_loaded()
        if self._model is None:
            return self._zero().tolist()
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts into vector embeddings."""
        self._ensure_loaded()
        if self._model is None:
            return [self._zero().tolist() for _ in texts]
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32,
        )
        return [np.asarray(v, dtype=np.float32).tolist() for v in vecs]

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def most_similar(
        self,
        query_embedding: list[float],
        candidates: list[tuple[str, list[float]]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return the top_k most similar (id, score) pairs from candidates."""
        if not candidates:
            return []
        vq = np.asarray(query_embedding, dtype=np.float32)
        ids = [c[0] for c in candidates]
        matrix = np.asarray([c[1] for c in candidates], dtype=np.float32)

        # Batch cosine: (q . M^T) / (||q|| * ||M_row||)
        q_norm = np.linalg.norm(vq)
        m_norms = np.linalg.norm(matrix, axis=1)
        denom = q_norm * m_norms
        denom[denom == 0.0] = 1.0  # avoid division by zero; those scores will be 0 anyway
        scores = (matrix @ vq) / denom

        if top_k >= len(ids):
            order = np.argsort(-scores)
        else:
            order = np.argpartition(scores, -top_k)[-top_k:]
            order = order[np.argsort(-scores[order])]

        return [(ids[i], float(scores[i])) for i in order]
