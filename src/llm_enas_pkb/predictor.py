"""Regression-based LLM surrogate model (Sec. 3.2.1).

Ties the architecture encoder, the population knowledge base and the DeepSeek
client together: given a set of labelled PKB exemplars and unlabelled candidate
architectures, it builds the regression prompt (Fig. 2), calls the LLM, and
returns a normalised predicted performance value per candidate.

In ``offline`` mode no API call is made; a deterministic k-NN surrogate (cosine
similarity over the PKB vectors) produces the predictions instead, exercising the
exact same prompt/parse/validate path so the full pipeline can be smoke-tested.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import Config
from .encoder import ArchitectureInformationEncoder
from .llm import DeepSeekClient, _extract_json_object
from .prompts import build_regression_prompt

logger = logging.getLogger(__name__)

_OFFLINE_K = 5  # number of nearest PKB exemplars used by the offline surrogate


class RegressionPredictor:
    def __init__(
        self,
        config: Config,
        encoder: ArchitectureInformationEncoder | None = None,
    ):
        self.config = config
        # Allow injecting a pre-built encoder (e.g. to share a consistent feature
        # space across components); otherwise construct the default one.
        self.encoder = encoder or ArchitectureInformationEncoder(
            depth=config.aggregation_depth,
            use_projection=config.use_projection,
            seed=config.seed,
        )
        # In offline mode there is no API client; the k-NN surrogate is used instead.
        self.client = None if config.offline else DeepSeekClient(config)

    # ------------------------------------------------------------- core API
    def predict_values(
        self,
        pkb_vectors: np.ndarray,
        pkb_values: np.ndarray,
        candidate_vectors: np.ndarray,
    ) -> np.ndarray:
        """Predict a value in [0, 1] for each candidate, given PKB exemplars."""
        candidate_vectors = np.asarray(candidate_vectors, dtype=np.float64)
        n = len(candidate_vectors)
        if n == 0:
            return np.array([], dtype=np.float64)

        # Build the prompt (Fig. 2) from the labelled PKB + unlabelled candidates.
        prompt = build_regression_prompt(pkb_vectors, pkb_values, candidate_vectors)
        if self.config.offline:
            preds = self._offline_predict(pkb_vectors, pkb_values, candidate_vectors)
        else:
            content = self.client.chat([{"role": "user", "content": prompt}])
            preds = self._parse_predictions(content, n)
        return self._validate(preds, n)

    def predict_archs(
        self,
        candidate_archs,
        pkb_archs,
        pkb_values,
    ) -> np.ndarray:
        """Convenience wrapper: encode architectures, then ``predict_values``."""
        pkb_vecs = self.encoder.encode_batch(pkb_archs)
        cand_vecs = self.encoder.encode_batch(candidate_archs)
        return self.predict_values(pkb_vecs, np.asarray(pkb_values, dtype=np.float64), cand_vecs)

    # -------------------------------------------------------------- helpers
    def _parse_predictions(self, content: str, n: int) -> np.ndarray:
        """Extract the ``Value`` list from the LLM's JSON response.

        Tolerates a scalar response for a single candidate, and broadcasts a
        single-element list to ``n`` candidates (some models collapse the list).
        """
        obj = _extract_json_object(content)
        if "Value" not in obj:
            raise ValueError(f"LLM response missing 'Value' key: {content[:200]!r}")
        vals = obj["Value"]
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        vals = [float(v) for v in vals]
        if len(vals) == 1 and n > 1:
            vals = vals * n
        if len(vals) != n:
            raise ValueError(
                f"LLM returned {len(vals)} values but {n} candidates were requested"
            )
        return np.asarray(vals, dtype=np.float64)

    @staticmethod
    def _validate(preds: np.ndarray, n: int) -> np.ndarray:
        """Sanitise predictions: replace NaN, check the count, clip to [0, 1]."""
        preds = np.nan_to_num(np.asarray(preds, dtype=np.float64), nan=0.0)
        if len(preds) != n:
            raise ValueError(f"Expected {n} predictions, got {len(preds)}")
        return np.clip(preds, 0.0, 1.0)

    def _offline_predict(
        self,
        pkb_vectors: np.ndarray,
        pkb_values: np.ndarray,
        candidate_vectors: np.ndarray,
    ) -> np.ndarray:
        """Deterministic k-NN surrogate used in ``offline`` mode.

        For each candidate, average the values of its K nearest PKB exemplars (by
        cosine similarity), weighted by the non-negative similarity.  This needs
        no API call and gives a stable baseline for smoke-testing the pipeline.
        """
        sim = self._cross_similarity(candidate_vectors, pkb_vectors)
        preds = np.zeros(len(candidate_vectors), dtype=np.float64)
        for i in range(len(candidate_vectors)):
            k = min(_OFFLINE_K, len(pkb_values))
            nn = np.argsort(-sim[i])[:k]  # indices of the K most similar exemplars
            w = np.maximum(sim[i][nn], 0.0)  # drop negative similarities
            w = w / (w.sum() + 1e-12)  # normalise to a convex combination
            preds[i] = float(np.dot(w, pkb_values[nn]))
        return preds

    @staticmethod
    def _cross_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Row-normalised cosine similarity matrix between two sets of vectors."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        return a @ b.T
