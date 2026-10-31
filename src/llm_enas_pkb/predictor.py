"""Regression-based LLM surrogate model (Sec. 3.2.1).

Ties the architecture encoder, the population knowledge base and the DeepSeek
client together: given a set of labelled PKB exemplars and unlabelled candidate
architectures, it builds the regression prompt (Fig. 2), calls the LLM, and
returns a normalised predicted performance value per candidate.

In ``offline`` mode no API call is made; a deterministic k-NN surrogate (matching
edge operations in the PKB) produces the predictions instead. Input validation
and prompt construction are shared; API response parsing is only used online.
"""

from __future__ import annotations

import logging

import numpy as np

from .categorical import categorical_similarity_matrix, validate_architecture_vectors
from .config import Config
from .encoder import ArchitectureInformationEncoder
from .llm import DeepSeekClient, DeepSeekError, _extract_json_object
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
        self.encoder = encoder or ArchitectureInformationEncoder()
        # In offline mode there is no API client; the k-NN surrogate is used instead.
        self.client = None if config.offline else DeepSeekClient(config)

    # ------------------------------------------------------------- core API
    def predict_values(
        self,
        pkb_vectors: np.ndarray,
        pkb_values: np.ndarray,
        candidate_vectors: np.ndarray,
        *,
        generation: int | None = None,
    ) -> np.ndarray:
        """Predict a value in [0, 1] for each candidate, given PKB exemplars.

        When the LLM returns a malformed response (unparseable JSON, missing
        ``Value`` key, or the wrong number of predictions), the candidate count is
        re-stated and the request re-issued instead of failing the whole search;
        after the retries are exhausted the deterministic k-NN surrogate is used as
        a last-resort fallback so a stubborn model can never abort the run.
        """
        candidate_vectors = validate_architecture_vectors(candidate_vectors)
        n = len(candidate_vectors)
        if n == 0:
            return np.array([], dtype=np.float64)

        # Build the prompt (Fig. 2) from the labelled PKB + unlabelled candidates.
        prompt = build_regression_prompt(
            pkb_vectors,
            pkb_values,
            candidate_vectors,
            dataset=self.config.dataset,
            generation=generation,
        )
        if self.config.offline:
            preds = self._offline_predict(pkb_vectors, pkb_values, candidate_vectors)
            return self._validate(preds, n)

        # Online: ask the LLM; on a malformed answer, append the bad response plus
        # a corrective instruction and re-ask (multi-turn) rather than crashing.
        messages = [{"role": "user", "content": prompt}]
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            content = self.client.chat(messages)
            try:
                preds = self._parse_predictions(content, n)
            except (ValueError, DeepSeekError) as exc:
                last_err = exc
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {"role": "user", "content": self._correction_prompt(n, str(exc))}
                )
                logger.warning(
                    "LLM response rejected (attempt %d/%d): %s; re-asking",
                    attempt + 1, self.config.max_retries + 1, exc,
                )
                continue
            return self._validate(preds, n)

        # Last resort: fall back to the deterministic k-NN surrogate so a
        # stubbornly malformed LLM response cannot abort the whole search.
        logger.error(
            "LLM could not return %d valid predictions after %d attempts (%s); "
            "falling back to offline k-NN surrogate for this generation",
            n, self.config.max_retries + 1, last_err,
        )
        preds = self._offline_predict(pkb_vectors, pkb_values, candidate_vectors)
        return self._validate(preds, n)

    def predict_archs(
        self,
        candidate_archs,
        pkb_archs,
        pkb_values,
        *,
        generation: int | None = None,
    ) -> np.ndarray:
        """Convenience wrapper: encode architectures, then ``predict_values``."""
        pkb_vecs = self.encoder.encode_batch(pkb_archs)
        cand_vecs = self.encoder.encode_batch(candidate_archs)
        return self.predict_values(
            pkb_vecs,
            np.asarray(pkb_values, dtype=np.float64),
            cand_vecs,
            generation=generation,
        )

    # -------------------------------------------------------------- helpers
    def _parse_predictions(self, content: str, n: int) -> np.ndarray:
        """Require exactly one finite accuracy per candidate in the JSON response."""
        obj = _extract_json_object(content)
        if not isinstance(obj, dict) or set(obj) != {"Value"}:
            raise ValueError("LLM response must be a JSON object with only the 'Value' key")
        vals = obj["Value"]
        if not isinstance(vals, list):
            raise ValueError("LLM 'Value' must be a list with one number per candidate")
        if len(vals) != n:
            raise ValueError(
                f"LLM returned {len(vals)} values but {n} candidates were requested"
            )
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vals):
            raise ValueError("LLM predictions must be numeric accuracy values")
        if any(not np.isfinite(v) or not 0.0 <= v <= 1.0 for v in vals):
            raise ValueError("LLM predictions must be finite accuracy values in [0, 1]")
        return np.asarray(vals, dtype=np.float64)

    @staticmethod
    def _validate(preds: np.ndarray, n: int) -> np.ndarray:
        """Sanitise predictions: replace NaN, check the count, clip to [0, 1]."""
        preds = np.nan_to_num(np.asarray(preds, dtype=np.float64), nan=0.0)
        if len(preds) != n:
            raise ValueError(f"Expected {n} predictions, got {len(preds)}")
        return np.clip(preds, 0.0, 1.0)

    @staticmethod
    def _correction_prompt(n: int, reason: str) -> str:
        """Corrective re-ask message used when the LLM's answer was rejected."""
        return (
            f"Your previous answer was rejected because: {reason} "
            f"There are exactly {n} candidate architectures in the list above. "
            f"Output exactly one JSON object {{\"Value\": [...]}} whose list contains "
            f"exactly {n} floats — one predicted value per candidate, in the same "
            f"order they were listed. Do not output anything besides this JSON object."
        )

    def _offline_predict(
        self,
        pkb_vectors: np.ndarray,
        pkb_values: np.ndarray,
        candidate_vectors: np.ndarray,
    ) -> np.ndarray:
        """Deterministic k-NN surrogate used in ``offline`` mode.

        Average the values of the K PKB exemplars with the most matching edge
        operations, weighted by the fraction of matching positions. If none
        match any position, use the complete PKB mean; an empty PKB yields zero.
        This needs no API call and gives a stable baseline for smoke tests.
        """
        pkb_values = np.asarray(pkb_values, dtype=np.float64)
        if pkb_values.ndim != 1 or len(pkb_values) != len(pkb_vectors):
            raise ValueError("Expected one performance value per PKB architecture")
        sim = self._cross_similarity(candidate_vectors, pkb_vectors)
        preds = np.zeros(len(candidate_vectors), dtype=np.float64)
        if len(pkb_values) == 0:
            return preds
        for i in range(len(candidate_vectors)):
            k = min(_OFFLINE_K, len(pkb_values))
            nn = np.argsort(-sim[i], kind="stable")[:k]
            w = sim[i][nn]
            total_weight = w.sum()
            preds[i] = (
                float(np.dot(w / total_weight, pkb_values[nn]))
                if total_weight > 0.0
                else float(pkb_values.mean())
            )
        return preds

    @staticmethod
    def _cross_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fraction of identical edge operations between two architecture sets."""
        return categorical_similarity_matrix(a, b)
