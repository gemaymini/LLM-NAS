"""Prompt construction for the regression-based LLM surrogate (Sec. 3.2.1, Fig. 2).

The regression prompt is composed of five parts:

1. task description,
2. reasoning-step constraints,
3. the population knowledge base (labelled exemplars),
4. the candidate architectures to predict,
5. output constraints (JSON format, value range, precision).
"""

from __future__ import annotations

import numpy as np

# Number of decimal places used when serialising an architecture feature vector.
_FEATURE_PRECISION = 4
# Number of decimal places used for ground-truth exemplar values.
_VALUE_PRECISION = 6


def _format_vector(vec: np.ndarray) -> str:
    """Format a feature vector as a compact JSON-style list, e.g. ``[0.1234, ...]``."""
    return "[" + ", ".join(f"{float(v):.{_FEATURE_PRECISION}f}" for v in vec) + "]"


# The fixed task description and step-by-step reasoning instructions.  Kept as a
# single block (rather than templated) because it never changes between queries;
# the PKB and candidate lists are appended dynamically in ``build_regression_prompt``.
_TASK_HEADER = """Regression Task:
You need to predict the target value (Value) for the given Candidate Architectures \
(Architecture Information) based on the Population Knowledge Base. This task involves \
inferring the behavior of a black-box function and understanding the non-linear \
relationships between the architecture's attribute vector and its corresponding Value. \
The Value is the classification accuracy on the validation set, a real number between 0 and 1 \
where larger is better.

Steps:
1. Analyze the Population Knowledge Base to uncover relationships between \
Architecture Information and Value.
2. Based on these insights, predict the approximate Value for each given Candidate \
Architecture.
3. Ensure every prediction follows the specified range and format.
"""


def build_regression_prompt(
    pkb_vectors: np.ndarray,
    pkb_values: np.ndarray,
    candidate_vectors: np.ndarray,
) -> str:
    """Build the full regression prompt (Fig. 2).

    Parameters
    ----------
    pkb_vectors: (N, 30) encoded PKB architecture vectors.
    pkb_values:  (N,)   ground-truth values of the PKB exemplars.
    candidate_vectors: (M, 30) encoded candidate architectures to predict.
    """
    lines: list[str] = [_TASK_HEADER]

    # Part 3: the population knowledge base — one "<Architecture Information>;
    # <Value>;" line per labelled exemplar.
    lines.append("Population Knowledge Base:")
    for vec, val in zip(pkb_vectors, pkb_values):
        lines.append(
            f"<Architecture Information>: {_format_vector(vec)}; "
            f"<Value>: {float(val):.{_VALUE_PRECISION}f};"
        )

    # Part 5: output constraints (valid JSON, value range, precision, exact shape).
    lines.append("")
    lines.append("Constraints:")
    # DeepSeek JSON mode requires the prompt to contain the literal word "json"
    # (lowercase) plus a sample of the expected JSON structure; without it the
    # API can return empty content.  Keep the lowercase keyword here.
    lines.append("1. Output must be valid json.")
    lines.append('   Example outputs: {"Value": [0.926452, 0.875612]}')
    lines.append("2. Value range: 0.00 <= Value <= 1.00.")
    lines.append("3. Precision: each Value should have at most six decimal places.")
    lines.append("")
    lines.append(
        "Output exactly one JSON object with a single key \"Value\" whose value is a "
        "list of floats — one predicted Value per candidate architecture, in the same "
        "order as the candidates listed below. Do not output anything besides this JSON."
    )
    lines.append("")

    # Part 4: the candidate architectures, each with an empty "<Value>: ;"
    # placeholder to be filled by the model's JSON output.
    lines.append("Candidate Architectures:")
    for vec in candidate_vectors:
        lines.append(f"<Architecture Information>: {_format_vector(vec)}; <Value>: ;")

    return "\n".join(lines)
