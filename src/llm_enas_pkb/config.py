"""Configuration for LLM-ENAS-PKB (NAS-Bench-201 regression surrogate).

The defaults mirror Table 1 of the paper, restricted to the NAS-Bench-201 /
DARTS column (``d = 30`` input dimension).  Values may be overridden either by
constructing ``Config`` directly or via environment variables with an
``LLMENAS_`` prefix (``LLMENAS_MODEL``, ``LLMENAS_DATASET``, ...).

``Config`` is a plain ``@dataclass``: every knob is a typed field with a default,
so the whole experiment is reproducible from a single object.  ``from_env()`` is
the single place where string environment variables are coerced back to their
declared field types.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


def _as_bool(value: str) -> bool:
    """Parse a string environment variable into a boolean.

    Accepts the common truthy spellings (``1``/``true``/``yes``/``on``); anything
    else (including an empty string) is treated as falsy.  Used only by
    ``Config.from_env`` for fields declared as ``bool``.
    """
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # ------------------------------------------------------------------ LLM
    # The paper uses DeepSeek-R1 ("deepseek-reasoner" on the official API).
    # "deepseek-chat" (DeepSeek-V3) is a faster drop-in alternative.
    model: str = "deepseek/deepseek-r1-0528"
    # api_base: str = "https://api.deepseek.com"
    # OpenAI-compatible base URL; the client appends ``/chat/completions``.
    api_base: str = "https://openrouter.ai/api/v1"
    # Name of the environment variable holding the DeepSeek API key.
    api_key_env: str = "DEEPSEEK_API_KEY"
    # Sampling controls passed straight through to the completion API.
    temperature: float = 0.3
    top_p: float = 0.8
    # The paper reports a ``repetition_penalty`` of 1.8 (a HuggingFace-style
    # parameter).  The official OpenAI-compatible API has no repetition_penalty;
    # ``frequency_penalty`` is the closest analogue and is kept at 0.0 so the
    # prompt-contracted JSON output is not distorted.
    frequency_penalty: float = 0.0
    max_tokens: int = 4096
    request_timeout: float = 300.0
    max_retries: int = 6
    # Force the model to emit JSON (DeepSeek supports json_object mode).
    json_mode: bool = True

    # ------------------------------------------------------------- Evolution
    max_iterations: int = 30
    population_size: int = 20
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2  # per-gene mutation rate (Table 1)
    offspring_size: int = 20  # candidate offspring generated per generation

    # -------------------------------------------------------------- Encoder
    aggregation_depth: int = 3  # k in Eq. (7)
    input_dim: int = 30  # d in Table 1 (6 edges x 5 ops for NAS-Bench-201)
    # Apply the fixed linear projection of Eq. (6) on top of the one-hot
    # operator encoding.  False -> pure one-hot + aggregation.
    use_projection: bool = True

    # ------------------------------------------------------------------- PKB
    alpha: float = 0.6  # performance weight in Eq. (4)
    k_perf: int = 10  # top-K by combined score (K_perf)
    k_div: int = 12  # top-K by crowding distance (K_div)
    selected_size: int = 20  # final number of PKB exemplars (selected_size)

    # ---------------------------------------------------------- NAS-Bench-201
    dataset: str = "cifar10"  # cifar10 | cifar100 | ImageNet16-120
    # Path to the NAS-Bench-201 ``.pth`` archive (auto-discovered if None).
    data_path: str | None = None

    # ------------------------------------------------------------ Query budget
    # Number of ground-truth benchmark lookups ("queries").  The paper uses 100
    # for NAS-Bench-201 (Table 4).
    query_budget: int = 100
    n_init: int = 20  # ground-truth evaluations used to seed the archive

    # ---------------------------------------------------------------- Runtime
    seed: int = 0
    # If True, run in "offline" mode: use a deterministic pseudo-LLM instead of
    # calling the DeepSeek API (useful for smoke-testing the pipeline).
    offline: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a ``Config``, overriding defaults from ``LLMENAS_*`` env vars.

        For every declared field ``name``, the environment variable
        ``LLMENAS_<NAME>`` (upper-cased) is read if present and coerced to the
        field's declared type (bool/int/float, otherwise kept as a string).  Any
        variable not set is silently ignored, leaving the dataclass default.
        """
        cfg = cls()
        # Map field name -> Field object so each field's declared type is readable.
        env_map = {f.name: f for f in fields(cls)}
        for name, fld in env_map.items():
            key = f"LLMENAS_{name.upper()}"
            if key not in os.environ:
                continue
            raw = os.environ[key]
            if fld.type is bool:
                setattr(cfg, name, _as_bool(raw))
            elif fld.type is int:
                setattr(cfg, name, int(raw))
            elif fld.type is float:
                setattr(cfg, name, float(raw))
            elif fld.type in (str | None, None):
                # Optional[str] (``str | None``) or plain ``str``: keep the raw string.
                setattr(cfg, name, raw)
            else:
                setattr(cfg, name, raw)
        return cfg
