"""LLM-ENAS-PKB.

Large language model (DeepSeek) assisted evolutionary neural architecture search
with population knowledge base enhancement.

Reference
---------
Fang, W., Xue, Y., Yuan, L., Hasan, M. K., Aurangzeb, K. (2026).
"Large language model assisted evolutionary neural architecture search with
population knowledge base enhancement." Information Sciences, 735, 123110.

This package implements a *regression-based* LLM surrogate variant on
NAS-Bench-201.  Architectures use six categorical operation IDs directly, and
prompts include search-space and network-construction code.  This encoding and
prompt variant differs from the paper's architecture information encoder.
"""

from .config import Config

# ``Config`` is the single public entry point re-exported at the package top level,
# so callers can write ``from llm_enas_pkb import Config`` without reaching into
# the submodule.  The remaining modules (search_space, encoder, pkb, predictor,
# evolution, ...) are used internally by the pipeline and need not be public.
__all__ = ["Config"]
__version__ = "0.1.0"
