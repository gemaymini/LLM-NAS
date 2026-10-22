"""LLM-ENAS-PKB.

Large language model (DeepSeek) assisted evolutionary neural architecture search
with population knowledge base enhancement.

Reference
---------
Fang, W., Xue, Y., Yuan, L., Hasan, M. K., Aurangzeb, K. (2026).
"Large language model assisted evolutionary neural architecture search with
population knowledge base enhancement." Information Sciences, 735, 123110.

This package implements the *regression-based* LLM surrogate model evaluated on
NAS-Bench-201 (see Sec. 3.2.1, 3.3, 3.4 and Table 1 of the paper).
"""

from .config import Config

# ``Config`` is the single public entry point re-exported at the package top level,
# so callers can write ``from llm_enas_pkb import Config`` without reaching into
# the submodule.  The remaining modules (search_space, encoder, pkb, predictor,
# evolution, ...) are used internally by the pipeline and need not be public.
__all__ = ["Config"]
__version__ = "0.1.0"
