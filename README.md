# LLM-ENAS-PKB — Regression surrogate on NAS-Bench-201

A faithful re-implementation of the **regression-based** LLM surrogate model from

> Fang, W., Xue, Y., Yuan, L., Hasan, M. K., Aurangzeb, K. (2026).
> *Large language model assisted evolutionary neural architecture search with
> population knowledge base enhancement.* Information Sciences, 735, 123110.
> ([official code](https://github.com/baigeixiaowang/LLM-ENAS-PKB))

The paper replaces the expert-designed surrogate model in evolutionary NAS with a
large language model (DeepSeek-R1). This repository implements the regression
variant on **NAS-Bench-201** (`d = 30` input dimension, see Table 1 of the paper),
driven by the **DeepSeek official API**.

## What is implemented

| Component | Module | Paper |
|---|---|---|
| NAS-Bench-201 search space (6 edges × 5 ops) | `search_space.py` | Sec. 4.1 |
| Architecture information encoder (Eq. 6–7, 30-dim) | `encoder.py` | Sec. 3.4 |
| Population knowledge base selection | `pkb.py` | Sec. 3.3, Alg. 2 |
| Regression prompt builder | `prompts.py` | Sec. 3.2.1, Fig. 2 |
| DeepSeek official API client | `llm.py` | Table 1 |
| Regression predictor (ties the above together) | `predictor.py` | Sec. 3.2 |
| Evolutionary loop | `evolution.py` | Sec. 3.1, Alg. 1 |
| CLI entry point | `main.py` | — |

## Environment (uv)

```bash
uv sync                 # create .venv + install deps (incl. torch, nas-bench-201)
export DEEPSEEK_API_KEY=sk-...   # or put it in .env
uv run llm-enas-pkb --dataset cifar10 --iterations 30
```

Python is managed by `uv` (`requires-python >=3.10,<3.13`); no system Python is
used.

## DeepSeek API

* Endpoint: `https://api.deepseek.com/chat/completions` (OpenAI-compatible).
* Model: `deepseek-v4-pro` (DeepSeek-V4 reasoning model). The paper used
  `deepseek-reasoner` (DeepSeek-R1), which the official API has since retired.
  Set `LLMENAS_MODEL=deepseek-flash` to use the faster V4.1-Flash model instead.
* Auth: `DEEPSEEK_API_KEY` (or `LLMENAS_API_KEY_ENV` to point at another variable).

Note: the paper reports a `repetition_penalty` of 1.8 (a HuggingFace parameter);
the OpenAI-compatible API has no direct equivalent, so `frequency_penalty` (default
0.0) is the closest analogue and is exposed via `LLMENAS_FREQUENCY_PENALTY`.

## NAS-Bench-201 data

`nas-bench-201` (import name `nas_201_api`) is installed from PyPI. The benchmark
archive (`NAS-Bench-201-v1_1-096897.pth`, ~4.7 GB) must be obtained from the
[official repo](https://github.com/D-X-Y/NAS-Bench-201) (Google Drive; automated
download is frequently blocked by Drive's quota limit, so manual download is
recommended):

```bash
# Put the file at either of these locations (auto-detected), or pass --data-path.
data/NAS-Bench-201-v1_1-096897.pth          # project-local (recommended)
~/.torch/NAS-Bench-201-v1_1-096897.pth      # legacy TORCH_HOME location
uv run llm-enas-pkb --data-path /abs/path/NAS-Bench-201-v1_1-096897.pth
```

A best-effort auto-download (via `gdown`) is attempted as a last resort
(`scripts/download_data.py`).

## Running

```bash
# Full evolution (regression surrogate, 30 generations, 100-query budget)
uv run llm-enas-pkb --dataset cifar10 --seed 0

# Quick smoke test without calling the API (deterministic pseudo-LLM)
uv run llm-enas-pkb --offline --iterations 3

# Predict a couple of candidate architectures from a small labelled archive
uv run python - <<'PY'
from llm_enas_pkb import Config
from llm_enas_pkb.nasbench201 import NASBench201
from llm_enas_pkb.predictor import RegressionPredictor

cfg = Config(offline=True)          # deterministic pseudo-LLM (no API key)
nb = NASBench201(cfg)
pred = RegressionPredictor(cfg)

pkb = [(0, 1, 2, 3, 4, 1), (4, 3, 2, 1, 0, 1), (1, 1, 1, 1, 1, 1)]
pkb_vals = [nb.valid_acc(a) for a in pkb]   # ground-truth labels
candidates = [(2, 2, 2, 2, 2, 2), (0, 0, 0, 0, 0, 0)]
print(pred.predict_archs(candidates, pkb, pkb_vals))
PY
```

## Key hyperparameters (Table 1)

| Parameter | Value | Env override |
|---|---|---|
| model | `deepseek-v4-pro` | `LLMENAS_MODEL` |
| temperature | `0.3` | `LLMENAS_TEMPERATURE` |
| top_p | `0.8` | `LLMENAS_TOP_P` |
| input dim `d` | `30` | — |
| aggregation depth `k` | `3` | `LLMENAS_AGGREGATION_DEPTH` |
| iterations | `30` | `LLMENAS_MAX_ITERATIONS` |
| population size | `20` | `LLMENAS_POPULATION_SIZE` |
| crossover prob | `0.8` | `LLMENAS_CROSSOVER_PROB` |
| PKB `alpha` | `0.6` | `LLMENAS_ALPHA` |
| query budget | `100` | `LLMENAS_QUERY_BUDGET` |
