# LLM-ENAS-PKB — Regression surrogate on NAS-Bench-201

An implementation of the **regression-based** LLM surrogate search described in

> Fang, W., Xue, Y., Yuan, L., Hasan, M. K., Aurangzeb, K. (2026).
> *Large language model assisted evolutionary neural architecture search with
> population knowledge base enhancement.* Information Sciences, 735, 123110.
> ([official code](https://github.com/baigeixiaowang/LLM-ENAS-PKB))

The paper replaces the expert-designed surrogate model in evolutionary NAS with a
large language model. This repository runs a regression surrogate on
**NAS-Bench-201** through an OpenAI-compatible chat-completions API.

The current version is an explicit encoding and prompt variant: each architecture
is a six-element list of categorical operation IDs, and the prompt explains the
operators and executable network construction. The former one-hot projection and
neighbour aggregation encoder has been removed. PKB diversity and the offline
surrogate use the fraction of matching edge operations. These choices differ from
the paper's Eq. (2) and architecture encoder in Eq. (6–7), so results should be
reported with this variant identified.

## What is implemented

| Component | Module | Paper |
|---|---|---|
| NAS-Bench-201 search space (6 edges × 5 ops) | `search_space.py` | Sec. 4.1 |
| Direct categorical encoding (6 operation IDs) | `encoder.py` | Encoding variant |
| Executable cell/network and architecture conversion | `network.py` | NAS-Bench-201 construction |
| Population knowledge base selection | `pkb.py` | Sec. 3.3, Alg. 2, with matching-position similarity |
| Seven-section regression prompt | `prompts.py` | Expanded prompt variant |
| OpenAI-compatible API client | `llm.py` | Configurable provider |
| Regression predictor (ties the above together) | `predictor.py` | Sec. 3.2 |
| Evolutionary loop | `evolution.py` | Sec. 3.1, Alg. 1, with configurable elitism |
| CLI entry point | `main.py` | — |

## Environment (uv)

```bash
uv sync                 # create .venv + install deps (incl. torch, nas-bench-201)
export DEEPSEEK_API_KEY=sk-...   # or put it in .env
uv run llm-enas-pkb --dataset cifar10 --iterations 30
```

Python is managed by `uv` (`requires-python >=3.10,<3.13`); no system Python is
used.

## API configuration

* The checked-in defaults are base URL `https://llmapi.paratera.com` and model
  `DeepSeek-V4-Pro-0813`. The client appends `/chat/completions` to the base URL.
* Override the provider and model with `LLMENAS_API_BASE` and `LLMENAS_MODEL`.
* Auth: `DEEPSEEK_API_KEY` (or `LLMENAS_API_KEY_ENV` to point at another variable).

Note: the paper reports a `repetition_penalty` of 1.8 (a HuggingFace parameter);
the OpenAI-compatible API has no direct equivalent, so `frequency_penalty` (default
0.0) is the closest analogue and is exposed via `LLMENAS_FREQUENCY_PENALTY`.

### Prompt logging

Every prompt sent to the API and the text the model returns are appended in real
time (one flush per exchange). By default the CLI creates a timestamped,
hyperparameter-tagged filename for each run. Set an explicit filename to follow
one log:

```bash
uv run llm-enas-pkb --prompt-log-path llm_prompts.log
tail -f llm_prompts.log
```

Disable it with `LLMENAS_PROMPT_LOG_PATH=""` (or `--prompt-log-path ""`), or
redirect it with `LLMENAS_PROMPT_LOG_PATH=/tmp/prompts.log`. Failed writes are
warned and ignored, so a missing/invalid path never aborts the run.

## Architecture codes and prompt contract

An architecture is `[op_01, op_02, op_12, op_03, op_13, op_23]`. Positions always
refer to the directed edges `0→1`, `0→2`, `1→2`, `0→3`, `1→3`, and `2→3`.
Node 0 is the cell input; each following node sums its incoming edge outputs;
node 3 is the cell output. Every position contains one of these IDs:

| ID | Operation | Meaning inside a searchable cell |
|---|---|---|
| `0` | `none` | Produce zeros; remove the edge's contribution |
| `1` | `skip_connect` | Pass the input unchanged |
| `2` | `nor_conv_1x1` | ReLU → 1×1 convolution → batch normalization |
| `3` | `nor_conv_3x3` | ReLU → 3×3 convolution → batch normalization |
| `4` | `avg_pool_3x3` | 3×3 average pooling, stride 1, padding 1 |

For example, `[0, 1, 2, 3, 4, 1]` means a zero edge at `0→1`, a skip at `0→2`,
a 1×1 convolution at `1→2`, a 3×3 convolution at `0→3`, average pooling at
`1→3`, and a skip at `2→3`. Its official benchmark string is
`|none~0|+|skip_connect~0|nor_conv_1x1~1|+|nor_conv_3x3~0|avg_pool_3x3~1|skip_connect~2|`.
IDs are category labels: `4` does not imply better performance or greater
distance than `1`. Similarity is the number of equal positions divided by six.

`network.py` provides `NASBench201Cell` and the complete `TinyNetwork`, using
`C=16`, `N=5`, and dataset-specific classifier sizes by default. Conversion works
on these explicit searchable network instances, including inspection of their
edge operations; it does not reconstruct arbitrary PyTorch models or weights:

```python
from llm_enas_pkb.network import arch_to_network, network_to_arch, network_to_code

code = [0, 1, 2, 3, 4, 1]
network = arch_to_network(code, dataset="cifar10")
assert network_to_arch(network) == tuple(code)
assert network_to_code(network) == code
```

Each prediction call still sends one `user` message, with seven sections:

1. The predictor's role.
2. The validation-accuracy regression task and dataset.
3. The search space, operation descriptions, and operation implementation code.
4. The code positions, network construction, and architecture/network conversion code.
5. The prediction instructions for this generation and its actual candidate count.
6. PKB architecture/ground-truth accuracy pairs and ordered candidate codes.
7. Output constraints: a JSON object containing `Value`, one numeric value
   in `[0, 1]` per candidate in the original order, with no extra prose.

The prompt contains the code definition every time, so neither server-side
conversation state nor a separate system message is required. Predictions are
estimates of benchmark validation accuracy; the code in the prompt is structural
context rather than a request to train or evaluate the network.

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

## Validation

The unit tests use synthetic PKB labels and mocked API responses, so no API key
or benchmark archive is required:

```bash
uv run python -m unittest discover -s tests -v
```

The `--offline` search example above avoids LLM API calls but still needs the
NAS-Bench-201 archive for ground-truth queries.

## Key hyperparameters

After merging and de-duplicating parents and offspring, survivor selection first
keeps the best `floor(population_size * elite_ratio)` individuals by current
fitness (ground-truth accuracy when available, otherwise the surrogate prediction).
Binary tournament selection fills the remaining slots from candidates excluding
those elites, without duplicate survivors. The default `elite_ratio=0.25` keeps
5 elites for a population of 20. Set `Config(elite_ratio=...)`,
`LLMENAS_ELITE_RATIO`, or `--elite-ratio` to a value in `[0, 1]`; `0` disables
elitism and `1` selects only the highest-fitness individuals. If there are fewer
unique candidates than the target population size, all available candidates survive.

| Parameter | Value | Env override |
|---|---|---|
| model | `DeepSeek-V4-Pro-0813` | `LLMENAS_MODEL` |
| temperature | `0.3` | `LLMENAS_TEMPERATURE` |
| top_p | `0.8` | `LLMENAS_TOP_P` |
| architecture code | 6 categorical IDs in `0..4` | Fixed by the search space |
| iterations | `30` | `LLMENAS_MAX_ITERATIONS` |
| population size | `20` | `LLMENAS_POPULATION_SIZE` |
| elite ratio | `0.25` | `LLMENAS_ELITE_RATIO` |
| crossover prob | `0.8` | `LLMENAS_CROSSOVER_PROB` |
| PKB `alpha` | `0.6` | `LLMENAS_ALPHA` |
| query budget | `100` | `LLMENAS_QUERY_BUDGET` |
| prompt log | `llm_prompts.log` | `LLMENAS_PROMPT_LOG_PATH` |
