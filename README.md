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
| Population knowledge base selection | `pkb.py` | Candidate coverage + global references; original Alg. 2 variant remains selectable |
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

The default PKB strategy is **candidate-aware**: select 6 global references using
the existing performance/diversity criteria, then fill the remaining 14 of 20
slots by greedy coverage of the unknown prediction batch. For each candidate,
coverage is the mean of its two highest matching-position similarities to the
selected references; the objective averages this over distinct unknown candidates.
Missing neighbors contribute zero. Already-measured candidate architectures do not
receive coverage weight. If parent prediction refresh is enabled, its unmeasured
parents participate in the same coverage objective.

The six global slots rescale `k_perf:k_div` (default `10:12`) to `3:3` before
merging, de-duplicating and filling; they are not a truncation of the original
20-example PKB. Each subsequent example maximizes coverage gain. Ties favor
non-redundancy to the selected references, the original combined score, then archive
order. When all gains are zero, mean candidate similarity is the first tie-breaker.
Small archives are kept whole; if there are no unknown candidates, selection falls
back to the original strategy. Examples remain **architecture + measured value**;
historical predictions and error categories are not included.

Use `--pkb-strategy original` for the previous selector. Configure size with
`--selected-size`, `--pkb-global-size`, and `--pkb-coverage-neighbors` (defaults
20, 6, 2). A global size of zero selects only by candidate coverage; a global
size above the available PKB size is capped. PKB diagnostics in the search log
and `history[*]["pkb"]` record each example's source, mean/worst nearest and
top-k similarities, and per-candidate coverage. These are structural coverage
metrics, not prediction confidence or evidence of improved accuracy. See
[the selection specification](docs/candidate_aware_pkb.md) for details and evaluation.

Each generation refreshes predictions, queries ground truth from the merged,
de-duplicated parent/offspring pool, and **then selects survivors once**. A new
prediction replaces an unmeasured architecture's previous prediction, while
archived ground truth always takes precedence. By default only offspring are
predicted. `--refresh-parent-predictions` also includes unmeasured parents in the
same prediction batch; this increases token usage, not the number of prediction
batches. `--no-refresh-parent-predictions` overrides an enabled environment setting.

Queries no longer require prior survivor selection. The per-generation quota is
`ceil(remaining_budget / remaining_generations)`, capped by the number of unique
unmeasured candidates. Initialization counts toward `query_budget`; a budget
smaller than the initial population is rejected before search. Unspent queries
roll forward, but unused budget is possible if no unmeasured candidates remain.

The default `query_explore_ratio=1/3` reserves approximately one third of
post-initialization queries for uniform sampling without replacement; the other
queries take the highest current predictions. Exploitation candidates are removed
before exploration sampling. Fractional exploration slots carry across generations:
after `Q` actual post-initialization queries, the exploration total is
`floor(Q * query_explore_ratio)` (with floating-point roundoff tolerance).
For single-query generations at the default ratio, the allocation repeats
`exploit, exploit, explore`. Set `--query-explore-ratio` or
`LLMENAS_QUERY_EXPLORE_RATIO` in `[0, 1]`; `0` queries only by predicted rank and
`1` queries only by uniform exploration. These endpoints still query before selection.

After query corrections, up to `floor(population_size * elite_ratio)` protected
slots go to **measured candidates ranked by ground truth**, within the current
merged pool. Binary tournaments fill the remaining slots using ground truth when
available and the latest prediction otherwise. Missing measured elites release
their slots to tournaments; unmeasured candidates cannot occupy protected slots.
The default `elite_ratio=0.25` reserves up to 5 true elites for a population of 20.
Set `Config(elite_ratio=...)`, `LLMENAS_ELITE_RATIO`, or `--elite-ratio` in `[0, 1]`;
`0` disables protected slots, while `1` uses only true elites if enough measured
candidates exist, and tournaments otherwise. Small pools are kept without padding
or duplicate survivors. Prediction errors can still affect tournament survivors.

Search logs record the policy and new settings, incoming parent score/tag snapshots,
the generation of each active prediction, query reasons (`exploit` / `explore`),
pre-query predictions, true accuracies, signed errors (`prediction - truth`),
whether queried candidates survived, and the final measured elites. A queried
candidate may legitimately be rejected in the same generation. `mean-pop-valid`
still averages only measured members of the **final next population**. Unique
survival counts satisfy `parents_retained + offspring_selected - overlap_selected
= next_population_size`; `new_offspring_selected` excludes parent/offspring overlap.
The returned `history` also includes query records, elites, and these counts.

This policy changes the search trajectory relative to the previous mixed-score
elitism. Compare repeated runs with the same initial populations, seed sets, and
query budgets; a seed controls local randomness, not remote LLM response determinism.
See [the implementation and evaluation notes](docs/prediction_error_mitigation.md).

| Parameter | Value | Env override |
|---|---|---|
| model | `DeepSeek-V4-Pro-0813` | `LLMENAS_MODEL` |
| temperature | `0.3` | `LLMENAS_TEMPERATURE` |
| top_p | `0.8` | `LLMENAS_TOP_P` |
| architecture code | 6 categorical IDs in `0..4` | Fixed by the search space |
| iterations | `30` | `LLMENAS_MAX_ITERATIONS` |
| population size | `20` | `LLMENAS_POPULATION_SIZE` |
| elite ratio | `0.25` | `LLMENAS_ELITE_RATIO` |
| query exploration ratio | `1/3` | `LLMENAS_QUERY_EXPLORE_RATIO` |
| refresh unmeasured parent predictions | `False` | `LLMENAS_REFRESH_PARENT_PREDICTIONS` |
| crossover prob | `0.8` | `LLMENAS_CROSSOVER_PROB` |
| PKB `alpha` | `0.6` | `LLMENAS_ALPHA` |
| PKB strategy | `candidate_aware` | `LLMENAS_PKB_STRATEGY` |
| PKB size | `20` | `LLMENAS_SELECTED_SIZE` |
| PKB global reference count | `6` | `LLMENAS_PKB_GLOBAL_SIZE` |
| PKB coverage neighbors | `2` | `LLMENAS_PKB_COVERAGE_NEIGHBORS` |
| query budget | `100` | `LLMENAS_QUERY_BUDGET` |
| prompt log | `llm_prompts.log` | `LLMENAS_PROMPT_LOG_PATH` |
