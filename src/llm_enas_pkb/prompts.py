"""Single-user-message regression prompt with executable search-space context."""

from __future__ import annotations

import ast
import json
from functools import lru_cache
from importlib.resources import files

import numpy as np

from .architecture_cost import (
    feature_description,
    format_architecture_features,
    validate_feature_mode,
)
from .categorical import validate_architecture_vectors
from .pkb import PredictionFeedback
from .search_space import EDGES, OPS

_VALUE_PRECISION = 6
_DATASETS = {"cifar10": (10, 32), "cifar100": (100, 32), "ImageNet16-120": (120, 16)}
_OP_DESCRIPTIONS = (
    "Return zeros; the edge contributes nothing.",
    "Pass the input unchanged.",
    "ReLU -> 1x1 convolution -> BatchNorm; padding 0.",
    "ReLU -> 3x3 convolution -> BatchNorm; padding 1.",
    "3x3 average pooling; padding 1; exclude padded zeros; no BatchNorm.",
)


@lru_cache(maxsize=1)
def _reference_code() -> tuple[str, str]:
    """Read actual module definitions without importing torch into the search loop."""
    source = files(__package__).joinpath("network.py").read_text(encoding="utf-8")
    definitions = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    preamble = (
        "from numbers import Integral\nfrom torch import nn\n\n"
        f"OPS = {OPS!r}\nEDGES = {EDGES!r}\n\n"
    )
    operations = preamble + "\n\n".join(definitions[name] for name in (
        "_as_arch", "Zero", "ReLUConvBN", "make_operation",
    ))
    construction = "\n\n".join(definitions[name] for name in (
        "NASBench201Cell", "ResNetBasicblock", "TinyNetwork", "build_network",
        "arch_to_network", "_operation_id", "network_to_arch", "network_to_code",
    ))
    return operations, construction


def build_regression_prompt(
    pkb_vectors: np.ndarray,
    pkb_values: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    dataset: str = "cifar10",
    generation: int | None = None,
    pkb_feedback: list[PredictionFeedback | None] | None = None,
    architecture_features: str = "both",
) -> str:
    """Describe the space and predict (M,) accuracies from (N, 6)/(M, 6) codes."""
    validate_feature_mode(architecture_features)
    pkb = validate_architecture_vectors(pkb_vectors)
    candidates = validate_architecture_vectors(candidate_vectors)
    values = np.asarray(pkb_values, dtype=np.float64)
    if values.shape != (len(pkb),) or not np.all(np.isfinite(values)):
        raise ValueError("PKB values must be a finite one-dimensional array matching the PKB")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("PKB accuracy values must lie in [0, 1]")
    if dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if generation is not None and (
        isinstance(generation, bool) or not isinstance(generation, (int, np.integer))
        or generation < 1
    ):
        raise ValueError("Generation must be a positive integer when provided")
    feedback = [None] * len(pkb) if pkb_feedback is None else list(pkb_feedback)
    if len(feedback) != len(pkb):
        raise ValueError("Expected one feedback entry per PKB row")
    for entry in feedback:
        if entry is None:
            continue
        if not isinstance(entry, PredictionFeedback):
            raise ValueError("PKB feedback entries must be PredictionFeedback or None")
        if generation is not None and entry.query_generation >= generation:
            raise ValueError("PKB feedback must come from an earlier search generation")

    classes, image_size = _DATASETS[dataset]
    op_code, network_code = _reference_code()
    n_candidates = len(candidates)
    generation_text = str(generation) if generation is not None else "unspecified"
    lines = [
        "1. Role Definition",
        "",
        "You are a regression surrogate for neural architecture search in NAS-Bench-201. "
        "Use the search-space definition, network implementation, and labelled examples "
        "provided below to estimate the validation accuracy of candidate architectures. "
        "Your predictions will guide selection in the current search generation.",
        "",
        "2. Task Definition",
        "",
        f"Predict validation classification accuracy on {dataset} ({classes} classes, "
        f"RGB input {image_size}x{image_size}) under the 200-epoch NAS-Bench-201 protocol.",
        "",
        "The prediction target, Value, is validation accuracy expressed as a fraction "
        "in [0, 1]: 0.90 means 90% of validation examples are classified correctly. "
        "Use this accuracy scale consistently for the supplied labels and all predictions.",
        "",
        "The Population Knowledge Base (PKB) pairs architecture codes with measured "
        "validation accuracies from benchmark models trained under this protocol on the "
        "specified dataset. Treat these pairs as labelled regression examples. Your task "
        "is to estimate the same validation metric for each candidate code in Section 6.",
        "",
        "3. Search Space Definition: Operation Descriptions and Code",
        "",
        "The searchable component is a normal cell represented by a directed acyclic "
        "graph with four nodes, numbered 0 through 3. Node 0 receives the cell input, "
        "and node 3 produces the cell output. The six possible edges are 0->1, 0->2, "
        "1->2, 0->3, 1->3, and 2->3. Each edge independently selects one of five "
        "operations, giving 5^6 = 15625 architecture codes.",
        "",
        "Let h_i denote the feature tensor at node i, and let o_ij denote the "
        "operation selected for edge i->j. The cell computes:",
        "h_0 = cell input",
        "h_1 = o_01(h_0)",
        "h_2 = o_02(h_0) + o_12(h_1)",
        "h_3 = o_03(h_0) + o_13(h_1) + o_23(h_2)",
        "cell output = h_3",
        "Thus, nodes 1-3 sum their incoming edge outputs elementwise. All edge "
        "operations preserve channel count and spatial size, so these sums are "
        "well-defined.",
        "",
        "The operation IDs have the following fixed meanings:",
    ]
    lines.extend(f"{i} = {name}: {description}" for i, (name, description)
                 in enumerate(zip(OPS, _OP_DESCRIPTIONS)))
    lines.extend([
        "All edge operations use stride 1. Convolutions have no bias. BatchNorm uses "
        "affine=True and track_running_stats=True. The following code defines these "
        "operations and validates the six-ID architecture codes used in the next section.",
        "```python", op_code, "```",
        "",
        "4. Network Construction and Bidirectional Architecture Encoding",
        "",
        "Each architecture is represented directly by a list of six integers in [0, 4]. "
        "Each position specifies the operation on one fixed edge:",
        "[op_01, op_02, op_12, op_03, op_13, op_23]",
        "Position-to-edge mapping (zero-based): "
        + "; ".join(f"{i}: {src}->{dst}" for i, (src, dst) in enumerate(EDGES)) + ".",
        "Operation IDs are categorical labels. Their numerical order or difference "
        "does not indicate operation similarity or expected accuracy. Interpret each "
        "ID together with its edge position and the operation definition in Section 3.",
        "",
        "For example, [0, 1, 2, 3, 4, 1] assigns none to 0->1, skip_connect to 0->2, "
        "nor_conv_1x1 to 1->2, nor_conv_3x3 to 0->3, avg_pool_3x3 to 1->3, and "
        "skip_connect to 2->3.",
        "",
        "This cell is repeated within a fixed classifier. C=16 is the initial channel "
        "count, and N=5 is the number of normal cells per stage. The network is:",
        "Conv3x3+BN stem",
        "-> 5 normal cells at 16 channels",
        "-> stride-2 residual block",
        "-> 5 normal cells at 32 channels",
        "-> stride-2 residual block",
        "-> 5 normal cells at 64 channels",
        "-> BN+ReLU",
        "-> global average pooling",
        f"-> {classes}-class linear head.",
        "All 15 normal cells use the same six operation choices, with independent "
        "weights in each cell. Each residual block halves the spatial resolution and "
        "doubles the channel count. The stem, residual blocks, and classification head "
        "have fixed structures for this task; the six-ID code specifies only the "
        "operations inside the normal cells.",
        "",
        "The implementation below makes the connection between a code and its network "
        "explicit. arch_to_network builds a network with freshly initialized weights. "
        "network_to_code recovers the six IDs by inspecting the normal-cell operations "
        "and checks that all normal cells use the same code. The final example "
        "demonstrates a code-to-network-to-code round trip.",
        "",
        "This code continues the previous Python block and serves as a structural "
        "reference for prediction. The measured accuracy evidence is supplied separately "
        "in the PKB; the conversion example is an illustration of the encoding.",
        "```python", network_code, "",
        "code = [0, 1, 2, 3, 4, 1]",
        f"network = arch_to_network(code, dataset={dataset!r})",
        "assert network_to_code(network) == code", "```",
        "",
        "5. Prediction Instructions for This Generation",
        "",
        f"Generation: {generation_text}.",
        f"Use the {len(pkb)} PKB examples in Section 6 to predict Value for all "
        f"{n_candidates} candidate entries in this generation. The generation number "
        "identifies the search round; base the predictions on the candidate structures "
        "and the supplied measured labels.",
        "",
        "If a candidate code appears in the PKB, use its supplied measured Value. "
        "For other candidates, use the PKB examples to relate architecture differences "
        "to validation-accuracy differences. Compare operation choices at corresponding "
        "edge positions, paths from the cell input to its output, and interactions "
        "among operations. Evaluate each code as a complete cell within the fixed "
        "network described in Section 4; the effect of an operation depends on its "
        "position and the surrounding operations.",
        "",
        "Use the observed labels to calibrate the accuracy scale and the size of "
        "predicted differences. The PKB is a selected subset of evaluated architectures, "
        "so candidate accuracies may fall outside its observed range when supported "
        "by the architecture evidence. When comparable examples are sparse or conflicting, "
        "make conservative estimates and avoid large adjustments based on weak evidence.",
        "",
        "The supplied PKB is the only source of measured labels for this prediction "
        "task. Estimate accuracy from the provided information without training or "
        "running networks, or retrieving external benchmark results. Assign identical "
        "predictions to repeated candidate codes and keep a separate output entry for "
        "each occurrence.",
        "",
        "6. Population Knowledge Base and Candidate Architecture Codes",
        "",
        "Each PKB entry contains an Architecture Code and its measured Value. Each "
        "candidate entry contains an Architecture Code with an empty Value field for "
        "you to estimate. Both lists use the same six-ID encoding. Entry numbers "
        "identify rows. Use candidate numbering to preserve output order, and use "
        "architecture codes and measured values as the evidence for prediction.",
        "",
        "Population Knowledge Base (measured validation accuracy):",
    ])
    if architecture_features != "none":
        lines.extend(["", feature_description(architecture_features, dataset), "",
                      "Both PKB and candidate rows include the enabled architecture features.", ""])
    if any(entry is not None for entry in feedback):
        lines.extend([
            "",
            "Some PKB rows also include historical surrogate feedback. Historical Prediction "
            "is the latest estimate made before that architecture's first ground-truth query. "
            "Prediction Error = Historical Prediction - measured Value, on the same [0, 1] "
            "accuracy scale: +0.02 means overestimated by 2 percentage points, and -0.02 "
            "means underestimated by 2 percentage points. Direction is based on the displayed "
            "six-decimal values; matched means equal at that precision. Prediction Generation "
            "and Query Generation identify when the estimate and its verification occurred.",
            "Use these past errors as calibration evidence when comparing similar structures. "
            "The measured Value remains the regression label. Do not copy the historical "
            "prediction or mechanically apply its error to every candidate. These are selected "
            "observations from earlier generations, not proof of a universal structural bias. "
            "Missing feedback means no pre-query prediction is available, not zero error.",
            "",
        ])
    for i, (code, value, entry) in enumerate(zip(pkb, values, feedback), 1):
        row = (f"PKB {i}: <Architecture Code>: {json.dumps(code.tolist())}; "
               f"<Value>: {float(value):.{_VALUE_PRECISION}f};")
        features = format_architecture_features(code, dataset=dataset, mode=architecture_features)
        if features:
            row += " " + features
        if entry is not None:
            row += " " + entry.describe(float(value))
        lines.append(row)
    if not len(pkb):
        lines.append("(No measured labels are available.)")
    lines.extend(["", "Candidate Architectures (predict in this order):"])
    for i, code in enumerate(candidates, 1):
        row = f"Candidate {i}: <Architecture Code>: {json.dumps(code.tolist())}; <Value>: ;"
        features = format_architecture_features(code, dataset=dataset, mode=architecture_features)
        if features:
            row += " " + features
        lines.append(row)
    lines.extend([
        "",
        "7. Constraints",
        "",
        'Return only a valid JSON object with the single key "Value".',
        f'"Value" must be a list of exactly {n_candidates} numbers in candidate order, '
        "including repeated candidates.",
        "The first number must correspond to Candidate 1, the second to Candidate 2, "
        "and so on. Preserve this order and include every candidate occurrence.",
        "Each number must be finite, between 0 and 1 inclusive, with at most six "
        "decimal places.",
        "Return numeric accuracy estimates on the scale defined in Section 2. "
        "Exclude explanations, Markdown fences, candidate codes, and additional keys "
        "from the response.",
    ])
    return "\n".join(lines)
