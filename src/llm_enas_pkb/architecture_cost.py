"""Label-free costs of instantiated networks, shared by prompts and search logs.

Meta tensors carry shapes but no weight/activation data. Profiling therefore
does not train a model, consume random initialization draws, or allocate GPU
memory. FLOPs count only Conv2d/Linear multiply-accumulates (2 FLOPs/MAC).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

FEATURE_MODES = ("none", "params", "flops", "both")
_IMAGE_SIZES = {"cifar10": 32, "cifar100": 32, "ImageNet16-120": 16}
COST_PROTOCOL = "conv_linear_2flops_per_mac_v1"


def validate_feature_mode(mode: str) -> None:
    if mode not in FEATURE_MODES:
        raise ValueError(f"architecture_features must be one of {FEATURE_MODES}, got {mode!r}")


@dataclass(frozen=True)
class ArchitectureCost:
    params: int
    flops: int | None


def architecture_cost(code, *, dataset: str = "cifar10", C: int = 16, N: int = 5,
                      include_flops: bool = True) -> ArchitectureCost:
    """Count full-network trainable parameters and optional inference FLOPs.

    Validate before caching: Python treats bools/ints/floats with equal values
    as identical keys. Cache only immutable statistics, never model objects.
    """
    from .network import _as_arch

    code = _as_arch(code)
    if dataset not in _IMAGE_SIZES:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in (C, N)):
        raise ValueError("C and N must be positive integers")
    if not isinstance(include_flops, bool):
        raise ValueError("include_flops must be a boolean")
    return _measure_cost(code, dataset, int(C), int(N), include_flops)


@lru_cache(maxsize=49152)
def _measure_cost(code: tuple[int, ...], dataset: str, C: int, N: int,
                  include_flops: bool) -> ArchitectureCost:
    import torch
    from torch import nn

    from .network import arch_to_network

    macs = 0

    def count_macs(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            macs += output.numel() * (module.in_channels // module.groups) * kernel
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    with torch.device("meta"), torch.inference_mode():
        network = arch_to_network(code, dataset=dataset, C=C, N=N).eval()
        params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        if not include_flops:
            return ArchitectureCost(params=params, flops=None)
        handles = [module.register_forward_hook(count_macs) for module in network.modules()
                   if isinstance(module, (nn.Conv2d, nn.Linear))]
        try:
            size = _IMAGE_SIZES[dataset]
            network(torch.empty(1, 3, size, size))
        finally:
            for handle in handles:
                handle.remove()
    return ArchitectureCost(params=params, flops=2 * macs)


def format_architecture_features(code, *, dataset: str, mode: str) -> str:
    """Render the same exact integer counts for PKB, candidates and logs."""
    validate_feature_mode(mode)
    if mode == "none":
        return ""
    cost = architecture_cost(code, dataset=dataset, include_flops=mode in ("flops", "both"))
    fields = []
    if mode in ("params", "both"):
        fields.append(f"<Params>: {cost.params};")
    if mode in ("flops", "both"):
        fields.append(f"<FLOPs>: {cost.flops};")
    return " ".join(fields)


def feature_description(mode: str, dataset: str) -> str:
    validate_feature_mode(mode)
    if mode == "none":
        return ""
    size = _IMAGE_SIZES[dataset]
    descriptions = [
        f"Architecture features: mode={mode}, protocol={COST_PROTOCOL}. "
        "These are exact integer counts for the entire instantiated network "
        "(C=16, N=5, including stem, stage transitions and classification head), "
        "computed locally from architecture and input shapes without training "
        "or querying benchmark accuracy."
    ]
    if mode in ("params", "both"):
        descriptions.append("Params counts trainable scalar parameters, including BatchNorm "
                            "affine parameters and classifier bias; it excludes buffers.")
    if mode in ("flops", "both"):
        descriptions.append(
            f"FLOPs counts Conv2d and Linear operations for one RGB {size}x{size} input "
            "(batch size 1, inference), with 1 multiply-accumulate = 2 FLOPs. "
            "It excludes bias additions, BatchNorm, activations, pooling, elementwise "
            "sums and zero operations; it is not total runtime or measured latency. "
            "All executed convolutions are counted, including those on zero or "
            "output-disconnected branches; no graph pruning is performed."
        )
    descriptions.append(
        "Use these features only as auxiliary evidence calibrated against PKB labels. "
        "Larger counts do not guarantee higher accuracy. Architectures with equal "
        "counts can have different connectivity and accuracy; keep comparing edge "
        "positions and input-to-output paths. When both features are present, they "
        "can be highly correlated and must not be treated as independent proof."
    )
    return " ".join(descriptions)
