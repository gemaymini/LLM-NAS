"""Executable NAS-Bench-201 network reference used by the regression prompt.

The six operation IDs describe the repeated normal cell, not trained weights.
The macro structure follows the benchmark's C=16, N=5 configuration. Reference:
https://github.com/D-X-Y/AutoDL-Projects/tree/main/xautodl/models/cell_infers
https://github.com/D-X-Y/AutoDL-Projects/blob/main/xautodl/models/cell_operations.py

Search still queries the benchmark accuracy table; constructing these modules
initializes fresh weights and does not reproduce the benchmark's trained weights.
"""

from __future__ import annotations

from numbers import Integral

from torch import nn

from .search_space import EDGES, OPS


def _as_arch(code) -> tuple[int, ...]:
    code = tuple(code)
    if len(code) != len(EDGES) or any(
        isinstance(op, bool) or not isinstance(op, Integral) or not 0 <= op < len(OPS)
        for op in code
    ):
        raise ValueError("Architecture must contain six integer operation IDs in [0, 4]")
    return tuple(int(op) for op in code)


class Zero(nn.Module):
    def forward(self, x):
        return x.mul(0.0)


class ReLUConvBN(nn.Sequential):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int = 1):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(C_out, affine=True, track_running_stats=True),
        )


def make_operation(op_id: int, C: int) -> nn.Module:
    """Build a normal-cell edge operation: equal channels and stride one."""
    if isinstance(op_id, bool) or not isinstance(op_id, Integral):
        raise ValueError("Operation ID must be an integer in [0, 4]")
    if op_id == 0:
        return Zero()
    if op_id == 1:
        return nn.Identity()
    if op_id in (2, 3):
        return ReLUConvBN(C, C, 1 if op_id == 2 else 3)
    if op_id == 4:
        return nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
    raise ValueError("Operation ID must be an integer in [0, 4]")


class NASBench201Cell(nn.Module):
    def __init__(self, code, C: int):
        super().__init__()
        self.ops = nn.ModuleList(make_operation(op, C) for op in _as_arch(code))

    def forward(self, x):
        nodes = [x]
        for dst in range(1, 4):
            nodes.append(sum(
                self.ops[i](nodes[src])
                for i, (src, target) in enumerate(EDGES) if target == dst
            ))
        return nodes[3]


class ResNetBasicblock(nn.Module):
    """Fixed stage transition; its operations are not part of the six-ID code."""

    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        self.main = nn.Sequential(
            ReLUConvBN(C_in, C_out, 3, stride=2),
            ReLUConvBN(C_out, C_out, 3),
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, stride=2), nn.Conv2d(C_in, C_out, 1, bias=False)
        )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class TinyNetwork(nn.Module):
    def __init__(self, code, num_classes: int, C: int = 16, N: int = 5):
        super().__init__()
        code = _as_arch(code)
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1
               for v in (C, N, num_classes)):
            raise ValueError("C, N, and num_classes must be positive integers")
        self.stem = nn.Sequential(nn.Conv2d(3, C, 3, padding=1, bias=False), nn.BatchNorm2d(C))
        cells = []
        for stage in range(3):
            channels = C * 2 ** stage
            if stage:
                cells.append(ResNetBasicblock(channels // 2, channels))
            cells.extend(NASBench201Cell(code, channels) for _ in range(N))
        self.cells = nn.ModuleList(cells)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C * 4), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        features = self.global_pooling(self.lastact(x)).flatten(1)
        return features, self.classifier(features)


def build_network(code, *, dataset: str = "cifar10", C: int = 16, N: int = 5) -> TinyNetwork:
    """Decode six operation IDs into a freshly initialized classifier."""
    classes = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}
    if dataset not in classes:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return TinyNetwork(code, classes[dataset], C=C, N=N)


def arch_to_network(code, *, dataset: str = "cifar10", C: int = 16, N: int = 5) -> TinyNetwork:
    return build_network(code, dataset=dataset, C=C, N=N)


def _operation_id(op: nn.Module) -> int:
    """Recover the ID from the actual normal-cell module, not a saved genotype."""
    if isinstance(op, Zero):
        return 0
    if isinstance(op, nn.Identity):
        return 1
    if isinstance(op, ReLUConvBN) and len(op) == 3:
        relu, conv, bn = op
        if (
            isinstance(relu, nn.ReLU) and not relu.inplace
            and isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
            and conv.kernel_size in ((1, 1), (3, 3)) and conv.stride == (1, 1)
            and conv.padding == tuple(k // 2 for k in conv.kernel_size)
            and conv.dilation == (1, 1) and conv.groups == 1 and conv.bias is None
            and conv.padding_mode == "zeros" and conv.in_channels == conv.out_channels
            and bn.num_features == conv.out_channels and bn.affine and bn.track_running_stats
        ):
            return 2 if conv.kernel_size == (1, 1) else 3
    if isinstance(op, nn.AvgPool2d) and op.kernel_size == 3 and op.stride == 1:
        if (op.padding == 1 and not op.count_include_pad and not op.ceil_mode
                and op.divisor_override is None):
            return 4
    raise ValueError(f"Unsupported normal-cell operation: {op!r}")


def network_to_arch(network: nn.Module) -> tuple[int, ...]:
    """Encode a cell or TinyNetwork made here; reject inconsistent cell choices."""
    if isinstance(network, NASBench201Cell):
        return _as_arch(_operation_id(op) for op in network.ops)
    if not isinstance(network, TinyNetwork):
        raise TypeError("Expected a NASBench201Cell or TinyNetwork built by this module")
    codes = [network_to_arch(cell) for cell in network.cells
             if isinstance(cell, NASBench201Cell)]
    if not codes or any(code != codes[0] for code in codes[1:]):
        raise ValueError("A six-ID architecture requires identical codes in all normal cells")
    return codes[0]


def network_to_code(network: nn.Module) -> list[int]:
    return list(network_to_arch(network))
