from __future__ import annotations

import sys
from contextlib import contextmanager
import importlib
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader
from typing import Iterator


def ensure_limix_torch_norm_interface() -> None:
    names = (
        "model.v2_0.operators.rmsnorm.torch_rmsnorm",
        "model.v2_0.operators.rmsnorm",
    )
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        cls = getattr(module, "RMSNormMixedPrecision", None)
        if cls is not None and not hasattr(cls, "hs"):
            cls.hs = property(lambda self: self.normalized_shape[-1])


class _OptionalTritonLoader(SourceFileLoader):
    def get_code(self, fullname: str):
        source = self.get_data(self.path).decode("utf-8")
        original = (
            "from .triton_rmsnorm import TritonRMSNorm\n"
            "from .triton_qknorm import TritonQKNorm"
        )
        replacement = (
            "try:\n"
            "    from .triton_rmsnorm import TritonRMSNorm\n"
            "    from .triton_qknorm import TritonQKNorm\n"
            "except ModuleNotFoundError as exc:\n"
            "    if exc.name is None or not (exc.name == 'triton' or exc.name.startswith('triton.')):\n"
            "        raise\n"
            "    TritonRMSNorm = None\n"
            "    TritonQKNorm = None"
        )
        if original in source:
            anchor_class = "from .torch_rmsnorm import RMSNormMixedPrecision\n"
            class_adapter = (
                "if not hasattr(RMSNormMixedPrecision, 'hs'):\n"
                "    RMSNormMixedPrecision.hs = property(lambda self: self.normalized_shape[-1])\n"
            )
            if anchor_class not in source:
                raise RuntimeError("Unsupported LimiX PyTorch RMSNorm layout.")
            source = source.replace(anchor_class, anchor_class + class_adapter, 1)
            source = source.replace(original, replacement, 1)
            anchor = "    norm_impl = kwargs.get('norm_impl', 'torch')\n"
            guard = (
                "    if norm_impl == 'triton' and (TritonRMSNorm is None or TritonQKNorm is None):\n"
                "        norm_impl = 'torch'\n"
            )
            if anchor not in source:
                raise RuntimeError("Unsupported LimiX RMSNorm factory layout.")
            source = source.replace(anchor, anchor + guard, 1)
        elif "from .triton_rmsnorm import TritonRMSNorm" in source:
            raise RuntimeError("Unsupported LimiX RMSNorm import layout.")
        return compile(source, self.path, "exec")


class _OptionalTritonFinder(MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname != "model.v2_0.operators.rmsnorm":
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, SourceFileLoader):
            spec.loader = _OptionalTritonLoader(fullname, spec.origin)
        return spec


_V1_TORCH_RMSNORM_SOURCE = """from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TritonRMSNorm(nn.Module):
    def __init__(self, hs, eps, elementwise_affine=True, device=None, dtype=torch.float32, recompute=False):
        super().__init__()
        self.hs = hs
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.device = device
        self.dtype = dtype
        self.recompute = recompute
        weight = nn.Parameter(torch.ones(self.hs, dtype=self.dtype)) if self.elementwise_affine else None
        self.register_parameter('weight', weight)

    def forward(self, input, output=None):
        shape = (int(self.hs),) if isinstance(self.hs, int) else tuple(int(x) for x in self.hs)
        weight = self.weight
        if weight is not None:
            weight = weight.to(device=input.device, dtype=input.dtype)
        return F.rms_norm(input, shape, weight, self.eps)

    def extra_repr(self):
        return 'hs=%s, eps=%s, elementwise_affine=%s, dtype=%s, recompute=%s' % (
            self.hs, self.eps, self.elementwise_affine, self.dtype, self.recompute
        )


def build_norm(
    normalized_shape,
    eps=1e-5,
    elementwise_affine=True,
    *,
    use_rmsnorm=False,
    recompute=False,
    device=None,
    dtype=None,
):
    if use_rmsnorm:
        hs = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        return TritonRMSNorm(
            hs=hs,
            eps=eps,
            elementwise_affine=elementwise_affine,
            recompute=recompute,
        )
    return nn.LayerNorm(
        normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
        device=device,
        dtype=dtype,
    )
"""


class _V1TorchRMSNormLoader(SourceFileLoader):
    def get_code(self, fullname: str):
        source = self.get_data(self.path).decode("utf-8")
        if "import triton" not in source or "class TritonRMSNorm(" not in source:
            raise RuntimeError("Unsupported LimiX V1 RMSNorm implementation.")
        return compile(_V1_TORCH_RMSNORM_SOURCE, self.path, "exec")


class _V1TorchRMSNormFinder(MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname != "model.v1_0.operators.triton_rmsnorm":
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, SourceFileLoader):
            spec.loader = _V1TorchRMSNormLoader(fullname, spec.origin)
        return spec


@contextmanager
def optional_limix_triton() -> Iterator[None]:
    finder = _OptionalTritonFinder()
    v1_finder = None
    try:
        importlib.import_module("triton")
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
        v1_finder = _V1TorchRMSNormFinder()
    sys.meta_path.insert(0, finder)
    if v1_finder is not None:
        sys.meta_path.insert(0, v1_finder)
    try:
        ensure_limix_torch_norm_interface()
        yield
        ensure_limix_torch_norm_interface()
    finally:
        if v1_finder is not None:
            sys.meta_path.remove(v1_finder)
        sys.meta_path.remove(finder)
