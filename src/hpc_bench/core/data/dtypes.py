"""Data type utilities for hpc_bench."""

from __future__ import annotations

import torch

from .definition import DType


# Mapping from DType enum to torch.dtype
_DTYPE_TO_TORCH: dict[DType, torch.dtype] = {
    DType.FLOAT64: torch.float64,
    DType.FLOAT32: torch.float32,
    DType.FLOAT16: torch.float16,
    DType.BFLOAT16: torch.bfloat16,
    DType.FLOAT8_E4M3FN: torch.float8_e4m3fn,
    DType.FLOAT8_E5M2: torch.float8_e5m2,
    DType.FLOAT4_E2M1: torch.float32,  # FP4 is not directly supported, use float32
    DType.FLOAT4_E2M1FN_X2: torch.float4_e2m1fn_x2,
    DType.INT64: torch.int64,
    DType.INT32: torch.int32,
    DType.INT16: torch.int16,
    DType.INT8: torch.int8,
    DType.BOOL: torch.bool,
}

# Reverse mapping
_TORCH_TO_DTYPE: dict[torch.dtype, DType] = {v: k for k, v in _DTYPE_TO_TORCH.items()}


def dtype_str_to_torch_dtype(dtype_str: str | DType) -> torch.dtype:
    """Convert a dtype string or DType enum to torch.dtype."""
    if isinstance(dtype_str, DType):
        return _DTYPE_TO_TORCH[dtype_str]
    dtype = DType(dtype_str)
    return _DTYPE_TO_TORCH[dtype]


def torch_dtype_to_dtype_str(dtype: torch.dtype) -> str:
    """Convert a torch.dtype to dtype string."""
    if dtype not in _TORCH_TO_DTYPE:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return _TORCH_TO_DTYPE[dtype].value
