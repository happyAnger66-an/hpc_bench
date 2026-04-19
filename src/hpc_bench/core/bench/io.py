"""Input generation and output allocation utilities for hpc_bench."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from hpc_bench.core.data import Definition, Workload
from hpc_bench.core.data.workload import CustomInput, SafetensorsInput, ScalarInput
from hpc_bench.core.data.dtypes import dtype_str_to_torch_dtype


def _cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to FP4 E2M1 and pack into uint8 (2 FP4 values per byte)."""
    result = torch.zeros_like(x, dtype=torch.uint8)

    # Positive values
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    # Negative values
    result[(x >= -0.25) & (x < 0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    # Pack two FP4 values into one byte along cols dimension
    packed = result[..., ::2] + result[..., 1::2] * 16
    return packed.view(torch.float4_e2m1fn_x2)


def _rand_tensor(
    shape: List[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Generate a random tensor with appropriate values for the dtype."""
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device)

    # Low-precision floats
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)
    elif dtype == torch.float4_e2m1fn_x2:
        return _cast_to_fp4x2(torch.randn(shape, dtype=torch.float32, device=device))

    # Booleans
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)

    # Integers
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        ranges = {
            torch.int8: (-128, 128),
            torch.int16: (-1024, 1024),
            torch.int32: (-1024, 1024),
            torch.int64: (-1024, 1024),
        }
        low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=device, dtype=dtype)

    raise ValueError(f"Unsupported random dtype: {dtype}")


# Heuristic tensor generation helpers


def _is_norm_weight(name: str) -> bool:
    """Check if tensor is a normalization weight."""
    if name == "norm_weight":
        return True
    if name.endswith("_weight"):
        prefix = name[: -len("_weight")]
        if prefix.endswith(("_norm", "_layernorm", "layernorm")):
            return True
        stripped = prefix.rstrip("0123456789")
        if stripped and stripped.endswith(("norm", "layernorm")):
            return True
    return False


def _is_norm_bias(name: str) -> bool:
    """Check if tensor is a normalization bias."""
    if name == "norm_bias":
        return True
    if name.endswith("_bias"):
        prefix = name[: -len("_bias")]
        if prefix.endswith(("_norm", "_layernorm", "layernorm")):
            return True
        stripped = prefix.rstrip("0123456789")
        if stripped and stripped.endswith(("norm", "layernorm")):
            return True
    return False


def _is_weight_matrix(name: str, shape: tuple[int, ...]) -> bool:
    """Check if tensor is a weight matrix."""
    if len(shape) < 2:
        return False
    weight_suffixes = (
        "_weight",
        "_weights",
        "_proj",
        "_projs",
        "_proj_weight",
        "_proj_weights",
        "_weight_matrix",
    )
    if name.endswith(weight_suffixes) or name == "weight":
        return True
    stripped = name.rstrip("0123456789")
    if stripped and stripped in ("weight",) or stripped.endswith(weight_suffixes):
        return True
    return False


def _generate_heuristic_tensor(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    description: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Generate a tensor using heuristics based on the input name."""
    if not dtype.is_floating_point:
        return None

    # Low-precision floats don't support some ops
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2):
        return None

    if _is_norm_weight(name):
        return torch.ones(shape, dtype=dtype, device=device)

    if _is_norm_bias(name):
        return torch.zeros(shape, dtype=dtype, device=device)

    if _is_weight_matrix(name, shape):
        fan_in = shape[-1]
        return torch.randn(shape, dtype=dtype, device=device) / math.sqrt(fan_in)

    return None


def load_safetensors(
    definition: Definition,
    workload: Workload,
    blob_roots: Optional[List[Path]] = None,
) -> Dict[str, torch.Tensor]:
    """Load safetensors inputs for a workload."""
    try:
        import safetensors.torch as st
    except Exception as e:
        raise RuntimeError(
            "safetensors is not available in the current environment"
        ) from e

    expected = definition.get_input_shapes(workload.axes)

    safe_tensors: Dict[str, torch.Tensor] = {}
    loaded_files: Dict[str, Dict[str, torch.Tensor]] = {}

    for name, input_spec in workload.inputs.items():
        if not isinstance(input_spec, SafetensorsInput):
            continue

        path = input_spec.path
        if not Path(path).is_absolute() and blob_roots:
            for root in blob_roots:
                candidate = root / path
                if candidate.exists():
                    path = str(candidate)
                    break

        path = str(Path(path).resolve())

        if path not in loaded_files:
            loaded_files[path] = st.load_file(path)
        tensors = loaded_files[path]

        if input_spec.tensor_key not in tensors:
            raise ValueError(f"Missing key '{input_spec.tensor_key}' in '{path}'")
        t = tensors[input_spec.tensor_key]

        if tuple(t.shape) != expected[name]:
            raise ValueError(
                f"'{name}' expected {expected[name]}, got {list(t.shape)}"
            )

        expect_dtype = dtype_str_to_torch_dtype(definition.inputs[name].dtype)
        if t.dtype != expect_dtype:
            raise ValueError(f"'{name}' expected {expect_dtype}, got {t.dtype}")

        try:
            t = t.contiguous().pin_memory()
        except Exception:
            t = t.contiguous()
        safe_tensors[name] = t

    return safe_tensors


def gen_inputs(
    definition: Definition,
    workload: Workload,
    device: str,
    safe_tensors: Optional[Dict[str, torch.Tensor]] = None,
    custom_inputs_fn: Optional[Any] = None,
) -> List[Any]:
    """Generate input tensors in definition order."""
    shapes = definition.get_input_shapes(workload.axes)
    dev = torch.device(device)
    out: List[Any] = []
    custom_tensors = None

    # Regenerate custom tensors on the fly when a factory is provided
    if custom_inputs_fn is not None:
        axes_and_scalars = {
            **definition.get_resolved_axes_values(workload.axes),
            **workload.get_scalar_inputs(),
        }
        custom_tensors = custom_inputs_fn(axes_and_scalars, dev)

    for name, spec in definition.inputs.items():
        dtype = dtype_str_to_torch_dtype(spec.dtype)
        inp_spec = workload.inputs.get(name)

        if isinstance(inp_spec, SafetensorsInput):
            if safe_tensors is None or name not in safe_tensors:
                raise RuntimeError(f"Missing required safetensors input '{name}'")
            t_cpu = safe_tensors[name]
            out.append(t_cpu.to(device=dev, non_blocking=True))
        elif isinstance(inp_spec, ScalarInput):
            out.append(inp_spec.value)
        elif isinstance(inp_spec, CustomInput):
            if custom_tensors is None or name not in custom_tensors:
                raise RuntimeError(
                    f"CustomInput for '{name}' must be pre-generated"
                )
            val = custom_tensors[name]
            if isinstance(val, torch.Tensor):
                out.append(val.to(device=dev, non_blocking=True))
            else:
                out.append(val)
        else:  # random
            shape = shapes[name]

            if shape is None:
                value = _rand_tensor((), dtype, dev).item()
            else:
                value = _generate_heuristic_tensor(
                    name, tuple(shape), dtype, dev, spec.description
                )
                if value is None:
                    value = _rand_tensor(shape, dtype, dev)

            out.append(value)

    return out


def normalize_outputs(
    out: Any,
    *,
    device: torch.device,
    output_names: List[str],
    output_dtypes: Dict[str, torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Normalize function output to a dictionary of tensors."""

    def to_tensor(name: str, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.to(device) if v.device != device else v
        dtype = output_dtypes[name]
        return torch.tensor(v, dtype=dtype, device=device)

    if isinstance(out, dict):
        return {k: to_tensor(k, v) for k, v in out.items() if k in output_dtypes}

    if isinstance(out, torch.Tensor):
        if len(output_names) != 1:
            raise RuntimeError(
                "Single Tensor returned but multiple outputs are defined"
            )
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (int, float, bool)):
        if len(output_names) != 1:
            raise RuntimeError("Scalar returned but multiple outputs are defined")
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (tuple, list)):
        if len(out) != len(output_names):
            raise RuntimeError(
                f"Tuple/list has {len(out)} elements but {len(output_names)} outputs expected"
            )
        return {name: to_tensor(name, val) for name, val in zip(output_names, out)}

    raise RuntimeError(
        "Unexpected return type; must be Tensor, scalar, or dict[name -> Tensor/scalar]"
    )


def allocate_outputs(
    definition: Definition, resolved_axes: dict[str, int], device: str
) -> List[torch.Tensor]:
    """Allocate output tensors based on definition and resolved axis values."""
    output_shapes = list(definition.get_output_shapes(resolved_axes).values())
    dtypes = definition.torch_output_dtypes
    return [
        torch.zeros(shape, dtype=dtype, device=device)
        for shape, dtype in zip(output_shapes, dtypes)
    ]


class ShiftingMemoryPoolAllocator:
    """Pre-allocated memory pool that provides unique data_ptr per iteration.

    Allocates a buffer only slightly larger than the input tensors
    (overhead ~ total_iterations x 256 bytes per tensor). The source
    data is retained and copied into an advancing offset of the pool on
    each call to get_unique_args, so every iteration sees a
    distinct data_ptr while VRAM usage stays near 1x input size.
    """

    _POOL_ALIGNMENT = 256

    def __init__(
        self,
        inputs: List[Any],
        outputs: List[torch.Tensor],
        total_iterations: int,
    ) -> None:
        self._call_idx = 0
        self._total_iterations = total_iterations
        self._input_entries: List[Dict[str, Any]] = []
        self._output_entries: List[Dict[str, Any]] = []

        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                self._input_entries.append({"scalar": inp})
                continue
            self._input_entries.append(self._make_pool_entry(inp, total_iterations))

        for out in outputs:
            self._output_entries.append(self._make_pool_entry(out, total_iterations))

    @staticmethod
    def _storage_span(tensor: torch.Tensor) -> int:
        """Number of contiguous storage elements spanned by tensor."""
        if tensor.numel() == 0:
            return 0
        span = 1
        for s, st in zip(tensor.shape, tensor.stride()):
            if s > 1:
                span += (s - 1) * st
        return span

    @classmethod
    def _make_pool_entry(
        cls, tensor: torch.Tensor, total_iterations: int
    ) -> Dict[str, Any]:
        if any(st < 0 for st in tensor.stride()):
            tensor = tensor.contiguous()

        shape = tuple(tensor.shape)
        strides = tensor.stride()
        storage_span = cls._storage_span(tensor)
        elem_size = tensor.element_size()

        stride_numel = max(1, cls._POOL_ALIGNMENT // elem_size)
        pool_numel = storage_span + (total_iterations - 1) * stride_numel
        pool = torch.empty(pool_numel, dtype=tensor.dtype, device=tensor.device)

        source = tensor.as_strided((storage_span,), (1,))

        return {
            "pool": pool,
            "source": source,
            "shape": shape,
            "strides": strides,
            "storage_span": storage_span,
            "stride_numel": stride_numel,
        }

    def get_unique_args(self) -> List[Any]:
        """Copy source data into the next pool offset and return views."""
        if self._call_idx >= self._total_iterations:
            raise RuntimeError(
                f"ShiftingMemoryPoolAllocator exhausted: called {self._call_idx + 1} "
                f"times but was allocated for {self._total_iterations} iterations"
            )

        result: List[Any] = []
        idx = self._call_idx

        for entry in self._input_entries:
            if "scalar" in entry:
                result.append(entry["scalar"])
                continue

            start = idx * entry["stride_numel"]
            entry["pool"].narrow(0, start, entry["storage_span"]).copy_(entry["source"])
            result.append(
                entry["pool"].as_strided(entry["shape"], entry["strides"], start)
            )

        for entry in self._output_entries:
            start = idx * entry["stride_numel"]
            entry["pool"].narrow(0, start, entry["storage_span"]).zero_()
            result.append(
                entry["pool"].as_strided(entry["shape"], entry["strides"], start)
            )

        self._call_idx += 1
        return result
