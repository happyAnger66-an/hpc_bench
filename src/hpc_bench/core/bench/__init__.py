"""Benchmarking utilities for correctness checking and timing."""

from .correctness import check_tensor_sanity, compute_error_stats, set_seed
from .timing import time_runnable
from .io import gen_inputs, allocate_outputs, normalize_outputs, load_safetensors
from .config import BenchmarkConfig

__all__ = [
    "check_tensor_sanity",
    "compute_error_stats",
    "set_seed",
    "time_runnable",
    "gen_inputs",
    "allocate_outputs",
    "normalize_outputs",
    "load_safetensors",
    "BenchmarkConfig",
]
