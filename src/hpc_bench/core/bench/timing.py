"""Timing utilities for kernel benchmarking."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any, Literal, Union

import torch


def _get_empty_cache_for_benchmark(device) -> torch.Tensor:
    """Create a buffer for clearing L2 cache before benchmark runs."""
    # Use 256MB buffer (double typical L2 cache)
    cache_size = 256 * 1024 * 1024 * 2
    return torch.empty(int(cache_size), dtype=torch.int8, device=device)


def _clear_cache(cache: torch.Tensor) -> None:
    """Clear the cache buffer by zeroing it."""
    cache.zero_()


def _summarize_statistics(
    times: list[float],
    return_mode: Literal["mean", "median", "all"],
) -> Union[float, list[float]]:
    """Summarize timing statistics based on return mode."""
    if return_mode == "all":
        return times
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"Unknown return_mode: {return_mode}")


def clone_args(args: list[Any]) -> list[Any]:
    """Clone tensor arguments to prevent cross-iteration data contamination."""
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


def bench_time_with_cuda_events(
    fn: Callable[..., Any],
    warmup: int = 10,
    rep: int = 100,
    setup: Callable[[], Any] | None = None,
    device: str = "cuda",
) -> list[float]:
    """Benchmark the runtime of the provided function using CUDA events.

    Derived from triton.testing.do_bench with modifications for
    explicit synchronization and L2 cache clearing.
    """
    cache = _get_empty_cache_for_benchmark(device)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    torch.cuda.synchronize()

    if setup is None:
        _fn = fn

        def fn(_):
            return _fn()

        def setup():
            return None

    # Warmup iterations
    for _ in range(warmup):
        args = setup()
        _clear_cache(cache)
        fn(args)

    # Timed iterations
    for i in range(rep):
        args = setup()
        _clear_cache(cache)
        start_events[i].record()
        fn(args)
        end_events[i].record()

    torch.cuda.synchronize()
    measured_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return measured_times


def time_runnable(
    fn: Any,
    inputs: list,
    outputs: list,
    device: str,
    warmup: int = 10,
    rep: int = 100,
    return_mode: Literal["mean", "median", "all"] = "median",
) -> Union[float, list[float]]:
    """Time the execution of a callable using CUDA events.

    Creates a ShiftingMemoryPoolAllocator from inputs and outputs
    so each timed iteration receives arguments with a unique data_ptr.
    """
    from .io import ShiftingMemoryPoolAllocator

    total_iterations = warmup + rep
    allocator = ShiftingMemoryPoolAllocator(inputs, outputs, total_iterations)

    with torch.cuda.device(device):
        times = bench_time_with_cuda_events(
            fn=lambda args: fn(*args),
            warmup=warmup,
            rep=rep,
            setup=allocator.get_unique_args,
            device=device,
        )
        return _summarize_statistics(times, return_mode)
