"""Benchmark configuration for hpc_bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    seed: int = 42
    warmup_runs: int = 10
    iterations: int = 50
    timeout: int = 300
    compile_timeout: int = 300
    lock_clocks: bool = False
    benchmark_reference: bool = True
    verbose: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkConfig":
        """Create a BenchmarkConfig from a dictionary."""
        return cls(
            seed=data.get("seed", 42),
            warmup_runs=data.get("warmup_runs", 10),
            iterations=data.get("iterations", 50),
            timeout=data.get("timeout", 300),
            compile_timeout=data.get("compile_timeout", 300),
            lock_clocks=data.get("lock_clocks", False),
            benchmark_reference=data.get("benchmark_reference", True),
            verbose=data.get("verbose", False),
        )

    def to_dict(self) -> dict:
        """Convert BenchmarkConfig to a dictionary."""
        return {
            "seed": self.seed,
            "warmup_runs": self.warmup_runs,
            "iterations": self.iterations,
            "timeout": self.timeout,
            "compile_timeout": self.compile_timeout,
            "lock_clocks": self.lock_clocks,
            "benchmark_reference": self.benchmark_reference,
            "verbose": self.verbose,
        }
