"""Trace data models for benchmark results."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field

from .base_model import BaseModelWithDocstrings


class EvaluationStatus(str, Enum):
    """Status of a workload evaluation."""

    PENDING = "PENDING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCORRECT_SHAPE = "INCORRECT_SHAPE"
    INCORRECT_DTYPE = "INCORRECT_DTYPE"
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    INVALID_REFERENCE = "INVALID_REFERENCE"
    REWARD_HACK = "REWARD_HACK"
    TIMEOUT = "TIMEOUT"
    BUILD_ERROR = "BUILD_ERROR"


class Correctness(BaseModelWithDocstrings):
    """Numerical correctness metrics."""

    max_absolute_error: float = 0.0
    max_relative_error: float = 0.0
    has_nan: bool = False
    has_inf: bool = False


class Performance(BaseModelWithDocstrings):
    """Performance metrics for a solution."""

    latency_ms: float
    reference_latency_ms: float = 0.0
    speedup_factor: float = 0.0


class Environment(BaseModelWithDocstrings):
    """Environment information for a benchmark run."""

    device_name: str
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None


class Evaluation(BaseModelWithDocstrings):
    """Complete evaluation result for a workload."""

    status: EvaluationStatus
    device: str
    log_path: Optional[str] = None
    message: Optional[str] = None
    correctness: Optional[Correctness] = None
    performance: Optional[Performance] = None
    environment: Optional[Environment] = None


class Trace(BaseModelWithDocstrings):
    """A single trace record for a benchmark evaluation."""

    definition: str
    solution: str
    workload: dict = Field(default_factory=dict)
    evaluation: Evaluation
    timestamp: Optional[str] = Field(default=None)
