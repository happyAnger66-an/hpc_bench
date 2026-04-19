"""Correctness computation utilities for hpc_bench."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch

from hpc_bench.core.data import Correctness
from hpc_bench.core.data.workload import ToleranceSpec


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_tensor_sanity(
    sol_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    allow_negative_inf: bool = False,
) -> Optional[Correctness]:
    """Check for non-finite values and all-zeros output.

    Returns a Correctness describing the failure when either tensor
    contains inf/nan values, or the solution is all zeros while the
    reference is not. Returns None when both tensors look sane.
    """
    # Check for non-finite values
    ref_nonfinite = ~torch.isfinite(ref_tensor)
    sol_nonfinite = ~torch.isfinite(sol_tensor)

    if allow_negative_inf:
        # Exclude positions where both tensors are -inf
        both_neg_inf = (ref_tensor == float("-inf")) & (sol_tensor == float("-inf"))
        ref_nonfinite = ref_nonfinite & ~both_neg_inf
        sol_nonfinite = sol_nonfinite & ~both_neg_inf

    has_nonfinite = ref_nonfinite.any().item() or sol_nonfinite.any().item()
    if has_nonfinite:
        has_nan = (
            torch.isnan(sol_tensor).any().item() or torch.isnan(ref_tensor).any().item()
        )
        return Correctness(has_nan=has_nan, has_inf=not has_nan)

    # Check for all-zeros output when reference is non-zero
    ref_norm = torch.linalg.vector_norm(ref_tensor.to(torch.float32))
    if (
        ref_norm.item() > 0
        and torch.linalg.vector_norm(sol_tensor.to(torch.float32)).item() == 0
    ):
        abs_err = float(ref_norm.item())
        return Correctness(
            max_absolute_error=abs_err,
            max_relative_error=abs_err,
        )

    return None


def compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, tolerance: ToleranceSpec
) -> Tuple[Correctness, bool]:
    """Compute numerical error between output and reference.

    Returns (correctness, exceeds) where correctness is a Correctness
    carrying error metrics, and exceeds is True when the tolerance is violated.
    """
    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    allow_neg_inf = tolerance.allow_negative_inf

    # Check for infs/nans
    infs_nans = check_tensor_sanity(x, y, allow_negative_inf=allow_neg_inf)
    if infs_nans is not None:
        return infs_nans, True

    # Exclude matching -inf positions from error computation
    if allow_neg_inf:
        both_neg_inf = (x == float("-inf")) & (y == float("-inf"))
        finite_mask = ~both_neg_inf
        x = x[finite_mask]
        y = y[finite_mask]

    abs_error = torch.abs(x - y)
    total_elements = abs_error.numel()
    if total_elements == 0:
        return Correctness(), False

    max_abs = float(abs_error.max().item())

    # torch.allclose style: |a - b| <= atol + rtol * |b|
    tol_bound = tolerance.max_atol + tolerance.max_rtol * torch.abs(y)
    exceeds_tol_mask = (abs_error > tol_bound) | ~torch.isfinite(abs_error)

    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    # Check tolerance violation
    exceeds_tol = matched_ratio < tolerance.required_matched_ratio
    if tolerance.max_error_cap is not None and max_abs > tolerance.max_error_cap:
        exceeds_tol = True

    # Compute relative error
    rel_error = abs_error / torch.clamp(torch.abs(y), min=tolerance.max_atol)
    max_rel = float(rel_error.max().item())

    return Correctness(
        max_absolute_error=max_abs, max_relative_error=max_rel
    ), exceeds_tol
