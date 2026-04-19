"""RMSNorm via Triton (hidden_size=4096, bfloat16)."""

import torch
import triton
import triton.language as tl

_HIDDEN = 4096


@triton.jit
def _rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    stride_m,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    row_off = row * stride_m

    x = tl.load(input_ptr + row_off + offs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)

    x_f32 = x.to(tl.float32)
    w_f32 = w.to(tl.float32)
    var = tl.sum(x_f32 * x_f32, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    out_f32 = x_f32 * rstd * w_f32
    out = out_f32.to(tl.bfloat16)
    tl.store(output_ptr + row_off + offs, out, mask=mask)


def run(input, weight, eps, output):
    """RMSNorm with DPS. Same contract as PyTorch example."""
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("input and weight must be bfloat16")
    if output.dtype != torch.bfloat16:
        raise TypeError("output must be bfloat16")
    if not input.is_cuda or not weight.is_cuda or not output.is_cuda:
        raise RuntimeError("Triton RMSNorm expects CUDA tensors")
    if input.shape[1] != _HIDDEN or weight.shape != (_HIDDEN,) or output.shape != input.shape:
        raise ValueError(f"expected input [*, {_HIDDEN}], weight [{_HIDDEN}], matching output")

    n_rows, n_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _rmsnorm_kernel[grid](
        input,
        weight,
        output,
        input.stride(0),
        n_cols,
        float(eps),
        BLOCK_SIZE=BLOCK_SIZE,
    )
