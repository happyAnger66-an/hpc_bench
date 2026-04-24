"""RMSNorm via TileLang (hidden_size=4096, bfloat16)."""

import tilelang
import torch
from tilelang import language as T

_HIDDEN = 4096
_THREADS = 1024
_TILES = _HIDDEN // _THREADS

_kernel_cache: dict[float, object] = {}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _make_rmsnorm_kernel(rms_eps: float):
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def rmsnorm_kernel(
        inp: T.Tensor[(num_tokens, _HIDDEN), T.bfloat16],
        wt: T.Tensor[(_HIDDEN,), T.bfloat16],
        out: T.Tensor[(num_tokens, _HIDDEN), T.bfloat16],
    ):
        with T.Kernel(num_tokens, threads=_THREADS) as row:
            partial = T.alloc_fragment((_THREADS,), T.float32)
            T.clear(partial)
            for t in T.serial(_TILES):
                for i in T.Parallel(_THREADS):
                    j = t * _THREADS + i
                    v = T.float32(inp[row, j])
                    partial[i] = partial[i] + v * v
            partial_2d = T.reshape(partial, (_THREADS, 1))
            sumsq = T.alloc_fragment((1,), T.float32)
            T.reduce_sum(partial_2d, sumsq, dim=0)
            rstd = T.alloc_fragment((1,), T.float32)
            rstd[0] = T.rsqrt(sumsq[0] / T.float32(_HIDDEN) + T.float32(rms_eps))
            for t in T.serial(_TILES):
                for i in T.Parallel(_THREADS):
                    j = t * _THREADS + i
                    xv = T.float32(inp[row, j])
                    wv = T.float32(wt[j])
                    out[row, j] = T.cast(xv * rstd[0] * wv, T.bfloat16)

    return rmsnorm_kernel


def run(input: torch.Tensor, weight: torch.Tensor, eps, output: torch.Tensor) -> None:
    """RMSNorm with DPS; matches ``examples/rmsnorm`` contract."""
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("input and weight must be bfloat16")
    if output.dtype != torch.bfloat16:
        raise TypeError("output must be bfloat16")
    if not input.is_cuda or not weight.is_cuda or not output.is_cuda:
        raise RuntimeError("TileLang RMSNorm expects CUDA tensors")
    if (
        input.shape[1] != _HIDDEN
        or weight.shape != (_HIDDEN,)
        or output.shape != input.shape
    ):
        raise ValueError(
            f"expected input [*, {_HIDDEN}], weight [{_HIDDEN}], matching output"
        )

    eps_f = float(eps)
    kernel = _kernel_cache.get(eps_f)
    if kernel is None:
        kernel = _make_rmsnorm_kernel(eps_f)
        _kernel_cache[eps_f] = kernel
    kernel(input, weight, output)
