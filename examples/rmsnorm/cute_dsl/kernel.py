"""RMSNorm: PyTorch reduction + CuTe DSL elementwise multiply (bf16 x bf16 -> output)."""

from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# cute.compile specializes on layouts; cache per (batch, hidden).
_compiled_mul: dict[tuple[int, int], object] = {}


@cute.kernel
def _mul_kernel(gA, gB, gC, cC, shape, thr_layout, val_layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    blk_coord = ((None, None), bidx)
    blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]
    blkCrd = cC[blk_coord]

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    tiled_cpy_A = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    tiled_cpy_B = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    tiled_cpy_C = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    thr_A = tiled_cpy_A.get_slice(tidx)
    thr_B = tiled_cpy_B.get_slice(tidx)
    thr_C = tiled_cpy_C.get_slice(tidx)

    thrA, thrB, thrC = thr_A.partition_S(blkA), thr_B.partition_S(blkB), thr_C.partition_S(blkC)
    frgA, frgB, frgC = (cute.make_fragment_like(t) for t in (thrA, thrB, thrC))

    thrCrd = thr_C.partition_S(blkCrd)
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in range(0, cute.size(frgPred), 1):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    cute.copy(copy_atom, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom, thrB, frgB, pred=frgPred)
    frgC.store(frgA.load() * frgB.load())
    cute.copy(copy_atom, frgC, thrC, pred=frgPred)


@cute.jit
def _elementwise_mul_2d(mA, mB, mC):
    # float32: 128 / 32 = 4 lanes (static; matches reference which keeps fp32 until final .to(bf16)).
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler=tiler_mn)

    _mul_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


@torch.no_grad()
def run(input, weight, eps, output):
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    # Match definition reference: (input * rstd) stays fp32; only cast after * weight.
    hidden_f32 = (input.to(torch.float32) * rstd).contiguous()

    b, h = input.shape
    w_row = weight.to(torch.float32).view(1, h).expand(b, h).contiguous()
    tmp_f32 = torch.empty((b, h), device=input.device, dtype=torch.float32)

    a_c = from_dlpack(hidden_f32)
    b_c = from_dlpack(w_row)
    o_c = from_dlpack(tmp_f32)

    key = (b, h)
    fn = _compiled_mul.get(key)
    if fn is None:
        fn = cute.compile(_elementwise_mul_2d, a_c, b_c, o_c)
        _compiled_mul[key] = fn
    fn(a_c, b_c, o_c)
    output.copy_(tmp_f32.to(torch.bfloat16))
