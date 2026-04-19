"""RMSNorm: PyTorch reduction + cuTile elementwise multiply."""

from __future__ import annotations

from math import ceil

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def mul_kernel(A, B, C, tm: ConstInt, tn: ConstInt):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    zero_pad = ct.PaddingMode.ZERO
    a = ct.load(A, index=(bidx, bidy), shape=(tm, tn), padding_mode=zero_pad)
    b = ct.load(B, index=(bidx, bidy), shape=(tm, tn), padding_mode=zero_pad)
    ct.store(C, index=(bidx, bidy), tile=(a * b))


@torch.no_grad()
def run(input, weight, eps, output):
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    hidden = (input * rstd).to(input.dtype)

    b, h = input.shape
    w_row = weight.view(1, h).expand(b, h).contiguous()
    h_c = hidden.contiguous()

    tm, tn = 32, 256
    grid = (ceil(b / tm), ceil(h / tn), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        mul_kernel,
        (h_c, w_row, output, tm, tn),
    )
