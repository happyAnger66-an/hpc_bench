import torch


def run(input, weight, eps, output):
    """RMSNorm kernel with DPS (Destination Passing Style).

    Args:
        input: [batch_size, hidden_size] tensor, dtype=bfloat16
        weight: [hidden_size] tensor, dtype=bfloat16
        eps: scalar float32
        output: pre-allocated output tensor [batch_size, hidden_size]

    Note:
        This is the reference implementation for RMSNorm:
        output = input / sqrt(mean(input^2) + eps) * weight
    """
    # Convert to float32 for numerical stability
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    hidden_states = input * rstd
    output[:] = (hidden_states * weight).to(input.dtype)
