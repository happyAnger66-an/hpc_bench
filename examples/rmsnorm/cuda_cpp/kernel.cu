#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// RMSNorm CUDA kernel for hidden_size=4096 with bfloat16
__global__ void rmsnorm_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int batch_size,
    float eps) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    int tid = threadIdx.x;
    const int hidden_size = 4096;
    const int vec_size = 2048; // 4096 / 2 for bfloat162

    // Vectorized pointers
    const __nv_bfloat162* input_vec = reinterpret_cast<const __nv_bfloat162*>(input) + row * vec_size;
    const __nv_bfloat162* weight_vec = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* output_vec = reinterpret_cast<__nv_bfloat162*>(output) + row * vec_size;

    // Shared memory for reduction
    __shared__ float shared_sum[1024];

    // Each thread processes 2 elements (1 bfloat162)
    float thread_sum = 0.0f;
    for (int i = tid; i < vec_size; i += blockDim.x) {
        __nv_bfloat162 val = input_vec[i];
        float2 f2 = __bfloat1622float2(val);
        thread_sum += f2.x * f2.x + f2.y * f2.y;
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();

    // Full block reduction (power-of-2 blockDim). Stopping at s>32 drops shared_sum[32..63].
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = shared_sum[0] / static_cast<float>(hidden_size);
        shared_sum[0] = rsqrtf(mean + eps);
    }
    __syncthreads();

    float inv_rms = shared_sum[0];

    // Write output
    for (int i = tid; i < vec_size; i += blockDim.x) {
        __nv_bfloat162 in_val = input_vec[i];
        __nv_bfloat162 w_val = weight_vec[i];

        float2 in_f2 = __bfloat1622float2(in_val);
        float2 w_f2 = __bfloat1622float2(w_val);

        in_f2.x = in_f2.x * inv_rms * w_f2.x;
        in_f2.y = in_f2.y * inv_rms * w_f2.y;

        output_vec[i] = __float22bfloat162_rn(in_f2);
    }
}

// Host launcher
torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);

    TORCH_CHECK(hidden_size == 4096, "hidden_size must be 4096");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");

    auto output = torch::empty_like(input);

    dim3 block(1024);
    dim3 grid(batch_size);

    rmsnorm_kernel<<<grid, block>>>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        batch_size,
        eps
    );

    return output;
}

// Entry point for hpc_bench (DPS mode: writes to pre-allocated output)
void run(torch::Tensor input, torch::Tensor weight, float eps, torch::Tensor output) {
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);

    TORCH_CHECK(hidden_size == 4096, "hidden_size must be 4096");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bfloat16");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");

    dim3 block(1024);
    dim3 grid(batch_size);

    rmsnorm_kernel<<<grid, block>>>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        batch_size,
        eps
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "RMSNorm CUDA kernel (DPS mode)");
    m.def("forward", &rmsnorm_forward, "RMSNorm forward pass (returns output)");
}
