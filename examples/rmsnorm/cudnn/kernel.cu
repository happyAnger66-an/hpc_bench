// RMSNorm split: cuDNN cudnnOpTensor (x*x) + cudnnReduceTensor (sum over features),
// then a small CUDA kernel for rsqrt(mean+eps) and multiply by bf16 weights.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdexcept>
#include <string>
#include <torch/extension.h>

#define CUDNN_CHECK(expr)                                                      \
    do {                                                                       \
        cudnnStatus_t _st = (expr);                                            \
        if (_st != CUDNN_STATUS_SUCCESS) {                                    \
            throw std::runtime_error(std::string("cuDNN error: ") +           \
                                      cudnnGetErrorString(_st));               \
        }                                                                      \
    } while (0)

// Phase 2: inv_rms = rsqrt(mean(x^2)+eps), output = x * inv_rms * weight
// (row_sum_sq comes from cudnnReduceTensor over x^2 along the feature dim)
__global__ void rmsnorm_apply_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ row_sum_sq,
    int batch_size,
    int hidden_size,
    float eps) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    int tid = threadIdx.x;
    const int vec_size = hidden_size / 2;

    float inv_rms = rsqrtf(row_sum_sq[row] / static_cast<float>(hidden_size) + eps);

    const __nv_bfloat162* input_vec =
        reinterpret_cast<const __nv_bfloat162*>(input) + row * vec_size;
    const __nv_bfloat162* weight_vec =
        reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* output_vec =
        reinterpret_cast<__nv_bfloat162*>(output) + row * vec_size;

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

void run(torch::Tensor input, torch::Tensor weight, float eps, torch::Tensor output) {
    const int64_t batch_size = input.size(0);
    const int64_t hidden_size = input.size(1);

    TORCH_CHECK(hidden_size == 4096, "hidden_size must be 4096");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bfloat16");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && weight.is_contiguous(),
                "tensors must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t in_desc = nullptr;
    cudnnTensorDescriptor_t sq_desc = nullptr;
    cudnnTensorDescriptor_t sum_desc = nullptr;
    cudnnOpTensorDescriptor_t op_desc = nullptr;
    cudnnReduceTensorDescriptor_t reduce_desc = nullptr;
    void* reduce_ws = nullptr;
    size_t reduce_ws_bytes = 0;

    // FP32 x² + reduce: matches reference (input.to(float32).pow(2).mean(...));
    // BF16 reduce on many cuDNN builds returns CUDNN_STATUS_NOT_SUPPORTED.
    at::Tensor in_f32 = input.to(torch::kFloat32).contiguous();
    auto sq_f32 = torch::empty_like(in_f32);
    auto row_sum_sq =
        torch::empty({batch_size, 1, 1, 1}, torch::dtype(torch::kFloat32).device(input.device()));

    try {
        CUDNN_CHECK(cudnnCreate(&handle));
        CUDNN_CHECK(cudnnSetStream(handle, stream));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            static_cast<int>(batch_size), static_cast<int>(hidden_size), 1, 1));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&sq_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            sq_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            static_cast<int>(batch_size), static_cast<int>(hidden_size), 1, 1));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&sum_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            sum_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            static_cast<int>(batch_size), 1, 1, 1));

        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&op_desc));
        CUDNN_CHECK(cudnnSetOpTensorDescriptor(
            op_desc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

        const float alpha1 = 1.f;
        const float alpha2 = 1.f;
        const float beta_sq = 0.f;
        CUDNN_CHECK(cudnnOpTensor(
            handle, op_desc,
            &alpha1, in_desc, in_f32.data_ptr<float>(),
            &alpha2, in_desc, in_f32.data_ptr<float>(),
            &beta_sq, sq_desc, sq_f32.data_ptr<float>()));

        CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_desc));
        CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
            reduce_desc,
            CUDNN_REDUCE_TENSOR_ADD,
            CUDNN_DATA_FLOAT,
            CUDNN_PROPAGATE_NAN,
            CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES));

        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            handle, reduce_desc, sq_desc, sum_desc, &reduce_ws_bytes));
        if (reduce_ws_bytes > 0) {
            cudaError_t cerr = cudaMalloc(&reduce_ws, reduce_ws_bytes);
            TORCH_CHECK(cerr == cudaSuccess, "cudaMalloc reduce workspace: ",
                        cudaGetErrorString(cerr));
        }

        const float alpha_reduce = 1.f;
        const float beta_reduce = 0.f;
        CUDNN_CHECK(cudnnReduceTensor(
            handle, reduce_desc,
            nullptr, 0,
            reduce_ws, reduce_ws_bytes,
            &alpha_reduce,
            sq_desc, sq_f32.data_ptr<float>(),
            &beta_reduce,
            sum_desc, row_sum_sq.data_ptr<float>()));

        dim3 block(1024);
        dim3 grid(static_cast<unsigned>(batch_size));
        rmsnorm_apply_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            row_sum_sq.data_ptr<float>(),
            static_cast<int>(batch_size),
            static_cast<int>(hidden_size),
            eps);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "rmsnorm_apply_kernel launch failed");

    } catch (...) {
        if (reduce_ws) {
            cudaFree(reduce_ws);
        }
        if (reduce_desc) {
            cudnnDestroyReduceTensorDescriptor(reduce_desc);
        }
        if (op_desc) {
            cudnnDestroyOpTensorDescriptor(op_desc);
        }
        if (sum_desc) {
            cudnnDestroyTensorDescriptor(sum_desc);
        }
        if (sq_desc) {
            cudnnDestroyTensorDescriptor(sq_desc);
        }
        if (in_desc) {
            cudnnDestroyTensorDescriptor(in_desc);
        }
        if (handle) {
            cudnnDestroy(handle);
        }
        throw;
    }

    if (reduce_ws) {
        cudaFree(reduce_ws);
    }
    cudnnDestroyReduceTensorDescriptor(reduce_desc);
    cudnnDestroyOpTensorDescriptor(op_desc);
    cudnnDestroyTensorDescriptor(sum_desc);
    cudnnDestroyTensorDescriptor(sq_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroy(handle);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "RMSNorm: cuDNN OpTensor (x^2) + Reduce (sum) + CUDA scale");
}
