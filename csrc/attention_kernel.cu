#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// A simple 'Hello World' CUDA kernel
__global__ void hello_world_kernel(float* out, const float* q, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = q[idx] + 1.0f; 
    }
}

// The C++ wrapper that PyTorch calls
torch::Tensor fast_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    auto out = torch::zeros_like(q);
    int size = q.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    hello_world_kernel<<<blocks, threads>>>(
        out.data_ptr<float>(), 
        q.data_ptr<float>(), 
        size
    );

    return out;
}