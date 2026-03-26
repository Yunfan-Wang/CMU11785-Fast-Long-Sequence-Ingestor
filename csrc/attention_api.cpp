#include <torch/extension.h>

// Forward declaration of CUDA function
torch::Tensor fast_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

// Bind C++ function to a Python name 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fast_attention_forward, "Custom Tiled Attention Forward (CUDA)");
}