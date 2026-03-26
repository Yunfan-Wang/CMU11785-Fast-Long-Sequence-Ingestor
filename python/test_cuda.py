import torch
import custom_attention_hpc

# Create a dummy tensor on GPU
x = torch.ones(10, device='cuda')

# Call dat CUSTOM CUDA function
y = custom_attention_hpc.forward(x, x, x)

print(f"Input: {x[0].item()}")
print(f"Output (should be Input + 1): {y[0].item()}")

if torch.allclose(y, x + 1):
    print("SUCCESS: current local CUDA pipeline is alive, as planned.")