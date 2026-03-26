import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

# 1. Define the Baseline Model (From your Proposal)
class LobsterBaseline(nn.Module):
    def __init__(self, d_model=64, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, 3) 

    def forward(self, x):
        B, N, D = x.shape
        # standard SDPA baseline as planned in methodology [cite: 19-20]
        qkv = self.qkv_proj(x).reshape(B, N, 3, 8, D//8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # This triggers the O(N^2) memory bottleneck 
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, D)
        output = self.out_proj(attn_output)
        return self.classifier(output[:, -1, :])

# 2. Memory & Latency Profiling Function
def profile_memory(seq_lengths):
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    
    model = LobsterBaseline().to(device)
    model.eval()
    
    for N in seq_lengths:
        torch.cuda.empty_cache()
        # Mocking LOBSTER input data [cite: 43]
        x = torch.randn(1, N, 64).to(device)
        try:
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            
            with torch.no_grad():
                out = model(x)
                
            torch.cuda.synchronize() # Wait for GPU to finish for accurate timing
            end = time.time()
            
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2) # Convert to MB
            results.append((N, peak_mem, end-start))
            print(f"N={N:6} | Memory: {peak_mem:8.2f} MB | Time: {end-start:8.4f}s")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"N={N:6} | OOM (Out of Memory) at {N} sequences")
                break
            else:
                raise e
    return results

# 3. Execution & Visualization
# Testing the limits of your 3070 Ti (8GB) or Colab resources
test_lengths = [1000, 5000, 10000, 20000, 30000, 40000, 50000]
data = profile_memory(test_lengths)

if data:
    n_vals, mem_vals, time_vals = zip(*data)
    
    plt.figure(figsize=(10, 5))
    
    # Memory Plot
    plt.subplot(1, 2, 1)
    plt.plot(n_vals, mem_vals, marker='o', color='red')
    plt.title("Sequence Length vs VRAM (Baseline)")
    plt.xlabel("Sequence Length (N)")
    plt.ylabel("Peak VRAM (MB)")
    plt.grid(True)

    # Time Plot
    plt.subplot(1, 2, 2)
    plt.plot(n_vals, time_vals, marker='o', color='blue')
    plt.title("Sequence Length vs Latency (Baseline)")
    plt.xlabel("Sequence Length (N)")
    plt.ylabel("Time (seconds)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("baseline_performance.png") # Save for your report!
    plt.show()