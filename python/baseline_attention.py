import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class LobsterBaseline(nn.Module):
    def __init__(self, d_model=64, nhead=8):
        super().__init__()
        self.d_model = d_model
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.classifier = nn.Linear(d_model, 3) 

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, 8, D//8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, D)
        output = self.out_proj(attn_output)
        
        return self.classifier(output[:, -1, :])