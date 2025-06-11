import torch
import torch.nn as nn
import numpy as np
from cpm_torch.CPM import CPM, CPM_config
import sys
sys.path.append('/app')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Key insight: Let's trace through exactly what happens in the training loop

# 1. First, let's understand what the CA network does
class CA(nn.Module):
    """Simplified version of the CA network from the training notebook"""
    def __init__(self, chn=32, in_chn=34):
        super().__init__()
        self.chn = chn
        self.in_chn = in_chn
        self.w1 = nn.Conv2d(in_chn, 128, 1)
        self.w2 = nn.Conv2d(128, chn, 1, bias=False)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.w1.weight)
            nn.init.zeros_(self.w1.bias)
            self.w2.weight.zero_()

    def forward(self, x):
        # x is (N, H, W, C) - convert to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Process channels
        y = self.w1(x)
        y = y * torch.sigmoid(y * 5.0)
        react = self.w2(y)
        
        # Only update channels 2 onwards (keeping channels 0-1 unchanged)
        x_out = x[:, -self.chn:] + react
        
        # Combine with original channels 0-1 and permute back
        result = torch.cat([x[:, :2], x_out], dim=1).permute(0, 2, 3, 1)
        return result

# 2. Create test scenario
print("\n=== Understanding the gradient flow problem ===")

# Create models
ca = CA().to(device)
config = CPM_config(size=(16, 16))
cpm = CPM(config, device=device)

# Create test input
x = torch.zeros((1, 16, 16, 34), device=device)
with torch.no_grad():
    x[0, 7:9, 7:9, 0] = 1.0  # Initial cell
    x[0, 10, 10, 1] = 1.0    # Target location
    # Add some values to other channels
    x[..., 3:6] = 0.1
x.requires_grad = True

print("\n1. Initial state:")
print(f"   x shape: {x.shape}")
print(f"   x.requires_grad: {x.requires_grad}")
print(f"   Channel 0 (cell IDs) sum: {x[..., 0].sum().item()}")
print(f"   Channels 3-6 (RGB) mean: {x[..., 3:6].mean().item()}")

# 3. Apply CPM step (no gradient flow through this)
print("\n2. After CPM step:")
x_cpm = cpm.cpm_checkerboard_step_single_masked_func(x, dH_NN_func=None)
print(f"   x_cpm.requires_grad: {x_cpm.requires_grad}")
print(f"   x_cpm.grad_fn: {x_cpm.grad_fn}")
print(f"   Channel 0 sum: {x_cpm[..., 0].sum().item()}")

# 4. Apply CA step (this should have gradients)
print("\n3. After CA step:")
# First prepare the input for CA
binary_id_channel = torch.where(x_cpm[..., 0:1] > 0, 1.0, 0.0)
other_channels = x_cpm[..., 1:]
x_ca_input = torch.cat([binary_id_channel, other_channels], dim=-1)
x_ca = ca(x_ca_input)
print(f"   x_ca.requires_grad: {x_ca.requires_grad}")
print(f"   x_ca.grad_fn: {x_ca.grad_fn}")

# 5. Test different loss functions
print("\n4. Testing loss functions:")

# Style loss (on channels 3-6)
style_loss = (x_ca[..., 3:6] + 0.5).mean()
print(f"   Style loss: {style_loss.item():.6f}")
print(f"   style_loss.requires_grad: {style_loss.requires_grad}")

# Target loss (on channel 0)
target_loss = (x_ca[..., 0] > 0).float().mean()
print(f"   Target loss: {target_loss.item():.6f}")
print(f"   target_loss.requires_grad: {target_loss.requires_grad}")

# 6. Try backward pass on both
print("\n5. Backward pass tests:")

# Style loss backward
ca.zero_grad()
style_loss.backward(retain_graph=True)
style_grad_norm = sum(p.grad.norm().item() for p in ca.parameters() if p.grad is not None)
print(f"   Style loss - CA gradient norm: {style_grad_norm}")

# Target loss backward
ca.zero_grad()
try:
    target_loss.backward()
    target_grad_norm = sum(p.grad.norm().item() for p in ca.parameters() if p.grad is not None)
    print(f"   Target loss - CA gradient norm: {target_grad_norm}")
except RuntimeError as e:
    print(f"   Target loss backward failed: {e}")

# 7. The key issue
print("\n=== KEY INSIGHT ===")
print("The problem is that:")
print("1. CPM modifies channel 0 (cell IDs) through non-differentiable operations")
print("2. CA network only modifies channels 2+ (it passes through channels 0-1 unchanged)")
print("3. target_loss_f uses channel 0, which has no gradient path to CA parameters")
print("4. style_loss_f uses channels 3-6, which ARE modified by the differentiable CA network")

# 8. Demonstrate the issue more clearly
print("\n=== Demonstrating the gradient blocking ===")
# Create a simple parameter
param = nn.Parameter(torch.tensor(1.0, device=device))

# Path 1: Direct differentiable operation
x1 = param * torch.ones((1, 4, 4, 1), device=device)
loss1 = x1.mean()
loss1.backward()
print(f"Direct path - param.grad: {param.grad}")

# Path 2: Through non-differentiable operation (like CPM)
param.grad = None
x2 = param * torch.ones((1, 4, 4, 1), device=device)
# Simulate CPM's discrete selection
with torch.no_grad():
    x2_discrete = torch.where(x2 > 0.5, torch.ones_like(x2), torch.zeros_like(x2))
loss2 = x2_discrete.mean()
try:
    loss2.backward()
    print(f"Through discrete op - param.grad: {param.grad}")
except RuntimeError as e:
    print(f"Through discrete op - backward failed: {e}")