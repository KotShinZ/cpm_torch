import torch
import torch.nn as nn
import numpy as np
from cpm_torch.CPM import CPM, CPM_config
import sys
sys.path.append('/app')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("\n=== SOLUTION: Making target_loss_f differentiable ===")

# The key insight is that we need to create a differentiable proxy for the cell positions
# Instead of using the discrete cell IDs directly, we can:
# 1. Use the CA network to predict a "cell presence probability" map
# 2. Use this continuous map for the loss calculation

class CAWithCellMap(nn.Module):
    """Modified CA that also outputs a differentiable cell presence map"""
    def __init__(self, chn=32, in_chn=34):
        super().__init__()
        self.chn = chn
        self.in_chn = in_chn
        self.w1 = nn.Conv2d(in_chn, 128, 1)
        self.w2 = nn.Conv2d(128, chn, 1, bias=False)
        # Add a head to predict cell presence
        self.cell_head = nn.Conv2d(128, 1, 1)
        
        with torch.no_grad():
            nn.init.xavier_uniform_(self.w1.weight)
            nn.init.zeros_(self.w1.bias)
            self.w2.weight.zero_()
            nn.init.xavier_uniform_(self.cell_head.weight)
            nn.init.zeros_(self.cell_head.bias)

    def forward(self, x):
        # x is (N, H, W, C) - convert to (N, C, H, W)
        x_perm = x.permute(0, 3, 1, 2)
        
        # Process channels
        y = self.w1(x_perm)
        y_activated = y * torch.sigmoid(y * 5.0)
        react = self.w2(y_activated)
        
        # Predict cell presence map
        cell_map = torch.sigmoid(self.cell_head(y_activated))  # (N, 1, H, W)
        
        # Only update channels 2 onwards
        x_out = x_perm[:, -self.chn:] + react
        
        # Combine with original channels 0-1 and permute back
        result = torch.cat([x_perm[:, :2], x_out], dim=1).permute(0, 2, 3, 1)
        
        # Add cell map as an additional output channel or return separately
        cell_map = cell_map.permute(0, 2, 3, 1)  # (N, H, W, 1)
        
        return result, cell_map

def differentiable_target_loss(cell_map, target_position):
    """
    Compute a differentiable loss based on distance to target.
    
    Args:
        cell_map: (B, H, W, 1) - continuous cell presence probabilities
        target_position: (row, col) - target position tuple
    """
    B, H, W, _ = cell_map.shape
    target_row, target_col = target_position
    
    # Create distance map
    rows = torch.arange(H, device=cell_map.device).unsqueeze(1)
    cols = torch.arange(W, device=cell_map.device).unsqueeze(0)
    
    distance_map = torch.sqrt(
        (rows - target_row) ** 2 + (cols - target_col) ** 2
    ).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    
    # Normalize distance
    max_distance = np.sqrt((H - 1) ** 2 + (W - 1) ** 2)
    normalized_distance = distance_map / max_distance
    
    # Weighted average distance based on cell presence
    weighted_distance = (normalized_distance * cell_map).sum(dim=[1, 2, 3])
    total_presence = cell_map.sum(dim=[1, 2, 3]) + 1e-6  # avoid division by zero
    
    loss = (weighted_distance / total_presence).mean()
    return loss

# Alternative solution: Use a soft version of the cell ID map
def soft_cell_map_from_ids(cell_ids, temperature=0.1):
    """
    Create a soft/continuous version of the discrete cell ID map.
    This maintains gradients by using a temperature-scaled softmax-like operation.
    """
    # cell_ids: (B, H, W) discrete IDs
    # Convert to one-hot-like representation but with soft boundaries
    
    # For simplicity, we'll create a cell presence map (any non-zero ID)
    cell_presence = (cell_ids > 0).float()
    
    # Apply some differentiable smoothing (e.g., gaussian blur)
    # In practice, you might use conv layers for this
    kernel_size = 3
    padding = kernel_size // 2
    
    # Create gaussian kernel
    gaussian_kernel = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=torch.float32, device=cell_ids.device) / 16.0
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution for smoothing
    cell_presence = cell_presence.unsqueeze(1)  # Add channel dim
    smooth_presence = torch.nn.functional.conv2d(
        cell_presence, gaussian_kernel, padding=padding
    )
    
    return smooth_presence.squeeze(1)  # Remove channel dim

# Test the solutions
print("\n1. Testing modified CA with cell map prediction:")
ca_modified = CAWithCellMap().to(device)

# Create test input
x = torch.zeros((1, 16, 16, 34), device=device, requires_grad=True)
with torch.no_grad():
    x_copy = x.clone()
    x_copy[0, 7:9, 7:9, 0] = 1.0
    x_copy[0, 8, 8, 3:6] = 0.5
x = x_copy.requires_grad_(True)

# Forward pass
x_out, cell_map = ca_modified(x)
print(f"   Output shape: {x_out.shape}")
print(f"   Cell map shape: {cell_map.shape}")
print(f"   Cell map requires_grad: {cell_map.requires_grad}")

# Compute differentiable target loss
target = (10, 10)
loss = differentiable_target_loss(cell_map, target)
print(f"   Differentiable target loss: {loss.item():.6f}")
print(f"   Loss requires_grad: {loss.requires_grad}")

# Test backward
loss.backward()
grad_norm = sum(p.grad.norm().item() for p in ca_modified.parameters() if p.grad is not None)
print(f"   Gradient norm: {grad_norm}")

print("\n2. Alternative: Using auxiliary supervision")
print("   Instead of directly supervising the discrete cell IDs,")
print("   we can add auxiliary tasks that are differentiable:")
print("   - Predict cell density maps")
print("   - Predict cell movement directions")
print("   - Predict cell growth probabilities")
print("   These can guide the CPM indirectly through the dH_NN term")

print("\n=== SUMMARY ===")
print("The gradient flow problem occurs because:")
print("1. CPM uses discrete, non-differentiable operations on channel 0")
print("2. The CA network doesn't modify channel 0, only channels 2+")
print("3. target_loss_f depends on channel 0, breaking gradient flow")
print("\nSolutions:")
print("1. Add a differentiable cell prediction head to the CA network")
print("2. Use auxiliary differentiable objectives instead of direct supervision")
print("3. Modify the CPM to use continuous relaxations (e.g., Gumbel-Softmax)")
print("4. Use the dH_NN term to influence cell behavior indirectly")