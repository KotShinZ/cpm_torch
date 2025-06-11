import torch
import torch.nn as nn
import numpy as np
from cpm_torch.CPM import CPM, CPM_config
import sys
sys.path.append('/app')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple test case
config = CPM_config(
    l_A=1.0,
    l_L=1.0, 
    A_0=80.0,
    L_0=50.0,
    T=10.0,
    size=(16, 16),
)

# Create CPM instance
cpm = CPM(config, device=device)

# Create a simple NN that modifies energy
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, sources, targets):
        # Simple energy modification based on learnable parameter
        return self.param * torch.ones((sources.shape[0], 1), device=sources.device)

nn_model = SimpleNN().to(device)

# Create test input
B, H, W, C = 1, 16, 16, 34
x = torch.zeros((B, H, W, C), device=device)
# Set some initial cell IDs
with torch.no_grad():
    x[0, 7:9, 7:9, 0] = 1.0
    x[0, 8, 8, 1] = 1.0  # target
x.requires_grad = True

print("\nInitial tensor properties:")
print(f"x.requires_grad: {x.requires_grad}")
print(f"x.grad_fn: {x.grad_fn}")

# Test 1: Apply CPM step with NN
print("\n=== Test 1: CPM step with NN ===")
x_cpm = cpm.cpm_checkerboard_step_single_masked_func(x, dH_NN_func=nn_model)
print(f"After CPM step - x_cpm.requires_grad: {x_cpm.requires_grad}")
print(f"After CPM step - x_cpm.grad_fn: {x_cpm.grad_fn}")

# Try to compute loss on channel 0
loss_ch0 = x_cpm[:, :, :, 0].mean()
print(f"\nLoss on channel 0: {loss_ch0}")
print(f"loss_ch0.requires_grad: {loss_ch0.requires_grad}")
print(f"loss_ch0.grad_fn: {loss_ch0.grad_fn}")

# Test backward pass
try:
    loss_ch0.backward()
    print("\nBackward pass succeeded!")
    print(f"nn_model.param.grad: {nn_model.param.grad}")
except RuntimeError as e:
    print(f"\nBackward pass failed with error: {e}")

# Test 2: Direct modification of other channels
print("\n=== Test 2: Direct modification of other channels ===")
x2 = x.clone()
x2[:, :, :, 3:6] = x2[:, :, :, 3:6] + 0.5  # Modify channels 3-6
loss_ch3_6 = x2[:, :, :, 3:6].mean()
print(f"Loss on channels 3-6: {loss_ch3_6}")
print(f"loss_ch3_6.requires_grad: {loss_ch3_6.requires_grad}")
print(f"loss_ch3_6.grad_fn: {loss_ch3_6.grad_fn}")

# Test 3: Understanding the issue with channel 0
print("\n=== Test 3: Examining channel 0 operations ===")
# Let's trace through what happens to channel 0
x3 = torch.zeros((1, 4, 4, 2), device=device)
with torch.no_grad():
    x3[0, 1:3, 1:3, 0] = 1.0  # Set some cell IDs
x3.requires_grad = True

# Simulate what CPM does: discrete selection
source_id = torch.tensor([[1.0]], device=device)
target_id = torch.tensor([[0.0]], device=device)
logit = torch.tensor([[0.7]], device=device, requires_grad=True)

# This is what happens in CPM - sampling breaks gradients
rand = torch.rand_like(logit)
select = torch.relu(torch.sign(logit - rand))  # Discrete 0 or 1
new_id = torch.where(select > 0, source_id, target_id)

print(f"\nlogit.requires_grad: {logit.requires_grad}")
print(f"select: {select}, select.requires_grad: {select.requires_grad}")
print(f"new_id: {new_id}, new_id.requires_grad: {new_id.requires_grad}")