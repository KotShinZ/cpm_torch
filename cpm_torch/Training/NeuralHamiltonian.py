import torch

class NeuralHamiltonian(torch.nn.Module):
    """
    Neural Hamiltonian Network (NHN) for learning Hamiltonian dynamics.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(NeuralHamiltonian, self).__init__()
        self.cnn1 = torch.nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.cnn2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(hidden_dim * 32 * 32, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x