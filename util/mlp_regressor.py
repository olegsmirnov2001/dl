import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.layers(x).flatten()
