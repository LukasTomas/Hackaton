import torch
from torch import nn

from hyperparams import NN_INPUT_SIZE

class LinearNN(nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(NN_INPUT_SIZE, 32),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 32),
            # nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x