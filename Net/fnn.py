"""A FNN with 3 Linear Layers"""
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,  output_dim):
        super(FNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1) ,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim2, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.layers(x)
        return out