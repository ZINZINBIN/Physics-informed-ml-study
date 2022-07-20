from phi.torch.flow import math
import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, n_layers : int, hidden_dims : int, output_dims : 1):
        super(Network, self).__init__()
        self.n_layers = n_layers
        self.input_dims = 2
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.layer = nn.Sequential()

        for idx in range(self.n_layers):
            if idx == 0:
                in_channels = self.input_dims
            else:
                in_channels = self.hidden_dims
            
            if idx == self.n_layers - 1:
                out_channels = self.output_dims
            else:
                out_channels = self.hidden_dims

            self.layer.add_module(
                'linear_layer%d'%idx, nn.Linear(in_channels, out_channels)
            )

            self.layer.add_module(
                'relu_layer%d'%idx, nn.ReLU(),
            )

    def forward(self, x : torch.Tensor, t : torch.Tensor):
        x.requires_grad = True
        t.requires_grad = True
        inputs = torch.stack((x,t), dim = len(x.size()))
        return self.layer(inputs)