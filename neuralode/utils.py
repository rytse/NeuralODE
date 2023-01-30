import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron
    """

    def __init__(self, in_dim, out_dim, width=16, depth=4, activation=nn.ReLU):
        """
        Construct a MLP of the given dimensions.

        Args:
            in_dim:     input dimension
            out_dim:    output dimension
            width:      MLP width (dimension of each linear layer)
            depth:      MLP depth (number of layers)
            activation: activation function to use between linear layers
        """
        super(MultiLayerPerceptron, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.activation = activation

        self.layers = []
        self.layers.append(nn.Linear(self.in_dim, self.width))
        self.layers.append(self.activation())
        for _ in range(depth):
            self.layers.append(nn.Linear(self.width, self.width))
            self.layers.append(self.activation())
        self.layers.append(nn.Linear(self.width, self.out_dim))
        self.layers.append(self.activation())

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.mlp(x)


class MultiLayerPerceptronStateTime(nn.Module):
    """
    Multi-layer perceptron that takes in f(x, t) where x is the "input
    dimension" and t is a scalar.
    """

    def __init__(self, in_dim, out_dim, width=16, depth=4, activation=nn.ReLU):
        """
        Construct a multi-layer perceptron with input dimension in_dim+1 to
        take in in_dim state and scalar time.

        Inputs should be of the form
            x[state_dim, time_step_index]   size (in_dim, n_steps)
            t[time_step_index]              size (n_steps)
        """
        super(MultiLayerPerceptronStateTime, self).__init__()
        self.mlp = MultiLayerPerceptron(in_dim + 1, out_dim, width, depth, activation)

    def forward(self, x, t):
        batch_size, x_dim = x.shape

        if t.dim() == 0:
            tt = torch.unsqueeze(t, 0)
        else:
            tt = t

        t_batch = tt.expand(batch_size, len(tt))
        aug = torch.hstack((x, t_batch))

        return self.mlp(aug)
