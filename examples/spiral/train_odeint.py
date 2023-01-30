import torch
from torch import nn

from neuralode.neuralode import NeuralODE
from neuralode.odesol import odesol_euler
from neuralode.utils import MultiLayerPerceptronStateTime
from examples.spiral.common import train_spiral, eval_spiral

MODEL_PATH = "./examples/spiral/spiral_odeint.pt"


class SpiralNeuralODE(nn.Module):
    def __init__(self, width=16, depth=3, activation=nn.Tanh, odesol=odesol_euler):
        """
        Neural ODE for modeling the spiral process.

        Args:
            width:      width of MLP model of the dynamics
            depth:      depth of the MLP model of the dynamics
            activation: activation function of the MLP model of the dynamics
            odesol:     ODE solver for the neural ODE
        """
        super(SpiralNeuralODE, self).__init__()

        self.width = width
        self.depth = depth
        self.activation = activation
        self.odesol = odesol

        self.dynamics = MultiLayerPerceptronStateTime(
            2, 2, self.width, self.depth, self.activation
        )
        self.odenet = NeuralODE(self.dynamics, self.odesol)

    def forward(self, z0, t):
        return self.odenet(z0, t)


if __name__ == "__main__":
    n_samples = 50
    n_traj = 50
    batch_size = 32
    epochs = 100

    model = SpiralNeuralODE()
    train_spiral(model, MODEL_PATH, n_samples, n_traj, batch_size, epochs)
    eval_spiral(model)
