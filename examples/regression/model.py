import math

from torch import nn
from torch.distributions import Normal

from torchbayes import bnn


class Model(bnn.BayesModel, nn.Sequential):
    def __init__(self, in_size, out_features):
        Linear = bnn.BayesLinear
        # Linear = nn.Linear

        super().__init__(
            nn.BatchNorm1d(in_size),

            Linear(in_size, 5),
            nn.ReLU(),

            Linear(5, 5),
            nn.ReLU(),

            Linear(5, out_features),
        )

        self.init_parameters()

    def init_parameters(self, sigma=-1):
        self.apply(bnn.init_priors(
            Normal, loc=0.0, scale=math.exp(-sigma)
        ))
