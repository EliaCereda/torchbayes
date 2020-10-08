import math

from torch import nn
from torch.distributions import Normal, Uniform, Exponential

import torchbayes.nn as bnn
from torchbayes.distributions import ScaleMixtureNormal


class Model(bnn.BayesModel):
    """Architecture used by [Blundell'15] for the experiments on MNIST."""

    def __init__(self, in_shape, out_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            bnn.BayesLinear(math.prod(in_shape), 1200),
            nn.ReLU(),

            bnn.BayesLinear(1200, 1200),
            nn.ReLU(),

            bnn.BayesLinear(1200, out_features),
        )

        self.init_parameters()

    def init_parameters(self):
        self.apply(bnn.init_priors(
            ScaleMixtureNormal, pi=1/2, sigma1=math.exp(0), sigma2=math.exp(-6)
        ))

        # FIXME: does not make sense as it is
        # self.apply(bnn.init_posteriors(
        #     Normal, loc=Normal(0.0, 0.01), scale=Normal(math.exp(-2), 0.01)
        # ))

    def forward(self, input):
        return self.net(input)
