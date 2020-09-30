import math

from torch import nn
from torch.distributions import Normal, Uniform

import torchbayes.nn as bnn
from torchbayes.distributions import NormalScaleMixture


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
        prior = NormalScaleMixture(pi=1/2, sigma1=1e-1, sigma2=1e-7)
        self.apply(bnn.init_priors(prior))

        # FIXME: does not make sense as it is
        posterior_cls = Normal
        posterior_params = dict(loc=Normal(0.0, 0.01), scale=Uniform(-0.01, +0.01))
        self.apply(bnn.init_posteriors(posterior_cls, **posterior_params))

    def forward(self, input):
        return self.net(input)
