import math

from torch import nn
from torch.distributions import Normal, Uniform, Exponential

from torchbayes import bnn
from torchbayes.distributions import ScaleMixtureNormal


class Model(bnn.BayesModel, nn.Sequential):
    """Architecture used by [Blundell'15] for the experiments on MNIST."""

    def __init__(self, in_shape, out_features, **kwargs):
        super().__init__(
            nn.Flatten(),

            bnn.BayesLinear(math.prod(in_shape), 1200),
            nn.ReLU(),

            bnn.BayesLinear(1200, 1200),
            nn.ReLU(),

            bnn.BayesLinear(1200, out_features),
        )

        self.init_parameters(**kwargs)

    def init_parameters(self, pi, sigma1, sigma2):
        self.apply(bnn.init_priors(
            ScaleMixtureNormal, pi=pi, sigma1=sigma1, sigma2=sigma2
        ))

        # FIXME: does not make sense as it is
        # self.apply(bnn.init_posteriors(
        #     Normal, loc=Normal(0.0, 0.01), scale=Normal(math.exp(-2), 0.01)
        # ))
