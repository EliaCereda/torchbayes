import math

from torch import nn
from torch.distributions import Normal, Uniform, Exponential

from torchbayes import bnn
from torchbayes.distributions import ScaleMixtureNormal


class Model(bnn.BayesModel, nn.Sequential):
    """Architecture used by [Blundell'15] for the experiments on MNIST."""

    def __init__(self, in_shape, out_features, approach, **kwargs):
        if approach == 'traditional':
            Linear = nn.Linear
        elif approach == 'bnn':
            Linear = bnn.BayesLinear
        else:
            raise ValueError(f"Unsupported approach '{approach}'.")

        super().__init__(
            nn.Flatten(),

            Linear(math.prod(in_shape), 1200),
            nn.ReLU(),

            Linear(1200, 1200),
            nn.ReLU(),

            Linear(1200, out_features),
        )

        self.init_parameters(**kwargs)

    def init_parameters(self, prior, sigma, pi, sigma1, sigma2):
        if prior == 'normal':
            self.apply(bnn.init_priors(
                Normal, loc=0.0, scale=sigma
            ))
        elif prior == 'scale_mixture':
            self.apply(bnn.init_priors(
                ScaleMixtureNormal, pi=pi, sigma1=sigma1, sigma2=sigma2
            ))
        else:
            raise ValueError(f"Unsupported prior distribution '{prior}'.")

        # FIXME: does not make sense as it is
        # self.apply(bnn.init_posteriors(
        #     Normal, loc=Normal(0.0, 0.01), scale=Normal(math.exp(-2), 0.01)
        # ))
