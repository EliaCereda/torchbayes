import math

from torch import nn
from torch.distributions import Normal, Uniform, Exponential

from torchbayes import bnn
from torchbayes.distributions import ScaleMixtureNormal


class Model(bnn.BayesModel, nn.Sequential):
    """Architecture used by [Blundell'15] for the experiments on MNIST."""

    def __init__(self, in_shape, out_features, **kwargs):
        approach = kwargs.get('approach')
        if approach == 'bnn' or approach is None:
            Linear = bnn.BayesLinear
        elif approach == 'traditional':
            Linear = nn.Linear
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

    def init_parameters(self, approach, prior, sigma, pi, sigma1, sigma2):
        if approach == 'bnn':
            if prior == 'normal':
                # FIXME: should've been exp(-sigma), experiments don't really compare between normal and scale_mixture.
                self.apply(bnn.init_priors(
                    Normal, loc=0.0, scale=math.exp(sigma)
                ))
            elif prior == 'scale_mixture':
                self.apply(bnn.init_priors(
                    ScaleMixtureNormal, pi=pi, sigma1=math.exp(-sigma1), sigma2=math.exp(-sigma2)
                ))
            else:
                raise ValueError(f"Unsupported prior distribution '{prior}'.")

            # FIXME: does not make sense as it is.
            # self.apply(bnn.init_posteriors(
            #     Normal, loc=Normal(0.0, 0.01), scale=Normal(math.exp(-2), 0.01)
            # ))
