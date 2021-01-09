import math
import torch
from torch.distributions import constraints, Distribution, Categorical, MixtureSameFamily, Normal
from torch.distributions.utils import broadcast_all, _standard_normal


class NormalNonSingular(Distribution):
    """
    Normal distribution, parametrized such that its scale can never become zero
    or negative. Should be used when the parameters of the distribution must be
    learned, since gradient descent cannot set them to invalid values.

    Source: Blundell et al., Weight Uncertainty in Neural Networks, 2015.
    """
    arg_constraints = {'mu': constraints.real, 'rho': constraints.real}
    has_rsample = True

    mu: torch.Tensor
    rho: torch.Tensor

    @property
    def mean(self):
        return self.mu

    @property
    def stddev(self):
        return torch.log1p(torch.exp(self.rho))

    @property
    def variance(self):
        return self.stddev ** 2

    def __init__(self, mu, rho, validate_args=None):
        self.mu, self.rho = broadcast_all(mu, rho)

        super().__init__(self.mu.size(), validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NormalNonSingular, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.rho = self.rho.expand(batch_shape)
        super(NormalNonSingular, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        sigma = self.stddev
        eps = _standard_normal(shape, dtype=self.mu.dtype, device=self.mu.device)
        return self.mu + sigma * eps

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        sigma = self.stddev
        return -0.5 * ((value - self.mu) / sigma) ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)


def ScaleMixtureNormal(pi, sigma1, sigma2):
    mixture = Categorical(torch.tensor([pi, 1 - pi], device=pi.device))
    components = Normal(0.0, torch.tensor([sigma1, sigma2], device=pi.device))

    return MixtureSameFamily(mixture, components)
