import torch
from torch import nn
from torch import Tensor
from torch.distributions import Distribution

from numbers import Number
from typing import Union, Tuple

TensorLike = Union[Number, Tensor]
SizeLike = Union[int, Tuple[int, ...], torch.Size]


class _DistributionWrapper(nn.Module):
    """
    Wrapper for `distributions.Distribution`s to help them participate in
    `nn.Module` behaviours.

    The arguments of the distribution appear as parameters or buffers of the
    module: collecting training parameters for the optimizers, moving modules
    between devices and converting data types with `to(...)` automatically
    consider the distribution arguments.

    Serialization through `state_dict()` works out of the box for the arguments,
    but not for the type of distribution.

    TODO: record the type of distribution in the metadata and recreate it at
      load time.
    """

    def __init__(self, distribution: Distribution, parameters: bool):
        super().__init__()

        self.distribution = distribution

        # TODO: being able to override might be useful
        args = distribution.arg_constraints.keys()

        for name in args:
            arg = getattr(distribution, name)

            if parameters:
                self.register_parameter(name, arg)
            else:
                self.register_buffer(name, arg)

    def forward(self):
        raise NotImplementedError()


def _distribution_shape(distribution: Distribution):
    return distribution.batch_shape + distribution.event_shape


class BayesModel(nn.Module):
    def sample_(self: nn.Module):
        self.apply(BayesModel.sample_parameters_)

    @staticmethod
    def sample_parameters_(module: nn.Module):
        if isinstance(module, BayesParameter):
            module.sample_()


class BayesParameter(nn.Module):
    def __init__(self, shape: SizeLike,
                 prior: Distribution = None, posterior: Distribution = None):
        super().__init__()

        self.shape = torch.Size(shape)

        self.prior = prior
        self.posterior = posterior

        self._sampled = None

    # TODO: a property decorator specific for _DistributionWrapper might be useful
    @property
    def prior(self) -> Distribution:
        if self._prior is not None:
            return self._prior.distribution

    @prior.setter
    def prior(self, prior: Distribution):
        if prior is not None:
            assert _distribution_shape(prior) == self.shape
            prior = _DistributionWrapper(prior, parameters=False)

        self._prior = prior

    @property
    def posterior(self) -> Distribution:
        if self._posterior is not None:
            return self._posterior.distribution

    @posterior.setter
    def posterior(self, posterior: Distribution):
        if posterior is not None:
            assert _distribution_shape(posterior) == self.shape
            assert posterior.has_rsample, \
                "BayesParameter requires a posterior distribution which supports rsample(...)."
            posterior = _DistributionWrapper(posterior, parameters=True)

        self._posterior = posterior

    def sample_(self):
        assert self.posterior is not None, \
            "The posterior distribution must be initialized before sampling."

        self._sampled = self.posterior.rsample()

    def forward(self) -> Tensor:
        assert self._sampled is not None, \
            "sample_() must be called before calling forward()."

        return self._sampled
