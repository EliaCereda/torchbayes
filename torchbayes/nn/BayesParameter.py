import torch
from torch import Tensor
from torch import nn
from torch.distributions import Distribution, kl_divergence

from numbers import Number
from typing import Union, Tuple

TensorLike = Union[Number, Tensor]


class _DistributionWrapper(nn.Module):
    """
    Wrapper for `Distribution`s to help them participate in `nn.Module` behaviours.

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
            arg: Tensor = getattr(distribution, name)

            if parameters:
                # FIXME: this is not ideal, Parameter(arg) makes a copy of the
                #   data but whoever created the Distribution may depend on
                #   changing `arg` through the original copy.
                #   Explore whether it's possible to directly expect the caller
                #   to pass a Parameter.
                if not isinstance(arg, nn.Parameter):
                    arg = nn.Parameter(arg)
                    setattr(distribution, name, arg)

                self.register_parameter(name, arg)
            else:
                self.register_buffer(name, arg)

    def forward(self):
        raise NotImplementedError()


SizeLike = Union[int, Tuple[int, ...], torch.Size]


class BayesParameter(nn.Module):
    def __init__(self, shape: SizeLike,
                 prior: Distribution = None, posterior: Distribution = None):
        super().__init__()

        self.shape = torch.Size(shape)

        self.prior = prior
        self.posterior = posterior

    # TODO: a property decorator specific for _DistributionWrapper might be useful
    @property
    def prior(self) -> Distribution:
        if self._prior is not None:
            return self._prior.distribution

    @prior.setter
    def prior(self, prior: Distribution):
        if prior is not None:
            prior = _DistributionWrapper(prior, parameters=False)

        self._prior = prior

    @property
    def posterior(self) -> Distribution:
        if self._posterior is not None:
            return self._posterior.distribution

    @posterior.setter
    def posterior(self, posterior: Distribution):
        if posterior is not None:
            assert posterior.has_rsample, \
                "BayesParameter requires a posterior distribution which supports rsample(...)."
            posterior = _DistributionWrapper(posterior, parameters=True)

        self._posterior = posterior

    def forward(self) -> Tensor:
        return self.posterior.rsample()


class ComplexityCost(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        modules = model.modules()
        self._params = list(filter(self._is_param, modules))

    @staticmethod
    def _is_param(module: nn.Module):
        return isinstance(module, BayesParameter)

    def forward(self):
        return sum(kl_divergence(param.posterior, param.prior) for param in self._params)

