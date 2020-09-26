import torch
from torch import Tensor
from torch.distributions import Distribution, kl_divergence
from torch.nn import Module, Parameter, ParameterDict

from numbers import Number
from typing import Union

TensorLike = Union[Number, Tensor]


class _DistributionWrapper(Module):
    def __init__(self, distribution: Distribution, parameters: bool):
        super().__init__()

        self.distribution = distribution

        # TODO: being able to override might be useful
        args = distribution.arg_constraints.keys()

        for name in args:
            arg: Tensor = getattr(distribution, name)

            if parameters:
                # FIXME: this is not ideal, Parameter(arg) makes a copy of the data
                if not isinstance(arg, Parameter):
                    arg = Parameter(arg)
                    setattr(distribution, name, arg)

                self.register_parameter(name, arg)
            else:
                self.register_buffer(name, arg)


class BayesParameter(Module):
    def __init__(self, prior: Distribution, posterior: Distribution):
        super().__init__()

        self._prior = _DistributionWrapper(prior, parameters=False)
        self._posterior = _DistributionWrapper(posterior, parameters=True)

    @property
    def prior(self) -> Distribution:
        return self._prior.distribution

    @property
    def posterior(self) -> Distribution:
        return self._posterior.distribution

    def forward(self) -> Tensor:
        return self.posterior.rsample()


class ComplexityCost(Module):
    def __init__(self, model: Module):
        super().__init__()

        modules = model.modules()
        self._params = list(filter(self._is_param, modules))

    @staticmethod
    def _is_param(module: Module):
        return isinstance(module, BayesParameter)

    def forward(self):
        return sum(kl_divergence(param.posterior, param.prior) for param in self._params)

