import torch
from torch import Tensor
from torch.distributions import Distribution, kl_divergence
from torch.nn import Module, Parameter, ParameterDict
from typing import Type, Union

TensorLike = Union[float, Tensor]


class BayesParameter(Module):
    def __init__(self, prior: Distribution, posterior: Type[Distribution], **posterior_args: TensorLike):
        super().__init__()

        posterior_params = {name: Parameter(torch.as_tensor(arg)) for name, arg in posterior_args.items()}

        for name, param in posterior_params.items():
            self.register_parameter(name, param)

        self.prior = prior
        self.posterior = posterior(**posterior_params)

        assert self.prior.batch_shape == self.posterior.batch_shape
        assert self.prior.event_shape == self.posterior.event_shape

    def forward(self) -> Tensor:
        return self.posterior.rsample()


class ComplexityCost(Module):
    def __init__(self, model: Module):
        super().__init__()

        self._params = [module for module in model.modules() if isinstance(module, BayesParameter)]

    def forward(self):
        return sum(kl_divergence(param.posterior, param.prior) for param in self._params)

