from torch import nn
from torch.distributions import Distribution

from typing import Callable, Type

from .core import BayesParameter


def init_priors(prior: Distribution) -> Callable[[nn.Module], None]:
    def initializer(module: nn.Module):
        if isinstance(module, BayesParameter):
            module.prior = prior.expand(module.shape)

    return initializer


def init_posteriors(posterior_cls: Type[Distribution], *args: Distribution, **kwargs: Distribution) -> Callable[[nn.Module], None]:
    def initializer(module: nn.Module):
        if isinstance(module, BayesParameter):
            shape = module.shape

            params = []
            for arg in args:
                param = nn.Parameter(arg.expand(shape).sample())
                params.append(param)

            kwparams = {}
            for key, arg in kwargs.items():
                param = nn.Parameter(arg.expand(shape).sample())
                kwparams[key] = param

            module.posterior = posterior_cls(*params, **kwparams)

    return initializer
