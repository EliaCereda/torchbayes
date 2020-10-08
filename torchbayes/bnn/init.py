import torch
from torch import nn
from torch.distributions import Distribution

from typing import Callable, Type, Union, Any

from .core import BayesParameter, TensorLike


DistributionFactory = Union[Type[Distribution], Callable[[Any], Distribution]]


def init_priors(prior_cls: DistributionFactory, **kwargs: TensorLike) -> Callable[[nn.Module], None]:
    def initializer(module: nn.Module):
        if isinstance(module, BayesParameter):
            module.set_prior(prior_cls, **kwargs)

    return initializer


def init_posteriors(posterior_cls: DistributionFactory, **kwargs: Distribution) -> Callable[[nn.Module], None]:
    def initializer(module: nn.Module):
        if isinstance(module, BayesParameter):
            module.set_posterior(posterior_cls, **kwargs)

    return initializer
