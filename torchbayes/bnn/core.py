import torch
from torch import nn
from torch import Tensor
from torch.distributions import Distribution

from numbers import Number
from typing import Union, Tuple, Dict, Type

TensorLike = Union[Number, Tensor]
SizeLike = Union[int, Tuple[int, ...], torch.Size]


class _DistributionWrapper(nn.Module):
    """
    Wrapper for `distributions.Distribution`s to help them participate in
    `bnn.Module` behaviours.

    The arguments of the distribution appear as parameters or buffers of the
    module: collecting training parameters for the optimizers, moving modules
    between devices and converting data types with `to(...)` automatically
    consider the distribution arguments.

    Serialization through `state_dict()` works out of the box for the arguments,
    but not for the type of distribution.

    TODO: record the type of distribution in the metadata and recreate it at
      load time.
    """

    def __init__(self, dist_cls: Type[Distribution], dist_args: Dict, parameters: bool, shape: SizeLike):
        super().__init__()

        for name, arg in dist_args.items():
            if isinstance(arg, Distribution):
                arg = arg.expand(shape).sample()

            arg = torch.as_tensor(arg)

            if parameters:
                arg = nn.Parameter(arg)
                self.register_parameter(name, arg)
            else:
                self.register_buffer(name, arg)

        self._dist_cls = dist_cls
        self._dist_args = list(dist_args.keys())
        self._shape = shape

    def forward(self):
        raise NotImplementedError()

    def distribution(self) -> Distribution:
        dist_args = {name: getattr(self, name) for name in self._dist_args}
        return self._dist_cls(**dist_args).expand(self._shape)


class BayesModel(nn.Module):
    def sample_(self: nn.Module):
        self.apply(BayesModel.sample_parameters_)

    @staticmethod
    def sample_parameters_(module: nn.Module):
        if isinstance(module, BayesParameter):
            module.sample_()


class BayesParameter(nn.Module):
    def __init__(self, shape: SizeLike):
        super().__init__()

        self.shape = torch.Size(shape)

        self._prior = None
        self._posterior = None
        self._sample = None

    @property
    def prior(self) -> Distribution:
        if self._prior is not None:
            return self._prior.distribution()

    def set_prior(self, prior_cls: Type[Distribution], **kwargs: TensorLike):
        self._prior = _DistributionWrapper(prior_cls, kwargs, parameters=False, shape=self.shape)

    @property
    def posterior(self) -> Distribution:
        if self._posterior is not None:
            return self._posterior.distribution()

    def set_posterior(self, posterior_cls: Type[Distribution], **kwargs: TensorLike):
        assert posterior_cls.has_rsample, \
            "BayesParameter requires a posterior distribution which supports rsample(...)."

        self._posterior = _DistributionWrapper(posterior_cls, kwargs, parameters=True, shape=self.shape)

    def sample_(self):
        assert self.posterior is not None, \
            "The posterior distribution must be initialized before sampling."

        self._sample = self.posterior.rsample()

    def forward(self) -> Tensor:
        # When tracing, sample_() must be called inside forward(), so that the
        # computation of self._sample can be correctly traced. Otherwise it looks
        # like an external constant requiring gradients, which is not supported.
        if torch.jit.is_tracing():
            self.sample_()

        assert self._sample is not None, \
            "sample_() must be called before calling forward()."

        return self._sample
