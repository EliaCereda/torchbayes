import torch
from torch import nn
from torch import Tensor

from .core import BayesParameter


class ComplexityCost(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        modules = model.modules()
        self._params = list(filter(self._is_param, modules))

    def forward(self):
        return sum(self._mc_kl_divergence(param) for param in self._params)

    @staticmethod
    def _is_param(module: nn.Module):
        return isinstance(module, BayesParameter)

    @staticmethod
    def _mc_kl_divergence(param: BayesParameter) -> Tensor:
        """TODO: revisit the name"""
        current_sample = param()
        return torch.sum(param.posterior.log_prob(current_sample) - param.prior.log_prob(current_sample))
