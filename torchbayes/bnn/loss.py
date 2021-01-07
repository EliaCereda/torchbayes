import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Callable

from .core import BayesParameter


def _zero(_batch_idx: int, _n_batches: int) -> float:
    return 0.0


def _uniform(_batch_idx: int, n_batches: int) -> float:
    return 1 / n_batches


def _exp_decay(batch_idx: int, _n_batches: int) -> float:
    return 2 ** (-batch_idx - 1)


def complexity_weights(name: str) -> Callable[[int, int], float]:
    return complexity_weights.choices[name]


complexity_weights.choices = {
    None: _zero,
    'uniform': _uniform,
    'exp_decay': _exp_decay,
}


def entropy(probs, dim=-1):
    # Clamp to an epsilon constant to ensure that 0 log 0 = 0, instead of nan:
    # the value is set as float32.ulp(0.0) = 1e-45, so that only 0.0 is changed.
    # TODO: ideally should use the ulp of the actual dtype of logits.
    eps = 1e-45

    return -torch.sum(probs * torch.log(probs.clamp_min(eps)), dim=dim)


class ComplexityCost(nn.Module):
    def __init__(self, model: nn.Module, reduction='sum'):
        super().__init__()

        modules = model.modules()
        self._params = list(filter(self._is_param, modules))

        if reduction == 'none':
            self.reduce = lambda x: x
        elif reduction == 'mean':
            # TODO: determine whether taking the mean makes any sense
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum
        else:
            raise ValueError(f"Unsupported reduction value '{reduction}'.")

    def forward(self):
        if len(self._params) == 0:
            return torch.tensor(0.0)

        return self.reduce(torch.stack([
            self._mc_kl_divergence(param) for param in self._params
        ]))

    @staticmethod
    def _is_param(module: nn.Module):
        return isinstance(module, BayesParameter)

    def _mc_kl_divergence(self, param: BayesParameter) -> Tensor:
        """TODO: revisit the name"""
        current_sample = param()
        return self.reduce(param.posterior.log_prob(current_sample) - param.prior.log_prob(current_sample))
