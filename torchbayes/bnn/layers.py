from torch import nn
from torch import Tensor
from torch.nn import functional as F

from .core import BayesParameter


class BayesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.weight = BayesParameter(shape=(out_features, in_features))
        self.bias = BayesParameter(shape=(out_features,))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # FIXME: clean up this copy/paste
        import math
        import torch
        from torch.nn import init
        from torchbayes.distributions import NormalNonSingular

        weight_mu = torch.empty(self.weight.shape)
        init.kaiming_uniform_(weight_mu, a=math.sqrt(5))

        weight_rho = torch.empty(self.weight.shape)
        init.normal_(weight_rho, mean=0.1, std=0.01)
        weight_rho.expm1_().log_()

        self.weight.set_posterior(NormalNonSingular, mu=weight_mu, rho=weight_rho)

        fan_in, _ = init._calculate_fan_in_and_fan_out(weight_mu)
        bound = 1 / math.sqrt(fan_in)

        bias_mu = torch.empty(self.bias.shape)
        init.uniform_(bias_mu, -bound, bound)

        bias_rho = torch.empty(self.bias.shape)
        init.normal_(bias_rho, mean=0.1, std=0.01)
        bias_rho.expm1_().log_()

        self.bias.set_posterior(NormalNonSingular, mu=bias_mu, rho=bias_rho)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight(), self.bias())
