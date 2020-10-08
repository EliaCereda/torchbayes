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
        import math
        import torch
        from torch.nn import init
        from torch.distributions import Normal

        weight_loc = torch.empty(self.weight.shape)
        init.kaiming_uniform_(weight_loc, a=math.sqrt(5))

        weight_scale = torch.empty(self.weight.shape)
        init.normal_(weight_scale, mean=0.1, std=0.01)

        self.weight.set_posterior(Normal, loc=weight_loc, scale=weight_scale)

        fan_in, _ = init._calculate_fan_in_and_fan_out(weight_loc)
        bound = 1 / math.sqrt(fan_in)

        bias_loc = torch.empty(self.bias.shape)
        init.uniform_(bias_loc, -bound, bound)

        bias_scale = torch.empty(self.bias.shape)
        init.normal_(bias_scale, mean=0.1, std=0.01)

        self.bias.set_posterior(Normal, loc=bias_loc, scale=bias_scale)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight(), self.bias())
