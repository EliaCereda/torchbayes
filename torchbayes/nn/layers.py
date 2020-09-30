from torch import nn
from torch import Tensor
from torch.nn import functional as F

from .core import BayesParameter


class BayesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.weight = BayesParameter(shape=(out_features, in_features))
        self.bias = BayesParameter(shape=(out_features,))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight(), self.bias())
