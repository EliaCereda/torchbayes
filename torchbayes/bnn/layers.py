import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from .core import SizeLike, BayesParameter
from ..distributions import NormalNonSingular
from ..utils import _pair


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

        weight_mu = torch.empty(self.weight.shape)
        init.kaiming_uniform_(weight_mu, a=math.sqrt(5))

        weight_rho = torch.empty(self.weight.shape)
        init.normal_(weight_rho, mean=0.1, std=0.01)
        weight_rho.expm1_().log_()

        self.weight.set_posterior(NormalNonSingular, mu=weight_mu, rho=weight_rho)

        if self.bias is not None:
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


class _BayesConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

        self.weight = BayesParameter(shape=(out_channels, in_channels, *kernel_size))
        self.bias = BayesParameter(shape=(out_channels,))

        self.reset_parameters()

    def reset_parameters(self):
        import math
        import torch
        from torch.nn import init

        weight_mu = torch.empty(self.weight.shape)
        init.kaiming_uniform_(weight_mu, a=math.sqrt(5))

        weight_rho = torch.empty(self.weight.shape)
        init.normal_(weight_rho, mean=0.1, std=0.01)
        weight_rho.expm1_().log_()

        self.weight.set_posterior(NormalNonSingular, mu=weight_mu, rho=weight_rho)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight_mu)
            bound = 1 / math.sqrt(fan_in)

            bias_mu = torch.empty(self.bias.shape)
            init.uniform_(bias_mu, -bound, bound)

            bias_rho = torch.empty(self.bias.shape)
            init.normal_(bias_rho, mean=0.1, std=0.01)
            bias_rho.expm1_().log_()

            self.bias.set_posterior(NormalNonSingular, mu=bias_mu, rho=bias_rho)


class BayesConv2d(_BayesConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, padding_mode)

    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode == 'zeros':
            return F.conv2d(input, self.weight(), self.bias(), self.stride, self.padding)
        else:
            raise NotImplementedError(
                "Support for non-zeros padding modes in BayesConv2d has not been implemented yet."
            )


class BayesConv2dFlipout(BayesConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert isinstance(self.weight.posterior, NormalNonSingular), \
            "Only NormalNonSingular posteriors are supported by BayesConv2dFlipout."
        self.weight.rsample_state = True

    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode == 'zeros':
            batch_size = input.shape[0]
            device = input.device

            r = randsign((batch_size, self.out_channels, 1, 1), device=device)
            s = randsign((batch_size, self.in_channels, 1, 1), device=device)

            weight_mean, weight_stdev, weight_eps = self.weight._sample_state
            weight_delta = weight_stdev * weight_eps

            output = F.conv2d(input, weight_mean, self.bias(), self.stride, self.padding)
            output += F.conv2d(input * s, weight_delta, None, self.stride, self.padding) * r

            return output
        else:
            raise NotImplementedError(
                "Support for non-zeros padding modes in BayesConv2dFlipout has not been implemented yet."
            )

def randsign(shape: SizeLike, dtype=None, device=None):
    # Sample a Bernoullian with the desired output shape
    out = torch\
        .empty(shape, dtype=dtype, device=device)\
        .bernoulli_()

    # Rescale the Bernoullian values from {0, 1} to {-1, +1}
    out = 2 * out - 1

    return out
