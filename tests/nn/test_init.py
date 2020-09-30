import torch
from torch.distributions import Normal

from torchbayes import nn as bnn
from torchbayes.nn import BayesModel, BayesLinear


def test_linear():
    net = BayesLinear(12, 24)

    prior = Normal(0.0, 1.0)

    net.apply(bnn.init_priors(Normal(0.0, 1.0)))
    net.apply(bnn.init_posteriors(Normal, 2.0, 4.0))

    BayesModel.sample_(net)

    input = torch.randn(12)
    output = net(input)
