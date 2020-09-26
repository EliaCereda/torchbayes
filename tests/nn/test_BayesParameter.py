import torch
from torch import Tensor
from torch.distributions import Normal, kl_divergence
from torch import nn
from torch.nn import functional as F

from torchbayes.nn import BayesParameter, ComplexityCost


def all_identical(it1, it2):
    return all(el1 is el2 for el1, el2 in zip(it1, it2))


def test_parameters():
    prior = Normal(loc=0.0, scale=1.0)
    posterior = Normal(loc=1.0, scale=1.0)

    bp1 = BayesParameter(prior, posterior)

    buffers1 = list(bp1.buffers())
    params1 = list(bp1.parameters())

    assert len(buffers1) == 2
    assert all_identical(buffers1, [prior.loc, prior.scale])

    assert len(params1) == 2
    assert all_identical(params1, [posterior.loc, posterior.scale])

    bp2 = BayesParameter(prior, posterior)

    params2 = list(bp2.parameters())

    assert all_identical(params1, params2)


def test_training():
    param = BayesParameter(
        prior=Normal(loc=0.0, scale=1.0),
        posterior=Normal(loc=1.0, scale=1.0)
    )

    optimizer = torch.optim.SGD(param.parameters(), lr=1.0)

    # Perform one training iteration
    optimizer.zero_grad()

    loss = kl_divergence(param.posterior, param.prior)
    assert loss == 0.5

    loss.backward()
    optimizer.step()

    # Check convergence
    loss = kl_divergence(param.posterior, param.prior)
    assert loss == 0.0


def test_ComplexityCost():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.param1 = BayesParameter(
                prior=Normal(loc=0.0, scale=1.0),
                posterior=Normal(loc=1.0, scale=1.0)
            )

            self.param2 = BayesParameter(
                prior=Normal(loc=0.0, scale=1.0),
                posterior=Normal(loc=0.0, scale=2.0)
            )

    model = Model()

    loss_fn = ComplexityCost(model)
    loss = loss_fn()

    # Check ComplexityCost of model is sum of ComplexityCosts of params
    losses = [ComplexityCost(param)() for param in [model.param1, model.param2]]
    assert loss == sum(losses)

    pass

def test_Linear():
    class BayesLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()

            self.weight = BayesParameter()
            self.bias = BayesParameter()

        def forward(self, input: Tensor) -> Tensor:
            return F.linear(input, self.weight(), self.bias())
