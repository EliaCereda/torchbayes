import torch
from torch.distributions import Normal, kl_divergence
from torch.nn import Module

from torchbayes.nn import BayesParameter, ComplexityCost


def test_BayesParameter():
    param = BayesParameter(
        prior=Normal(loc=0.0, scale=1.0),
        posterior=Normal, loc=1.0, scale=1.0
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
    class Model(Module):
        def __init__(self):
            super().__init__()

            self.param1 = BayesParameter(
                prior=Normal(loc=0.0, scale=1.0),
                posterior=Normal, loc=1.0, scale=1.0
            )

            self.param2 = BayesParameter(
                prior=Normal(loc=0.0, scale=1.0),
                posterior=Normal, loc=0.0, scale=2.0
            )

    model = Model()

    loss_fn = ComplexityCost(model)
    loss = loss_fn()

    # Check ComplexityCost of model is sum of ComplexityCosts of params
    losses = [ComplexityCost(param)() for param in [model.param1, model.param2]]
    assert loss == sum(losses)

    pass