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

    bp1 = BayesParameter((), prior, posterior)

    buffers1 = list(bp1.buffers())
    params1 = list(bp1.parameters())

    assert len(buffers1) == 2
    assert all_identical(buffers1, [prior.loc, prior.scale])

    assert len(params1) == 2
    assert all_identical(params1, [posterior.loc, posterior.scale])

    bp2 = BayesParameter((), prior, posterior)

    params2 = list(bp2.parameters())

    assert all_identical(params1, params2)


def test_training():
    param = BayesParameter(
        shape=(),
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


def test_linear():
    class BayesLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()

            self.weight = BayesParameter(shape=(out_features, in_features))
            self.bias = BayesParameter(shape=(out_features,))

        def forward(self, input: Tensor) -> Tensor:
            return F.linear(input, self.weight(), self.bias())

    net = BayesLinear(12, 24)

    prior = Normal(0.0, 1.0)
    net.weight.prior = prior.expand(net.weight.shape)
    net.bias.prior = prior.expand(net.bias.shape)

    net.weight.posterior = Normal(
        loc=torch.full(net.weight.shape, 8.0),
        scale=torch.full(net.weight.shape, 4.0)
    )
    net.bias.posterior = Normal(
        loc=torch.full(net.bias.shape, 1.0),
        scale=torch.full(net.bias.shape, 2.0)
    )

    input = torch.randn(12)
    output = net(input)


def test_shape_expand():
    bp = BayesParameter(shape=(12,))

    bp.prior = Normal(0.0, 1.0)
    bp.posterior = Normal(1.0, 2.0)

    sample: Tensor = bp()

    # Test that the parameter returns the correct shape, by expanding the result
    # of sampling the posterior. Test that in this case the returned values are
    # all the same.
    # TODO: does this even make sense? What should the semantics be if the
    #  distributions don't match the shape of the parameter?
    assert sample.shape == bp.shape
    assert torch.all(sample == sample[0])

    pass


def test_ComplexityCost():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.param1 = BayesParameter(
                shape=(),
                prior=Normal(loc=0.0, scale=1.0),
                posterior=Normal(loc=1.0, scale=1.0)
            )

            self.param2 = BayesParameter(
                shape=(),
                prior=Normal(loc=0.0, scale=1.0),
                posterior=Normal(loc=0.0, scale=2.0)
            )

    model = Model()

    loss_fn = ComplexityCost(model)
    loss = loss_fn()

    # Check ComplexityCost of model is sum of ComplexityCosts of params
    losses = [ComplexityCost(param)() for param in [model.param1, model.param2]]
    assert loss == sum(losses)
