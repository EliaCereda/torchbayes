import torch
from torch.distributions import Normal, kl_divergence

from torchbayes.nn import BayesParameter, ComplexityCost, BayesModel


def all_identical(it1, it2):
    return all(el1 is el2 for el1, el2 in zip(it1, it2))


def test_parameters():
    bp1 = BayesParameter(())
    bp1.set_prior(Normal, loc=0.0, scale=1.0)
    bp1.set_posterior(Normal, loc=1.0, scale=1.0)

    buffers1 = list(bp1.buffers())
    params1 = list(bp1.parameters())

    assert len(buffers1) == 2
    # assert all_identical(buffers1, [prior.loc, prior.scale])

    assert len(params1) == 2
    # assert all_identical(params1, [posterior.loc, posterior.scale])

    bp2 = BayesParameter(())

    params2 = list(bp2.parameters())

    assert all_identical(params1, params2)


def test_training():
    param = BayesParameter(shape=())

    param.set_prior(Normal, loc=0.0, scale=1.0)
    param.set_posterior(Normal, loc=1.0, scale=1.0)

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


# def test_shape_expand():
#     bp = BayesParameter(shape=(12,))
#
#     bp.prior = Normal(0.0, 1.0)
#     bp.posterior = Normal(1.0, 2.0)
#
#     sample: Tensor = bp()
#
#     # Test that the parameter returns the correct shape, by expanding the result
#     # of sampling the posterior. Test that in this case the returned values are
#     # all the same.
#     # TODO: does this even make sense? What should the semantics be if the
#     #  distributions don't match the shape of the parameter?
#     assert sample.shape == bp.shape
#     assert torch.all(sample == sample[0])


def test_ComplexityCost():
    class Model(BayesModel):
        def __init__(self):
            super().__init__()

            self.param1 = BayesParameter(shape=())
            self.param1.set_prior(Normal, loc=0.0, scale=1.0)
            self.param1.set_posterior(Normal, loc=1.0, scale=1.0)

            self.param2 = BayesParameter(shape=())
            self.param2.set_prior(Normal, loc=0.0, scale=1.0)
            self.param2.set_posterior(Normal, loc=0.0, scale=2.0)

        def forward(self, input):
            return self.param1() * input + self.param2()

    model = Model()
    loss_fn = ComplexityCost(model)

    input = torch.randn(1)

    n_samples = 1000
    loss = torch.tensor(0.0)

    for i in range(n_samples):
        model.sample_()

        output = model(input)

        sample_loss = loss_fn()
        loss += sample_loss

        # Check ComplexityCost of model is sum of ComplexityCosts of params
        losses = [ComplexityCost(param)() for param in [model.param1, model.param2]]
        assert torch.isclose(sample_loss, sum(losses))

    loss /= n_samples

    # Check ComplexityCost is reasonably close to the exact KL divergence
    kld = sum(kl_divergence(param.posterior, param.prior) for param in [model.param1, model.param2])
    assert torch.isclose(loss, kld, rtol=0.15)
