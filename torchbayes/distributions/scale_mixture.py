import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal


def NormalScaleMixture(pi, sigma1, sigma2):
    mixture = Categorical(torch.tensor([pi, 1 - pi]))
    components = Normal(0.0, torch.tensor([sigma1, sigma2]))

    return MixtureSameFamily(mixture, components)
