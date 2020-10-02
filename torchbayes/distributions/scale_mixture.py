import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal


def ScaleMixtureNormal(pi, sigma1, sigma2):
    mixture = Categorical(torch.tensor([pi, 1 - pi], device=pi.device))
    components = Normal(0.0, torch.tensor([sigma1, sigma2], device=pi.device))

    return MixtureSameFamily(mixture, components)
