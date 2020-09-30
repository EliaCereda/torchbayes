import os

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms

from tqdm import tqdm, trange

from torchbayes import nn as bnn

from examples.MNIST.model import Model


def main(*,
         data_dir=None,
         device=None,
         batch_size=128,
         learning_rate=1e-4,
         n_epochs=5):

    if data_dir is None:
        example_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(example_dir, 'data')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data preparation
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model setup
    model = Model([1, 28, 28], 10).to(device)
    complexity = bnn.ComplexityCost(model)
    likelihood = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in trange(n_epochs):
        for i, batch in enumerate(tqdm(train_loader)):
            inputs, targets = map(lambda x: x.to(device), batch)

            optimizer.zero_grad()

            model.sample_()
            logits = model(inputs)
            loss = complexity() + likelihood(logits, targets)

            loss.backward()
            optimizer.step()

            if i == 10:
                break


if __name__ == '__main__':
    main(n_epochs=1)