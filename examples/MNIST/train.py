import math
import os

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms

from tqdm import tqdm, trange

from torchbayes import nn as bnn

from model import Model


def main(*,
         data_dir=None,
         device=None,
         batch_size=128,
         learning_rate=1e-4,
         n_epochs=5,
         n_samples=10,
         loader_args=None):

    if data_dir is None:
        example_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(example_dir, 'data')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if loader_args is None:
        loader_args = dict(pin_memory=True, num_workers=6)

    # Data preparation
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_args)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_args)

    # Model setup
    model = Model([1, 28, 28], 10).to(device)
    complexity = bnn.ComplexityCost(model)
    likelihood = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    epochs = trange(n_epochs)
    for epoch in epochs:
        model.train()

        n_batches = len(train_loader)
        batches = tqdm(train_loader, leave=False)
        for batch, batch_ in enumerate(batches):
            inputs, targets = map(lambda x: x.to(device), batch_)

            optimizer.zero_grad()

            train_loss = 0.0
            train_accuracy = 0.0

            for sample in range(n_samples):
                model.sample_()
                logits = model(inputs)

                weight = 2 ** (-batch - 2) #math.exp(math.log(2) * (n_batches - batch - 1) - math.log(2 ** n_batches - 1))
                loss  = weight * complexity() + likelihood(logits, targets)
                loss /= n_samples
                loss.backward()

                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == targets).to(torch.float32).mean()
                accuracy /= n_samples

                train_loss += loss
                train_accuracy += accuracy

            optimizer.step()

            batches.set_postfix(dict(
                # complexity=complexity,
                # likelihood=likelihood,
                t_loss=train_loss.item(),
                t_accuracy=train_accuracy.item(),
            ))

        with torch.no_grad():
            model.eval()

            for batch in test_loader:
                inputs, targets = map(lambda x: x.to(device), batch)

                test_loss = 0.0
                test_accuracy = 0.0

                for sample in range(n_samples):
                    model.sample_()
                    logits = model(inputs)

                    loss = complexity() / n_batches + likelihood(logits, targets)
                    loss /= n_samples

                    preds = torch.argmax(logits, dim=1)
                    accuracy = (preds == targets).to(torch.float32).mean()

                    test_loss += loss
                    test_accuracy += accuracy / n_samples

                epochs.set_postfix(dict(
                    v_loss=test_loss.item(),
                    v_accuracy=test_accuracy.item(),
                ))


if __name__ == '__main__':
    main()
