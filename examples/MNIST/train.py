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

    import warnings
    from torch.utils.tensorboard import SummaryWriter
    from torch.jit import TracerWarning
    writer = SummaryWriter()
    inputs, targets = next(iter(train_loader))
    with warnings.catch_warnings():
        # Suppress warning that the traced module produces different values from
        # the Python module: this is expected with random functions.
        warnings.filterwarnings('ignore', '.* does not match the corresponding output .*', TracerWarning)
        writer.add_graph(model, inputs.to(device))

    sample_idx = 0
    epochs = trange(n_epochs)
    for epoch in epochs:
        model.train()

        n_batches = len(train_loader)
        batches = tqdm(train_loader, leave=False)
        for batch, batch_ in enumerate(batches):
            inputs, targets = map(lambda x: x.to(device), batch_)

            optimizer.zero_grad()

            train_comp = 0.0
            train_like = 0.0
            train_loss = 0.0
            train_accuracy = 0.0

            for sample in range(n_samples):
                model.sample_()
                weight = 1 / (n_batches * batch_size) # 2 ** (-batch - 2) #math.exp(math.log(2) * (n_batches - batch - 1) - math.log(2 ** n_batches - 1))
                complexity_ = weight * complexity()

                logits = model(inputs)

                likelihood_ = likelihood(logits, targets)
                loss = (complexity_ + likelihood_) / n_samples
                loss.backward()

                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == targets).to(torch.float32).mean()
                accuracy /= n_samples

                train_comp += complexity_ / n_samples
                train_like += likelihood_ / n_samples
                train_loss += loss
                train_accuracy += accuracy

            optimizer.step()

            batches.set_postfix_str(f't_comp={train_comp.item():.3f}, '
                                    f't_like={train_like.item():.3f}, '
                                    f't_loss={train_loss.item():.3f}, '
                                    f't_acc={train_accuracy.item():.3f}', refresh=False)

            sample_idx += inputs.shape[0]
            writer.add_scalar('train/complexity', train_comp.item(), sample_idx)
            writer.add_scalar('train/likelihood', train_like.item(), sample_idx)
            writer.add_scalar('train/loss', train_loss.item(), sample_idx)
            writer.add_scalar('train/accuracy', train_accuracy.item(), sample_idx)

        with torch.no_grad():
            model.eval()

            for batch in test_loader:
                inputs, targets = map(lambda x: x.to(device), batch)

                test_loss = 0.0
                test_accuracy = 0.0

                for sample in range(n_samples):
                    model.sample_()
                    logits = model(inputs)

                    weight = 1 / (n_batches * batch_size)
                    loss = weight * complexity() + likelihood(logits, targets)
                    loss /= n_samples

                    preds = torch.argmax(logits, dim=1)
                    accuracy = (preds == targets).to(torch.float32).mean()
                    accuracy /= n_samples

                    test_loss += loss
                    test_accuracy += accuracy

                epochs.set_postfix(dict(
                    v_loss=test_loss.item(),
                    v_accuracy=test_accuracy.item(),
                ))

                writer.add_scalar('test/loss', test_loss.item(), sample_idx)
                writer.add_scalar('test/accuracy', test_accuracy.item(), sample_idx)

    writer.close()


if __name__ == '__main__':
    main(learning_rate=1e-3)
