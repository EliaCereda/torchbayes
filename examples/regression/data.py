from argparse import ArgumentParser
import math
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict
import torch
from torch.distributions import Normal
from torch.utils import data


class RegressionData(pl.LightningDataModule):
    hparams_keys = [
        'noise',
        'train_seed', 'train_range', 'train_samples',
        'valid_range', 'valid_samples',
        'batch_size',
    ]

    @classmethod
    def add_data_args(cls):
        parser = ArgumentParser(add_help=False)
        group = parser.add_argument_group('Data Parameters')
        group.add_argument('--noise', default=0.02,
                           help='Standard deviation of the Gaussian noise added to samples (default: %(default)s)')
        group.add_argument('--train_seed', default=42,
                           help='Random seed used to generate the datasets (default: %(default)s)')
        group.add_argument('--train_range', nargs=2, default=[0.0, 0.5],
                           help='Domain of the training set (default: %(default)s)')
        group.add_argument('--train_samples', default=1024,
                           help='Number of samples of the training set (default: %(default)s)')
        group.add_argument('--valid_range', nargs=2, default=[-0.5, 1.2],
                           help='Domain of the validation set (default: %(default)s)')
        group.add_argument('--valid_samples', default=2048,
                           help='Number of samples of the validation set (default: %(default)s)')
        group.add_argument('--batch_size', type=int, default=128,
                           help='Batch size (default: %(default)s)')

        return parser

    def __init__(self, config):
        super().__init__()

        self.hparams = AttributeDict({key: getattr(config, key) for key in self.hparams_keys})

    @property
    def _loader_args(self):
        return dict(
            batch_size=self.hparams.batch_size,
            pin_memory=self.trainer.on_gpu
        )

    def groundtruth_function(self, x: torch.Tensor):
        return x + 0.3 * torch.sin(2 * math.pi * x) + 0.3 * torch.sin(4 * math.pi * x)

    def noisy_function(self, x: torch.Tensor):
        noise = Normal(0.0, self.hparams.noise).sample(x.shape)
        return self.groundtruth_function(x + noise)

    def train_set(self):
        with torch.random.fork_rng():
            torch.manual_seed(self.hparams.train_seed)

            x = torch\
                .empty(self.hparams.train_samples, 1)\
                .uniform_(*self.hparams.train_range)
            y = self.noisy_function(x)

        y = torch.cat([y, -y], dim=-1)

        return x, y

    def train_dataloader(self) -> data.DataLoader:
        x, y = self.train_set()
        dataset = data.TensorDataset(x, y)

        return data.DataLoader(dataset, shuffle=True, **self._loader_args)

    def validation_set(self):
        x = torch\
            .linspace(*self.hparams.valid_range, self.hparams.valid_samples)\
            .unsqueeze(-1)
        y = self.groundtruth_function(x)

        return x, y

    def val_dataloader(self) -> data.DataLoader:
        x, y = self.validation_set()
        dataset = data.TensorDataset(x, y)

        return data.DataLoader(dataset, shuffle=False, batch_size=self.hparams.valid_samples)

    def visualize(self, validation_preds=None):
        # Visualize the datasets
        train = self.train_set()
        validation = self.validation_set()

        train = train[0].expand_as(train[1]), train[1]

        plt.ion()
        plt.figure(0, clear=True)

        plt.title("Dataset")
        groundtruth, = plt.plot(*validation, linewidth=3)
        train = plt.scatter(*train, marker='.', c='k')

        valid = []
        if validation_preds is not None:
            x, _ = validation
            y = validation_preds

            valid = plt.plot(x, y, c='C1', linewidth=0.5)

        plt.legend([groundtruth, train] + valid[:1], ["Ground truth", "Training set", "Predicted functions"])
        plt.show()
        plt.pause(0.010)


def main():
    parser = ArgumentParser(parents=[
        RegressionData.add_data_args(),
    ])
    args = parser.parse_args()

    data = RegressionData(args)

    # Test dataset generation reproducibility
    # (only training, validation is deterministic)
    x1, y1 = data.train_set()
    x2, y2 = data.train_set()

    assert torch.allclose(x1, x2)
    assert torch.allclose(y1, y2)

    # Visualize the generated dataset
    data.visualize()

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
