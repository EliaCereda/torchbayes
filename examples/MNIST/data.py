import os
from typing import List
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict

import torch
from torch.utils import data
from torchvision import datasets, transforms


class MNISTData(pl.LightningDataModule):
    config_keys = [
        'data_dir', 'batch_size',
    ]

    @classmethod
    def add_data_args(cls, parent: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent], add_help=False)
        group = parser.add_argument_group('Data Parameters')
        group.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'data'),
                           help='Location of downloaded datasets (default: %(default)s)')
        group.add_argument('--batch_size', type=int, default=128,
                           help='Batch size (default: %(default)s)')

        return parser

    def __init__(self, config):
        super().__init__()

        self.config = AttributeDict({key: getattr(config, key) for key in self.config_keys})

    def prepare_data(self):
        datasets.MNIST(self.config.data_dir, download=True)
        datasets.FashionMNIST(self.config.data_dir, download=True)

    @property
    def _loader_args(self):
        return dict(
            batch_size=self.config.batch_size,
            num_workers=3,  # FIXME: read number of CPUs
            pin_memory=self.trainer.on_gpu
        )

    def train_dataloader(self) -> data.DataLoader:
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(self.config.data_dir, train=True, transform=transform)

        return data.DataLoader(dataset, shuffle=True, **self._loader_args)

    def val_dataloader(self) -> List[data.DataLoader]:
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(self.config.data_dir, train=False, transform=transform)

        # Out-of-domain dataset to evaluate entropy
        ood_dataset = datasets.FashionMNIST(self.config.data_dir, train=False, transform=transform)

        return [
            data.DataLoader(ds, shuffle=False, **self._loader_args)
            for ds in [dataset, ood_dataset]
        ]

