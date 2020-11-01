import os
from typing import List
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict

import torch
from torch.utils import data
from torchvision import datasets, transforms

import wandb

class MNISTData(pl.LightningDataModule):
    config_keys = [
        'data_dir', 'batch_size',
    ]

    @classmethod
    def add_data_args(cls, parser):
        parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'data'),
                           help='Location of downloaded datasets (default: %(default)s)')
        parser.add_argument('--batch_size', type=int, default=128,
                           help='Batch size (default: %(default)s)')

    def __init__(self, config):
        super().__init__()

        self.config = AttributeDict({key: getattr(config, key) for key in self.config_keys})

        self.transform = None

        self.mnist_splits = None
        self.fmnist_splits = None

    def prepare_data(self):
        datasets.MNIST(self.config.data_dir, download=True)
        datasets.FashionMNIST(self.config.data_dir, download=True)

    def setup(self, stage=None):
        self.transform = transforms.ToTensor()

        if stage == 'fit':
            splits_path = wandb.use_artifact('mnist:latest', type='dataset_split').file()
            self.mnist_splits = torch.load(splits_path)

            splits_path = wandb.use_artifact('fashion_mnist:latest', type='dataset_split').file()
            self.fmnist_splits = torch.load(splits_path)

    @property
    def _loader_args(self):
        return dict(
            batch_size=self.config.batch_size,
            num_workers=3,  # FIXME: read number of CPUs
            pin_memory=self.trainer.on_gpu
        )

    def train_dataloader(self) -> data.DataLoader:
        dataset = datasets.MNIST(self.config.data_dir, train=True, transform=self.transform)
        dataset = data.Subset(dataset, self.mnist_splits['train'])

        return data.DataLoader(dataset, shuffle=True, **self._loader_args)

    def val_dataloader(self) -> List[data.DataLoader]:
        dataset = datasets.MNIST(self.config.data_dir, train=True, transform=self.transform)
        dataset = data.Subset(dataset, self.mnist_splits['valid'])

        # Out-of-domain dataset to evaluate entropy
        ood_dataset = datasets.FashionMNIST(self.config.data_dir, train=True, transform=self.transform)
        ood_dataset = data.Subset(ood_dataset, self.fmnist_splits['valid'])

        return [
            data.DataLoader(ds, shuffle=False, **self._loader_args)
            for ds in [dataset, ood_dataset]
        ]

    def test_dataloader(self) -> List[data.DataLoader]:
        dataset = datasets.MNIST(self.config.data_dir, train=False, transform=self.transform)

        # Out-of-domain dataset to evaluate entropy
        ood_dataset = datasets.FashionMNIST(self.config.data_dir, train=False, transform=self.transform)

        return [
            data.DataLoader(ds, shuffle=False, **self._loader_args)
            for ds in [dataset, ood_dataset]
        ]
