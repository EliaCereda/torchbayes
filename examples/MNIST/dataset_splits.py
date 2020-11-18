from argparse import ArgumentParser
import os
import tempfile

import torch
import torch.utils.data as data
from torchvision.datasets import MNIST, FashionMNIST

import wandb

from data import MNISTData


def generate_splits(dataset, splits, out_file):
    subsets = data.random_split(dataset, splits.values())
    indices = dict(zip(splits.keys(), (s.indices for s in subsets)))

    torch.save(indices, out_file)


def main():
    parser = ArgumentParser(description='''
        Generate train/validation splits for the MNIST and FashionMNIST datasets.
    ''', parents=[
        MNISTData.add_data_args(),
    ])
    args = parser.parse_args()

    splits_sizes = {
        'train': 50000,
        'valid': 10000
    }

    wandb.init('dataset_splits')

    # MNIST
    artifact = wandb.Artifact('mnist', type='dataset_split')

    with artifact.new_file('splits.pt', mode='wb') as f:
        dataset = MNIST(args.data_dir, train=True, download=True)
        generate_splits(dataset, splits_sizes, f)

    wandb.log_artifact(artifact)

    # FashionMNIST
    artifact = wandb.Artifact('fashion_mnist', type='dataset_split')

    with artifact.new_file('splits.pt', mode='wb') as f:
        dataset = FashionMNIST(args.data_dir, train=True, download=True)
        generate_splits(dataset, splits_sizes, f)

    wandb.log_artifact(artifact)


if __name__ == '__main__':
    main()
