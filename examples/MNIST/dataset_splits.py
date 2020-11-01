from argparse import ArgumentParser
import os
import tempfile

import torch
import torch.utils.data as data
from torchvision.datasets import MNIST, FashionMNIST

import wandb

from data import MNISTData


def generate_splits(dataset, splits, out_path):
    subsets = data.random_split(dataset, splits.values())
    indices = dict(zip(splits.keys(), (s.indices for s in subsets)))

    torch.save(indices, out_path)


def main():
    parser = ArgumentParser(description='''
        Generate train/validation splits for the MNIST and FashionMNIST datasets.
    ''')
    MNISTData.add_data_args(parser.add_argument_group('Data Parameters'))
    args = parser.parse_args()

    splits_sizes = {
        'train': 50000,
        'valid': 10000
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        wandb.init('dataset_splits')

        # MNIST
        splits_path = os.path.join(tmp_dir, 'mnist-splits.pt')
        dataset = MNIST(args.data_dir, train=True, download=True)
        generate_splits(dataset, splits_sizes, splits_path)

        artifact = wandb.Artifact('mnist', type='dataset_split')
        artifact.add_file(splits_path, 'splits.pt')
        wandb.log_artifact(artifact)

        # FashionMNIST
        splits_path = os.path.join(tmp_dir, 'fmnist-splits.pt')
        dataset = FashionMNIST(args.data_dir, train=True, download=True)
        generate_splits(dataset, splits_sizes, splits_path)

        artifact = wandb.Artifact('fashion_mnist', type='dataset_split')
        artifact.add_file(splits_path, 'splits.pt')
        wandb.log_artifact(artifact)

        # Wait until all artifacts have been uploaded, then clean up the
        # temporary directory.
        wandb.finish()


if __name__ == '__main__':
    main()
