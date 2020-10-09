import argparse
import os

import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

import pytorch_lightning as pl
import pytorch_lightning.loggers

from torchbayes import bnn

import wandb

from model import Model


class LitModel(pl.LightningModule):
    def __init__(self, data_dir=None, batch_size=128, lr=1e-3):
        super().__init__()

        if data_dir is None:
            example_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(example_dir, 'data')

        self.batch_size = batch_size
        self.lr = lr

        self.model = Model([1, 28, 28], 10)

        self.complexity = bnn.ComplexityCost(self.model)
        self.likelihood = nn.CrossEntropyLoss(reduction='sum')

    def prepare_data(self):
        datasets.MNIST(self.data_dir, download=True)

    @property
    def _loader_args(self):
        return dict(
            batch_size=self.batch_size,
            num_workers=6,  # FIXME: read number of CPUs
            pin_memory=self.on_gpu
        )

    def train_dataloader(self):
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(self.data_dir, train=True, transform=transform)

        return data.DataLoader(dataset, shuffle=True, **self._loader_args)

    def val_dataloader(self):
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(self.data_dir, train=False, transform=transform)

        return data.DataLoader(dataset, shuffle=False, **self._loader_args)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        self.model.sample_()
        complexity = self.complexity()

        logits = self.model(inputs)
        likelihood = self.likelihood(logits, targets)

        weight = 2 ** (-batch_idx - 1)
        loss = weight * complexity + likelihood

        self.log('train/loss', loss)
        self.log('train/complexity', complexity)
        self.log('train/likelihood', likelihood)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        self.model.sample_()
        complexity = self.complexity()

        logits = self.model(inputs)
        likelihood = self.likelihood(logits, targets)

        weight = 2 ** (-batch_idx - 1)
        loss = weight * complexity + likelihood

        return loss, complexity, likelihood

    def validation_epoch_end(self, outputs):
        metrics = torch.as_tensor(outputs).t()
        loss, complexity, likelihood = metrics.mean(dim=1)

        self.log('valid/loss', loss)
        self.log('valid/complexity', complexity)
        self.log('valid/likelihood', likelihood)

    def forward(self, inputs):
        self.model.sample_()

        return self.model(inputs)


def main(args):
    logger = pl.loggers.WandbLogger(job_type='debug')
    logger.experiment # FIXME: only called for side effects

    model = LitModel()
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    # artifact_dir = wandb.use_artifact('model:v2').download()  # '~/Downloads'
    # model.load_from_checkpoint(os.path.join(artifact_dir, 'model.ckpt'))

    try:
        trainer.fit(model)
    except InterruptedError:
        pass

    trainer.save_checkpoint('model.ckpt')
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('model.ckpt')
    wandb.log_artifact(artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
