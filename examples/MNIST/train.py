import os
from argparse import ArgumentParser

import numpy as np

import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

import pytorch_lightning as pl
import pytorch_lightning.loggers

from torchbayes import bnn

import wandb

from model import Model


# FIXME: move to proper utils file
def take(it, n):
    for x, _ in zip(it, range(n)):
        yield x


class Task(pl.LightningModule):
    hparam_keys = ['batch_size', 'lr']

    @classmethod
    def add_model_args(cls, parent: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent], add_help=False)
        group = parser.add_argument_group('Model Hyper-Parameters')
        group.add_argument('--batch_size', type=int, default=128,
                           help='Batch size (default: %(default)s)')
        group.add_argument('--lr', type=float,  default=1e-3,
                           help='Learning rate (default: %(default)s)')

        return parser

    def __init__(self, hparams, data_dir=None):
        super().__init__()

        if data_dir is None:
            self.data_dir = os.path.join(os.getcwd(), 'data')

        self.hparams = {key: getattr(hparams, key) for key in self.hparam_keys}

        self.model = Model([1, 28, 28], 10)

        self.complexity = bnn.ComplexityCost(self.model)
        self.likelihood = nn.CrossEntropyLoss(reduction='sum')

        self.val_accuracy = pl.metrics.Accuracy()

    def prepare_data(self):
        datasets.MNIST(self.data_dir, download=True)

    @property
    def _loader_args(self):
        return dict(
            batch_size=self.hparams.batch_size,
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
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

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

        preds = torch.argmax(logits, dim=-1)
        self.val_accuracy.update(preds, targets)

        # Log predictions for debugging
        if batch_idx == 0:
            images = []
            for tensors in take(zip(inputs, targets, preds), 8):
                input, target, pred = map(lambda x: x.squeeze().cpu(), tensors)
                caption = f'target: {target}, prediction: {pred}'

                correctness_mask = dict(
                    mask_data=np.full_like(input, 1 if target == pred else 0),
                    class_labels={0: 'wrong', 1: 'correct'}
                )
                masks = {
                    'correctness': correctness_mask,
                }
                image = wandb.Image(input, caption=caption, masks=masks)
                images.append(image)

            self.logger.experiment.log({'predictions': images}, commit=False)

        return loss, complexity, likelihood

    def validation_epoch_end(self, outputs):
        metrics = torch.as_tensor(outputs).t()
        loss, complexity, likelihood = metrics.mean(dim=1)

        self.log('valid/loss', loss)
        self.log('valid/complexity', complexity)
        self.log('valid/likelihood', likelihood)
        self.log('valid/accuracy', self.val_accuracy.compute())

    def forward(self, inputs):
        """Used in inference mode."""
        self.model.sample_()

        return self.model(inputs)


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Task.add_model_args(parser)
    args = parser.parse_args()

    logger = pl.loggers.WandbLogger()

    task = Task(args)
    trainer: Trainer = Trainer.from_argparse_args(args, logger=logger)

    try:
        trainer.fit(task)
    except InterruptedError:
        pass


if __name__ == '__main__':
    main()
