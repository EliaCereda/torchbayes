import math
from argparse import ArgumentParser

import numpy as np

import os

import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning import Trainer

import torch
import torch.nn as nn

from torchbayes import bnn

import wandb

from model import Model
from data import MNISTData


# FIXME: move to proper utils file
def take(it, n):
    for x, _ in zip(it, range(n)):
        yield x


def heterogeneous_transpose(x, stack=None):
    transposed = np.asarray(x, dtype=object).T
    slices = []

    for slice in transposed:
        # Assumes that if the first element is zero-rank, then all are.
        if stack or (stack is None and len(slice) > 0 and slice[0].ndim == 0):
            slice = torch.stack(list(slice))
        else:
            slice = torch.cat(list(slice))

        slices.append(slice)

    return slices


class Task(pl.LightningModule):
    config_keys = [
        'lr', 'complexity_weight',
        'prior',
        'sigma',
        'pi', 'sigma1', 'sigma2',
        'val_samples',
    ]

    @classmethod
    def add_model_args(cls, parser):
        parser.add_argument('--lr', type=float,  default=1e-3,
                           help='Learning rate (default: %(default)s)')
        parser.add_argument('--complexity_weight', choices=bnn.complexity_weights.choices, default='uniform',
                           help='Complexity weight strategy (default: %(default)s)')

        parser.add_argument('--prior', choices=['normal', 'scale_mixture'], default='scale_mixture',
                           help='Prior distribution (default: %(default)s)')

        parser.add_argument('--sigma', type=float, default=0,
                           help='Parameter -log σ for the normal prior (default: %(default)s)')

        parser.add_argument('--pi', type=float, default=0.5,
                           help='Parameter π for the scale_mixture prior (default: %(default)s)')
        parser.add_argument('--sigma1', type=float, default=0,
                           help='Parameter -log σ1 for the scale_mixture prior (default: %(default)s)')
        parser.add_argument('--sigma2', type=float, default=6,
                           help='Parameter -log σ2 for the scale_mixture prior (default: %(default)s)')

        parser.add_argument('--val_samples', type=int, default=1,
                           help='Number of networks to sample when computing validation metrics  (default: %(default)s)')

    def __init__(self, config):
        super().__init__()

        self.hparams = {key: getattr(config, key) for key in self.config_keys}

        self.model = Model(
            [1, 28, 28], 10,
            prior=self.hparams.prior,
            sigma=math.exp(self.hparams.sigma),
            pi=self.hparams.pi,
            sigma1=math.exp(-self.hparams.sigma1),
            sigma2=math.exp(-self.hparams.sigma2)
        )

        self.complexity_weight = bnn.complexity_weights(self.hparams.complexity_weight)
        self.complexity = bnn.ComplexityCost(self.model)
        self.likelihood = nn.CrossEntropyLoss(reduction='sum')

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    @property
    def n_train_batches(self):
        # Should be zero only during pre-training validation sanity check, since
        # it's run before preparing the training data loader.
        return self.trainer.num_training_batches or 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        self.model.sample_()
        complexity = self.complexity()

        logits = self.model(inputs)
        likelihood = self.likelihood(logits, targets)

        weight = self.complexity_weight(batch_idx, self.n_train_batches)
        loss = weight * complexity + likelihood

        accuracy = self.train_accuracy(logits, targets)

        self.log('train/loss', loss)
        self.log('train/complexity', complexity)
        self.log('train/likelihood', likelihood)
        self.log('train/accuracy', accuracy)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        metrics = []
        for sample_idx in range(self.hparams.val_samples):
            sample_metrics = self.validation_step_sample(batch, sample_idx, batch_idx, dataloader_idx)
            metrics.append(sample_metrics)

        metrics = heterogeneous_transpose(metrics, stack=True)

        loss, complexity, likelihood, entropy = (metric.mean(dim=0) for metric in metrics)

        return loss, complexity, likelihood, entropy

    def validation_step_sample(self, batch, sample_idx, batch_idx, dataloader_idx):
        inputs, targets = batch

        self.model.sample_()
        complexity = self.complexity()

        logits = self.model(inputs)
        likelihood = self.likelihood(logits, targets)

        weight = self.complexity_weight(batch_idx, self.n_train_batches)
        loss = weight * complexity + likelihood

        preds = torch.argmax(logits, dim=-1)
        entropy = bnn.entropy(logits, dim=-1)

        if dataloader_idx == 0:
            self.val_accuracy.update(preds, targets)

        # Log predictions for debugging
        if batch_idx == 0 and sample_idx == 0:
            images = []
            for tensors in take(zip(inputs, targets, preds), 8):
                input, target, pred = map(lambda x: x.squeeze().cpu(), tensors)
                caption = f'target: {target}, prediction: {pred}'  # TODO: add entropy

                correctness_mask = dict(
                    mask_data=np.full_like(input, 0 if target == pred else 1),
                    class_labels={0: 'correct', 1: 'wrong'}
                )
                masks = {
                    'correctness': correctness_mask,
                }
                image = wandb.Image(input, caption=caption, masks=masks)
                images.append(image)

            ds = '_ood' if dataloader_idx == 1 else ''
            self.logger.experiment.log({f'predictions{ds}': images}, commit=False)

        return loss, complexity, likelihood, entropy

    def validation_epoch_end(self, outputs):
        # Outputs is a list of lists when using multiple validation dataloaders.
        for i, output in enumerate(outputs):
            metrics = heterogeneous_transpose(output)

            if i == 0:
                loss, complexity, likelihood = (metric.mean() for metric in metrics[:3])

                self.log('valid/loss', loss)
                self.log('valid/complexity', complexity)
                self.log('valid/likelihood', likelihood)
                self.log('valid/accuracy', self.val_accuracy.compute())

            entropy = metrics[3].cpu()

            ds = '_ood' if i == 1 else ''
            self.logger.experiment.log({f'valid/entropy{ds}': wandb.Histogram(entropy)}, commit=False)

    def forward(self, inputs):
        """Used in inference mode."""
        self.model.sample_()

        return self.model(inputs)


def log_checkpoints(trainer):
    for callback in trainer.callbacks:
        if not isinstance(callback, pl.callbacks.ModelCheckpoint):
            continue

        if callback.monitor:
            metric_name = callback.monitor
            metric_value = callback.best_model_score
        else:
            metric_name = 'epoch'
            metric_value = trainer.current_epoch

        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()

        metadata = dict(
            metric_name=metric_name,
            metric_value=metric_value
        )

        filepath = callback.best_model_path
        filename = os.path.relpath(filepath, callback.dirpath)

        # Handle metrics with a slash in the name
        metric_name = metric_name.replace('/', '.')
        filename = filename.replace('/', '.')

        name = f'{wandb.run.name}-{metric_name}'

        artifact = wandb.Artifact(name=name, type='checkpoint', metadata=metadata)
        artifact.add_file(filepath, name=filename)

        wandb.log_artifact(artifact)


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    Task.add_model_args(parser.add_argument_group('Model Hyper-Parameters'))
    MNISTData.add_data_args(parser.add_argument_group('Data Parameters'))
    args = parser.parse_args()

    callbacks = [
        pl.callbacks.EarlyStopping('valid/accuracy', mode='max'),
        pl.callbacks.ModelCheckpoint(
            filename='{epoch}-{valid/accuracy:.3f}',
            monitor='valid/accuracy', mode='max',
        ),
        pl.callbacks.ModelCheckpoint(
            filename='{epoch}-{valid/entropy_auc:.3f}',
            monitor='valid/entropy_auc', mode='max',
        )
    ]
    logger = pl.loggers.WandbLogger(job_type='train')
    trainer: Trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

    task = Task(args)
    data = MNISTData(args)

    logger.watch(task.model)
    logger.log_hyperparams(data.config)

    try:
        trainer.fit(task, data)
    except InterruptedError:
        pass

    log_checkpoints(trainer)


if __name__ == '__main__':
    main()
