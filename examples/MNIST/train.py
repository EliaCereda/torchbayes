import math
from argparse import ArgumentParser, Namespace

import numpy as np

import os

import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import AttributeDict

import re

from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbayes import bnn

from typing import overload, Union, Dict, Any

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
        'lr', 'approach',
        'complexity_weight', 'val_samples', 'prior',
        'sigma',
        'pi', 'sigma1', 'sigma2',
    ]

    @classmethod
    def add_model_args(cls, args=None):
        parser = ArgumentParser(add_help=False)
        group = parser.add_argument_group('Model Hyper-Parameters')
        group.add_argument('--lr', type=float,  default=1e-3,
                           help='Learning rate (default: %(default)s)')
        group.add_argument('--approach', choices=['traditional', 'bnn'], default='bnn',
                           help='Approach to NN training (default: %(default)s)')

        params, args = parser.parse_known_args(args)

        cls.add_approach_args(parser, params.approach, args)

        return parser

    @classmethod
    def add_approach_args(cls, parser, approach, args=None):
        group = parser.add_argument_group(f"Approach '{approach}'")

        if approach == 'traditional':
            pass
        elif approach == 'bnn':
            group.add_argument('--complexity_weight', choices=bnn.complexity_weights.choices, default='uniform',
                               help='Complexity weight strategy (default: %(default)s)')
            group.add_argument('--val_samples', type=int, default=1,
                               help='Number of networks to sample when computing validation metrics (default: %(default)s)')
            group.add_argument('--prior', choices=['normal', 'scale_mixture'], default='scale_mixture',
                               help='Prior distribution (default: %(default)s)')

            params, args = parser.parse_known_args(args)

            cls.add_prior_args(parser, params.prior, args)
        else:
            raise ValueError(f"Unsupported approach '{approach}'.")

    @classmethod
    def add_prior_args(cls, parser, prior, args=None):
        group = parser.add_argument_group(f"Prior Distribution '{prior}'")

        if prior == 'normal':
            group.add_argument('--sigma', type=float, default=0,
                               help='Parameter -log σ for the normal prior (default: %(default)s)')
        elif prior == 'scale_mixture':
            group.add_argument('--pi', type=float, default=0.5,
                               help='Parameter π for the scale_mixture prior (default: %(default)s)')
            group.add_argument('--sigma1', type=float, default=0,
                               help='Parameter -log σ1 for the scale_mixture prior (default: %(default)s)')
            group.add_argument('--sigma2', type=float, default=6,
                               help='Parameter -log σ2 for the scale_mixture prior (default: %(default)s)')
        else:
            raise ValueError(f"Unsupported prior distribution '{prior}'.")

    @overload
    def __init__(self, hparams: Union[Dict[str, Any], Namespace, AttributeDict]):
        ...

    @overload
    def __init__(self, **_kwargs):
        """
        Private initializer, do not call directly.
        
        It is used to load the hyperparameters from old checkpoints, which
        didn't store the name of hparams argument correctly.
        """
        ...

    def __init__(self, hparams=None, **_kwargs):
        super().__init__()

        if hparams is None:
            hparams = _kwargs

        if isinstance(hparams, Namespace):
            hparams = vars(hparams)

        hparams = {key: hparams.get(key, None) for key in self.config_keys}

        self.save_hyperparameters(hparams)

        self.model = Model(
            [1, 28, 28], 10,
            approach=self.hparams.approach,
            prior=self.hparams.prior,
            sigma=self.hparams.sigma,
            pi=self.hparams.pi,
            sigma1=self.hparams.sigma1,
            sigma2=self.hparams.sigma2
        )

        self.complexity_weight = bnn.complexity_weights(self.hparams.complexity_weight)
        self.complexity = bnn.ComplexityCost(self.model)
        self.likelihood = nn.CrossEntropyLoss(reduction='sum')

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    @property
    def _n_train_batches(self):
        # Should be zero only during pre-training validation sanity check, since
        # it's run before preparing the training data loader.
        return self.trainer.num_training_batches or 1

    @property
    def dataset_keys(self):
        return self.trainer.datamodule.dataset_keys

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        self.model.sample_()
        complexity = self.complexity()

        logits = self.model(inputs)
        likelihood = self.likelihood(logits, targets)

        weight = self.complexity_weight(batch_idx, self._n_train_batches)
        loss = weight * complexity + likelihood

        accuracy = self.train_accuracy(logits, targets)

        self.log('train/loss', loss)
        self.log('train/complexity', complexity)
        self.log('train/likelihood', likelihood)
        self.log('train/accuracy', accuracy)

        return loss

    def _validation_step_sample(self, inputs, targets, sample_idx, batch_idx, dl_idx):
        self.model.sample_()
        complexity = self.complexity()

        logits = self.model(inputs)
        likelihood = self.likelihood(logits, targets)

        weight = self.complexity_weight(batch_idx, self._n_train_batches)
        loss = weight * complexity + likelihood

        # Returning probs instead of logits because `validation_step` needs to
        # compute an average across all samples. Would it be possible to average
        # the logits pre-softmax and obtain the same result?
        probs = F.softmax(logits, dim=-1)

        return probs, loss, complexity, likelihood

    def validation_step(self, batch, batch_idx, dl_idx):
        inputs, targets = batch
        ds = self.dataset_keys[dl_idx]

        # Sample `val_samples` networks from the posterior distribution of the model
        # and compute the predictions of each one.
        val_samples = self.hparams.val_samples or 1
        outputs = []
        for sample_idx in range(val_samples):
            out = self._validation_step_sample(inputs, targets, sample_idx, batch_idx, dl_idx)
            outputs.append(out)

        # Average the predictions of the ensemble.
        outputs = heterogeneous_transpose(outputs, stack=True)
        probs, loss, complexity, likelihood = (output.mean(dim=0) for output in outputs)

        # Compute metrics from the ensemble predictions.
        preds = torch.argmax(probs, dim=-1)
        entropy = bnn.entropy(probs, dim=-1)

        if ds == 'mnist':
            self.val_accuracy.update(preds, targets)

        # Log a certain number of predictions to wandb for debugging.
        n_images = 8
        if batch_idx == 0:
            images = []
            for tensors in take(zip(inputs, targets, preds, entropy), n_images):
                input, target, pred, ent = map(lambda x: x.squeeze().cpu(), tensors)
                caption = f'target: {target}, prediction: {pred}, entropy: {ent:.3f}'

                correctness_mask = dict(
                    mask_data=np.full_like(input, 0 if target == pred else 1),
                    class_labels={0: 'correct', 1: 'wrong'}
                )
                masks = {
                    'correctness': correctness_mask,
                }
                image = wandb.Image(input, caption=caption, masks=masks)
                images.append(image)

            self.logger.experiment.log({f'valid/predictions/{ds}': images}, commit=False)

        return loss, complexity, likelihood, entropy

    def validation_epoch_end(self, outputs):
        entropies = []
        targets = []

        # Outputs is a list of lists when using multiple validation dataloaders.
        for i, output in enumerate(outputs):
            output = heterogeneous_transpose(output)
            ds = self.dataset_keys[i]

            if ds == 'mnist':
                loss, complexity, likelihood = (metric.mean() for metric in output[:3])

                self.log('valid/loss', loss)
                self.log('valid/complexity', complexity)
                self.log('valid/likelihood', likelihood)
                self.log('valid/accuracy', self.val_accuracy.compute())

            entropy = output[3].cpu()
            entropies.append(entropy)
            targets.append(np.full_like(entropy, i))

            self.logger.experiment.log({f'valid/entropy/{ds}': wandb.Histogram(entropy)}, commit=False)

        # Compute the AUROC metric to measure the power of the entropy to
        # discriminate the two datasets.
        entropies = np.concatenate(entropies)
        targets = np.concatenate(targets)

        fpr, tpr, _ = metrics.roc_curve(targets, entropies)
        roc_auc = metrics.auc(fpr, tpr)

        self.log('valid/entropy_auc', roc_auc)

    def forward(self, inputs):
        """Used in inference mode."""
        self.model.sample_()

        return self.model(inputs)


def log_checkpoints(trainer, save=False, log=True):
    for callback in trainer.callbacks:
        if not isinstance(callback, pl.callbacks.ModelCheckpoint):
            continue

        file_path = callback.best_model_path

        # callback.best_model_path is an empty string until the first checkpoint
        # has been saved.
        if not file_path:
            continue

        file_name = os.path.relpath(file_path, callback.dirpath)

        matches = re.match(r"^epoch=(\d+)(-.+)?\.ckpt$", file_name)
        epoch = matches.group(1) if matches else None

        if callback.monitor:
            metric_name = callback.monitor
            metric_value = callback.best_model_score
        else:
            metric_name = 'latest_epoch'
            metric_value = epoch

        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()

        metadata = dict(
            file_name=file_name,
            metric_name=metric_name,
            metric_value=metric_value,
            epoch=epoch
        )

        # Handle metrics with a slash in the name
        metric_slug = metric_name.replace('/', '_')

        if save:
            artifact_name = f'{wandb.run.id}'
            artifact = wandb.Artifact(name=artifact_name, type='checkpoint', metadata=metadata)
            artifact.add_file(file_path, name='checkpoint.ckpt')

            wandb.log_artifact(artifact, aliases=[metric_slug])

        if log and callback.monitor:
            wandb.summary[f'{metric_name}/best_value'] = metric_value
            wandb.summary[f'{metric_name}/best_epoch'] = epoch


def main():
    parser = ArgumentParser(parents=[
        Trainer.add_argparse_args(ArgumentParser(add_help=False)),
        MNISTData.add_data_args(),
        Task.add_model_args(),
    ])
    parser.add_argument('--save_checkpoints', action='store_true',
                        help="Store the model checkpoints to WandB (default: %(default)s)")
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
        ),
        pl.callbacks.ModelCheckpoint(
            filename='{epoch}',
            # Saves the checkpoint of the latest epoch
        ),
    ]
    logger = pl.loggers.WandbLogger(job_type='train')
    trainer: Trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

    task = Task(args)
    data = MNISTData(args)

    logger.watch(task.model, log='all')
    logger.log_hyperparams(data.config)

    try:
        trainer.fit(task, data)
    except InterruptedError:
        pass

    log_checkpoints(trainer, save=args.save_checkpoints)


if __name__ == '__main__':
    main()
