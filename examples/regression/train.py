from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import AttributeDict

import torch
from torch import nn

from torchbayes import bnn
from torchbayes.utils import heterogeneous_transpose

from typing import Union, Dict, Any

from model import Model
from data import RegressionData


class CauchyLoss(nn.Module):
    def __init__(self, c=0.1, reduction='mean'):
        super().__init__()

        self.c = c

        if reduction == 'none':
            self.reduce = lambda x: x
        elif reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum
        else:
            raise ValueError(f"Unsupported reduction value '{reduction}'.")

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        delta = input - target
        loss = torch.log1p(0.5 * (delta / self.c) ** 2)
        loss = self.reduce(loss)

        return loss


class Task(pl.LightningModule):
    hparam_keys = [
        'lr', 'val_samples'
    ]

    @classmethod
    def add_model_args(cls, args=None):
        parser = ArgumentParser(add_help=False)
        group = parser.add_argument_group('Model Hyper-Parameters')
        group.add_argument('--lr', type=float,  default=1e-3,
                           help='Learning rate (default: %(default)s)')
        group.add_argument('--val_samples', default=16,
                           help='Number of networks to sample when computing validation metrics (default: %(default)s)')
        return parser

    def __init__(self, hparams: Union[Dict[str, Any], Namespace, AttributeDict]):
        super().__init__()

        if isinstance(hparams, Namespace):
            hparams = vars(hparams)

        hparams = {key: hparams.get(key, None) for key in self.hparam_keys}

        self.save_hyperparameters(hparams)

        self.model = Model(1, 1)

        self.complexity_weight = bnn.complexity_weights('uniform')
        self.complexity = bnn.ComplexityCost(self.model)
        self.likelihood = CauchyLoss(reduction='sum')

    @property
    def _n_train_batches(self):
        # Should be zero only during pre-training validation sanity check, since
        # it's run before preparing the training data loader.
        return self.trainer.num_training_batches or 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        self.model.sample_()
        complexity = self.complexity()

        preds = self.model(inputs)
        likelihood = self.likelihood(preds, targets)

        weight = self.complexity_weight(batch_idx, self._n_train_batches)
        loss = weight * complexity + likelihood

        self.log('train/loss', loss)
        self.log('train/complexity', complexity)
        self.log('train/likelihood', likelihood)

        return loss

    def _validation_step_sample(self, inputs, targets, sample_idx, batch_idx):
        self.model.sample_()
        complexity = self.complexity()

        preds = self.model(inputs)
        likelihood = self.likelihood(preds, targets)

        weight = self.complexity_weight(batch_idx, self._n_train_batches)
        loss = weight * complexity + likelihood

        return preds, loss, complexity, likelihood

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Sample `val_samples` networks from the posterior distribution of the model
        # and compute the predictions of each one.
        val_samples = self.hparams.val_samples or 1
        outputs = []
        for sample_idx in range(val_samples):
            out = self._validation_step_sample(inputs, targets, sample_idx, batch_idx)
            outputs.append(out)

        # Average the predictions of the ensemble.
        outputs = heterogeneous_transpose(outputs, dim=-1)
        preds, (loss, complexity, likelihood) = outputs[0], (output.mean() for output in outputs[1:])

        return preds, loss, complexity, likelihood

    def validation_epoch_end(self, outputs):
        outputs = heterogeneous_transpose(outputs)
        preds, (loss, complexity, likelihood) = outputs[0], (metric.mean() for metric in outputs[1:])

        if not self.trainer.running_sanity_check:
            self.trainer.datamodule.visualize(preds)

        self.log('valid/loss', loss)
        self.log('valid/complexity', complexity)
        self.log('valid/likelihood', likelihood)

    def forward(self, inputs):
        """Used in inference mode."""
        self.model.sample_()

        return self.model(inputs)


def main():
    parser = ArgumentParser(parents=[
        Trainer.add_argparse_args(ArgumentParser(add_help=False)),
        RegressionData.add_data_args(),
        Task.add_model_args(),
    ])

    default_args = Namespace(
        check_val_every_n_epoch=50,
    )
    args = parser.parse_args(namespace=default_args)

    trainer = Trainer.from_argparse_args(args)

    task = Task(args)
    data = RegressionData(args)

    try:
        trainer.validate(task, datamodule=data)
        trainer.fit(task, data)
    except InterruptedError:
        pass

    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
