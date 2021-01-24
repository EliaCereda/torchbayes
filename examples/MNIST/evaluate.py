from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning import Trainer
import sklearn.metrics as m
import sys
import torch
import wandb

from train import Task
from data import MNISTData


def main():
    parser = ArgumentParser(parents=[
        Trainer.add_argparse_args(ArgumentParser(add_help=False)),
        MNISTData.add_data_args(),
    ])
    parser.add_argument('--checkpoint',
                        help='W&B artifact name of the checkpoint to be loaded.')
    args = parser.parse_args()

    logger = pl.loggers.WandbLogger(job_type='evaluate')
    trainer: Trainer = Trainer.from_argparse_args(args, logger=logger)

    if not hasattr(trainer, 'validate'):
        print("Trainer.validate is not available in your version of PyTorch Lighting.", file=sys.stderr)
        print("Check this pull request for updates: https://github.com/PyTorchLightning/pytorch-lightning/pull/4707", file=sys.stderr)
        exit(1)

    evaluate_run = logger.experiment
    artifact = evaluate_run.use_artifact(args.checkpoint, type='checkpoint')
    checkpoint_path = artifact.file()

    api = wandb.Api()
    train_id = artifact.name.split(':')[0]
    train_run = api.run(f'{artifact.entity}/{artifact.project}/{train_id}')
    evaluate_run.notes = f'Training Run: {train_run.name}, {train_run.url}'

    # Load the model to CPU memory, then leave it to Trainer to move it to the
    # device that will be used for evaluation.
    device = torch.device('cpu')

    # Must be strict=False because `Metric`s were persistent by default but now
    # they're not anymore. TODO: search for a better way
    task = Task.load_from_checkpoint(checkpoint_path, map_location=device, strict=False, log_predictions=True)
    data = MNISTData(args)

    logger.watch(task.model, log='all')
    logger.log_hyperparams(data.config)

    output = trainer.validate(task, datamodule=data)

    entropy_id = output[0]['valid/entropy/mnist/table/dataloader_idx_0']
    entropy_ood = output[1]['valid/entropy/fashion_mnist/table/dataloader_idx_1']

    ood_entropy_auc(entropy_id, entropy_ood)


def ood_entropy_auc(entropy_id, entropy_ood):
    target_id = np.zeros_like(entropy_id)
    target_ood = np.ones_like(entropy_ood)

    entropy = np.concatenate([entropy_id, entropy_ood])
    targets = np.concatenate([target_id, target_ood])

    ax1: plt.Axes
    ax2: plt.Axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.5))

    ax1.hist(entropy_id, 96, alpha=0.8, label="in domain (MNIST)")
    ax1.hist(entropy_ood, 96, alpha=0.8, label="out-of-domain (Fashion-MNIST)")

    ax1.legend()
    ax1.grid(True)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect((x1 - x0) / (y1 - y0))
    ax1.set_axisbelow(True)
    ax1.set_xlabel("Entropy")
    ax1.set_ylabel("Sample count")

    fpr, tpr, _ = m.roc_curve(targets, entropy)
    roc_auc = m.auc(fpr, tpr)

    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

    ax2.legend()
    ax2.grid(True)
    ax2.axis('square')
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate (FPR)")
    ax2.set_ylabel("True Positive Rate (TPR)")

    wandb.log({f'valid/ood_entropy_auc': wandb.Image(fig)})
    fig.savefig(f'valid/ood_entropy_auc.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
