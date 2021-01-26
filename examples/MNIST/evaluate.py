from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
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

    os.makedirs('valid', exist_ok=True)

    ood_entropy_auc(entropy_id, entropy_ood)


def ood_entropy_auc(entropy_id, entropy_ood, broken=False):
    target_id = np.zeros_like(entropy_id)
    target_ood = np.ones_like(entropy_ood)

    entropy = np.concatenate([entropy_id, entropy_ood])
    targets = np.concatenate([target_id, target_ood])

    # Histogram plot with a broken axis
    # Source: https://matplotlib.org/3.3.3/gallery/subplots_axes_and_figures/broken_axis.html
    fig: plt.Figure = plt.figure(figsize=(9.0, 4.5))
    gs = plt.GridSpec(1, 2, wspace=0.3, figure=fig)

    if broken:
        gs1 = gs[0].subgridspec(2, 1, hspace=-0.15)
        ax11: plt.Axes = fig.add_subplot(gs1[0])
        ax12: plt.Axes = fig.add_subplot(gs1[1], sharex=ax11)
        axs1 = [ax11, ax12]
    else:
        ax1: plt.Axes = fig.add_subplot(gs[0])
        axs1 = [ax1]

    for ax in axs1:
        ax.hist(entropy_id, 96, alpha=0.8, label="in domain (MNIST)")
        ax.hist(entropy_ood, 96, alpha=0.8, label="out of domain (Fashion-MNIST)")

        ax.grid(True)
        ax.set_axisbelow(True)

    if broken:
        ax11.legend()

        # Show y label centered at the bottom of the upper subplot, so it appears at
        # the center of the overall figure.
        # Using the upper subplot because it should generally have longer tick
        # labels, so the y label is positioned correctly.
        ax11.set_ylabel("Sample count", y=0, horizontalalignment='center')
        ax12.set_xlabel("Entropy")

        # zoom-in / limit the view to different portions of the data
        max_inlier = 800
        y0, y1 = ax11.get_ylim()
        ax11.set_ylim(bottom=y1 - max_inlier)  # outliers only
        ax12.set_ylim(0, max_inlier)  # most of the data

        # Modify the aspect ratio so that the plot is squared like the ROC curve
        for ax in [ax11, ax12]:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect(0.5 * (x1 - x0) / (y1 - y0))

        # hide the spines between ax and ax2
        ax11.spines['bottom'].set_visible(False)
        ax12.spines['top'].set_visible(False)
        ax11.xaxis.set_ticks_position('none')
        ax11.tick_params(labelbottom=False)  # don't put tick labels at the top
        ax12.xaxis.tick_bottom()

        # Cut-out slanted lines
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        axis_width = plt.rcParams['axes.linewidth']
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linewidth=axis_width,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax11.plot([0, 1], [0, 0], transform=ax11.transAxes, **kwargs)
        ax12.plot([0, 1], [1, 1], transform=ax12.transAxes, **kwargs)
    else:
        # Modify the aspect ratio so that the plot is squared like the ROC curve
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax1.set_aspect((x1 - x0) / (y1 - y0))

        ax1.legend()
        ax1.set_ylabel("Sample count")
        ax1.set_xlabel("Entropy")

    ## ROC curve
    ax2: plt.Axes = fig.add_subplot(gs[1])

    fpr, tpr, _ = m.roc_curve(targets, entropy)
    roc_auc = m.auc(fpr, tpr)

    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

    ax2.legend(loc='lower right')
    ax2.grid(True)
    ax2.axis('square')
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate (FPR)")
    ax2.set_ylabel("True Positive Rate (TPR)")

    wandb.log({f'valid/ood_entropy_auc': wandb.Image(fig)})
    fig.savefig(f'valid/ood_entropy_auc.pdf')
    fig.savefig(f'valid/ood_entropy_auc.png', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
