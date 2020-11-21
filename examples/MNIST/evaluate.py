from argparse import ArgumentParser
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning import Trainer
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
    parser.add_argument('checkpoint',
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
    task = Task.load_from_checkpoint(checkpoint_path, map_location=device, strict=False)
    data = MNISTData(args)

    logger.watch(task.model, log='all')
    logger.log_hyperparams(data.config)

    try:
        trainer.validate(task, datamodule=data)
    except InterruptedError:
        pass


if __name__ == '__main__':
    main()
