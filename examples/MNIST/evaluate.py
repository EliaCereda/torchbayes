from argparse import ArgumentParser
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning import Trainer
import torch

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

    wandb = logger.experiment
    checkpoint_path = wandb.use_artifact(args.checkpoint, type='checkpoint').file()

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
