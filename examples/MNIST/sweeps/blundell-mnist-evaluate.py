from argparse import ArgumentParser

import wandb


def main():
    parser = ArgumentParser()
    parser.add_argument('--tag', help="Select runs with the given tag.")
    parser.add_argument('--project', help="Path of the project, in the form entity_id/project_id.")
    args = parser.parse_args()

    api = wandb.Api()

    runs = api.runs(args.project, filters={
        'tags': args.tag
    })

    checkpoints = [f'{run.id}:latest_epoch' for run in runs]

    sweep_config = {
        "name": "blundell-mnist-evaluate",
        "description": "Run evaluate.py on a set of runs",
        "method": "grid",
        "parameters": {
            "checkpoint": {
                "values": checkpoints
            }
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--gpus",
            "1",
            "${args}"
        ]
    }

    sweep_id = wandb.sweep(sweep_config)


if __name__ == '__main__':
    main()
