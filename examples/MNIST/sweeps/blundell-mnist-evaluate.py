from argparse import ArgumentParser

import wandb


def main():
    parser = ArgumentParser()
    parser.add_argument('--tag', action="extend", nargs="+", help="Select runs with any of the given tags.")
    parser.add_argument('--project', help="Path of the project, in the form entity_id/project_id.")
    args = parser.parse_args()

    api = wandb.Api()

    runs = api.runs(args.project, filters={
        'tags': {'$in': args.tag}
    })

    print(f"Selected {len(runs)} runs to evaluate...")

    checkpoints = [f'{run.id}:latest_epoch' for run in runs]

    sweep_config = {
        "name": "blundell-mnist-evaluate",
        "description": "Run evaluate.py on a set of runs",
        "program": "../../examples/MNIST/evaluate.py",
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

    sweep_id = wandb.sweep(sweep_config, project=args.project)


if __name__ == '__main__':
    main()
