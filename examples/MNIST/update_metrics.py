from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
from tqdm import tqdm

import wandb
import wandb.apis.public as wandb_api


# TODO: sync with training script
CURRENT_VERSION = 2


def histogram_to_preds(histogram):
    bins, values = histogram['bins'], histogram['values']

    count = np.sum(values)
    preds = np.empty(count)

    next = 0
    for i, value in enumerate(values):
        left, right = bins[i], bins[i + 1]
        avg = (left + right) / 2

        preds[next:(next + value)] = avg
        next += value

    return preds


def add_entropy_roc(run, plots_dir):
    entropy_id = run.summary_metrics['valid/entropy']
    entropy_ood = run.summary_metrics['valid/entropy_ood']

    preds_id = histogram_to_preds(entropy_id)
    preds_ood = histogram_to_preds(entropy_ood)

    targets_id = np.zeros_like(preds_id)
    targets_ood = np.ones_like(preds_ood)

    preds = np.concatenate([preds_id, preds_ood])
    targets = np.concatenate([targets_id, targets_ood])

    fpr, tpr, _ = metrics.roc_curve(targets, preds)
    roc_auc = metrics.auc(fpr, tpr)

    tqdm.write(f"   + ROC AUC: {roc_auc}")
    run.summary['valid/entropy_auc'] = roc_auc

    if plots_dir:
        fig, ax = plt.subplots()
        metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax, name=run.name)

        # FIXME: unfortunately adding plots or images in retrospect is not supported
        # run.summary['valid/entropy_roc'] = wandb.Image(fig)

        # Save the ROC plot locally
        roc_dir = os.path.join(plots_dir, 'entropy_roc')
        os.makedirs(roc_dir, exist_ok=True)
        fig.savefig(os.path.join(roc_dir, f'{run.id}_{run.name}.pdf'))
        plt.close(fig)


def add_combined_score(run):
    accuracy = run.summary['valid/accuracy']
    auc = run.summary['valid/entropy_auc']

    score = np.linalg.norm([
        1 - accuracy,
        1 - auc
    ])

    tqdm.write(f"   + Combined score: {score}")
    run.summary['valid/combined_score'] = score


def main():
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--project', help="Path of the project, in the form entity_id/project_id.")
    group.add_argument('--sweep', help="Path of the sweep to be processed, in the form entity_id/project_id/sweep_id.")

    parser.add_argument('--tag', help="Only select runs with a certain tag.")

    parser.add_argument('--dry-run', action='store_true',
                        help="Describe the changes without actually performing them.")
    args = parser.parse_args()

    api = wandb.Api()

    if args.project:
        filters = {}

        if args.tag:
            filters['tags'] = args.tag

        runs = api.runs(args.project, filters=filters)
        plots_dir = os.path.join('update_metrics', args.project)
    elif args.sweep:
        sweep: wandb_api.Sweep = api.sweep(args.sweep)

        print(f"Processing sweep {sweep.url}")

        runs = sweep.runs
        plots_dir = os.path.join('update_metrics', sweep.id)
    else:
        raise ValueError("One of --project or --sweep must be provided.")

    run: wandb_api.Run
    for run in tqdm(runs):
        version = run.config.get('metrics_version', 0)

        if version == CURRENT_VERSION:
            continue

        tqdm.write(f"Run {run.name}:")
        tqdm.write(f" - URL {run.url}")
        tqdm.write(f" - current metrics version v{version}")

        if version < 1:
            tqdm.write(f" - adding entropy discrimination ROC curve")
            add_entropy_roc(run, plots_dir)

        if version < 2:
            tqdm.write(f" - adding accuracy / AUC combined score")
            add_combined_score(run)

        run.config['metrics_version'] = CURRENT_VERSION

        if not args.dry_run:
            run.update()


if __name__ == '__main__':
    main()
