from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
from tqdm import tqdm

import wandb
import wandb.apis.public as wandb_api


# TODO: sync with training script
CURRENT_VERSION = 1


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

    fig, ax = plt.subplots()
    metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax, name=run.name)

    tqdm.write(f"   + ROC AUC: {roc_auc}")
    # FIXME: unfortunately adding plots or images in retrospect is not supported
    # run.summary['valid/entropy_roc'] = wandb.Image(fig)
    run.summary['valid/entropy_auc'] = roc_auc

    # Save the ROC plot locally
    roc_dir = os.path.join(plots_dir, 'entropy_roc')
    os.makedirs(roc_dir, exist_ok=True)
    fig.savefig(os.path.join(roc_dir, f'{run.id}_{run.name}.pdf'))
    plt.close(fig)


def main():
    parser = ArgumentParser()
    parser.add_argument('sweep', help="Path of the sweep to be processed, in the form entity_id/project_id/sweep_id.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Describe the changes without actually performing them.")
    args = parser.parse_args()

    api = wandb.Api()
    sweep: wandb_api.Sweep = api.sweep(args.sweep)

    print(f"Processing sweep {_sweep_url(sweep)}")

    plots_dir = os.path.join('update_metrics', sweep.id)

    run: wandb_api.Run
    for run in tqdm(sweep.runs):
        tqdm.write(f"Run {run.name}:")
        tqdm.write(f" - URL {run.url}")

        version = run.config.get('metrics_version', 0)
        tqdm.write(f" - current metrics version v{version}")

        if version < 1:
            tqdm.write(f" - adding entropy discrimination ROC curve")
            add_entropy_roc(run, plots_dir)

        run.config['metrics_version'] = CURRENT_VERSION

        if not args.dry_run:
            run.update()


# FIXME: should be available in wandb 0.10.9
def _sweep_url(self):
    path = self.path
    path.insert(2, "sweeps")
    return "https://app.wandb.ai/" + "/".join(path)


if __name__ == '__main__':
    main()
