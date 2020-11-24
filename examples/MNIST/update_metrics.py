from argparse import ArgumentParser
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn import metrics
import subprocess
import tempfile
import time
from tqdm import tqdm
import wandb
import wandb.apis.public as wandb_api
import warnings


# TODO: sync with training script
CURRENT_VERSION = 4


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
    # Using the undocumented summary_metrics, because summary doesn't contain
    # histograms.
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

    if 'valid/entropy_auc' not in run.summary:
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

    if 'valid/combined_score' not in run.summary:
        tqdm.write(f"   + Combined score: {score}")
        run.summary['valid/combined_score'] = score


def add_default_approach(run):
    if 'approach' not in run.config:
        run.config['approach'] = 'bnn'


def add_checkpoint_artifact(run, api: wandb.Api, dry_run):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download checkpoints from Google Drive
        cmd = ['rclone', 'copy', f'drive:data/runs/{run.id}/checkpoints', tmp_dir]
        subprocess.run(cmd, check=True)

        artifacts = []
        artifact_name = run.id
        file_paths = glob.iglob(os.path.join(tmp_dir, '**/*.ckpt'), recursive=True)

        for file_path in file_paths:
            file_name = os.path.relpath(file_path, tmp_dir)

            matches = re.match(r"^epoch=(\d+)(-.+)?\.ckpt$", file_name)
            epoch = matches.group(1) if matches else None

            if matches.group(2) is None:
                metric_name = 'latest_epoch'
                metric_value = epoch
            else:
                warnings.warn("Support for checkpoints tracking a metric has not been implemented yet, skipping.")
                continue

            metadata = dict(
                file_name=file_name,
                metric_name=metric_name,
                metric_value=metric_value,
                epoch=epoch
            )

            # Handle metrics with a slash in the name
            metric_slug = metric_name.replace('/', '_')

            artifact = wandb.Artifact(artifact_name, type='checkpoint', metadata=metadata)
            artifact.add_file(file_path, name='checkpoint.ckpt')
            wandb.log_artifact(artifact, aliases=[metric_slug])

            artifacts.append((artifact, metric_slug))

        # Wait until each artifact has been uploaded and link it to its run.
        for artifact, metric_slug in artifacts:
            name = f'{artifact.name}:{metric_slug}'
            manifest = wait_pending_artifact(api, name, type='checkpoint')

            if not dry_run:
                run.log_artifact(manifest)


def wait_pending_artifact(api: wandb.Api, name: str, type: str = None) -> wandb_api.Artifact:
    # Source: https://github.com/wandb/client/issues/1486
    while True:
        try:
            return api.artifact(name, type)
        except wandb.errors.error.CommError:
            # Back-off and retry
            time.sleep(5)
            pass


def main():
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sweep', help="Select runs from the given sweep.")
    group.add_argument('--tag', help="Select runs with the given tag.")

    parser.add_argument('--project', help="Path of the project, in the form entity_id/project_id.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Describe the changes without actually performing them.")
    args = parser.parse_args()

    wandb.init(job_type='update_metrics', project=args.project)

    overrides = {}

    if args.project:
        overrides['project'] = args.project

    api = wandb.Api(overrides)

    if args.tag:
        runs = api.runs(args.project, filters={
            'tags': args.tag
        })

        plots_dir = os.path.join('update_metrics', args.tag)
    elif args.sweep:
        sweep: wandb_api.Sweep = api.sweep(args.sweep)

        print(f"Processing sweep {sweep.url}")

        runs = sweep.runs
        plots_dir = os.path.join('update_metrics', sweep.id)
    else:
        raise ValueError("One of --tag or --sweep must be provided.")

    run: wandb_api.Run
    for run in tqdm(runs):
        version = run.config.get('metrics_version', 0)

        if version == CURRENT_VERSION:
            continue

        tqdm.write(f"Run {run.name}:")
        tqdm.write(f" - URL {run.url}")
        tqdm.write(f" - current metrics version v{version}")

        try:
            if version < 1:
                tqdm.write(f" - adding entropy discrimination ROC curve")
                add_entropy_roc(run, plots_dir)

            if version < 2:
                tqdm.write(f" - adding accuracy / AUC combined score")
                add_combined_score(run)

            if version < 3:
                tqdm.write(f" - adding default approach config key")
                add_default_approach(run)

            if version < 4:
                tqdm.write(f" - adding checkpoint artifact")
                add_checkpoint_artifact(run, api, args.dry_run)

            run.config['metrics_version'] = CURRENT_VERSION

            if not args.dry_run:
                run.update()

        except Exception as e:
            tqdm.write(f" - ERROR: {e}")


if __name__ == '__main__':
    main()
