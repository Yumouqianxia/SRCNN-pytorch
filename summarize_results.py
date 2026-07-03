import argparse
import csv
import glob
import json
import os


def load_last_row(metrics_path):
    with open(metrics_path, 'r', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return rows[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--save-path', type=str, default='summary.csv')
    args = parser.parse_args()

    rows = []
    run_dirs = sorted(glob.glob(os.path.join(args.outputs_dir, '*')))
    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, 'config.json')
        metrics_path = os.path.join(run_dir, 'metrics.csv')
        if not os.path.isfile(config_path) or not os.path.isfile(metrics_path):
            continue

        with open(config_path, 'r', encoding='utf-8') as handle:
            config = json.load(handle)
        last_row = load_last_row(metrics_path)
        if last_row is None:
            continue

        rows.append(
            {
                'run_name': os.path.basename(run_dir),
                'scale': config.get('scale'),
                'model_name': config.get('model_name'),
                'attention_type': config.get('attention_type'),
                'loss_type': config.get('loss_type'),
                'num_channels': config.get('num_channels'),
                'kernel_sizes': '-'.join(str(size) for size in config.get('kernel_sizes', [9, 5, 5])),
                'parameter_count': config.get('parameter_count'),
                'last_epoch': last_row['epoch'],
                'eval_psnr': last_row['eval_psnr'],
                'eval_ssim': last_row['eval_ssim'],
            }
        )

    if not rows:
        print('No runs found.')
        return

    save_path = args.save_path
    if not os.path.isabs(save_path):
        save_path = os.path.join(args.outputs_dir, save_path)

    with open(save_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved summary to {save_path}')


if __name__ == '__main__':
    main()
