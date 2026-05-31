import argparse
import os
import subprocess
import sys


def run_command(command):
    print(f'>>> {" ".join(command)}')
    subprocess.run(command, check=True)


def build_train_command(args, scale, model_name, attention_type, loss_type, channels, exp_name, extra_args=None):
    command = [
        sys.executable,
        'train.py',
        '--train-file',
        args.train_files[scale],
        '--eval-file',
        args.eval_files[scale],
        '--outputs-dir',
        args.outputs_dir,
        '--experiment-name',
        exp_name,
        '--scale',
        str(scale),
        '--model-name',
        model_name,
        '--attention-type',
        attention_type,
        '--loss-type',
        loss_type,
        '--num-channels',
        str(channels),
        '--lr',
        str(args.lr),
        '--batch-size',
        str(args.batch_size),
        '--num-epochs',
        str(args.num_epochs),
        '--num-workers',
        str(args.num_workers),
        '--seed',
        str(args.seed),
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def parse_scale_map(raw_values, key_name):
    mapping = {}
    for value in raw_values:
        scale_str, path = value.split("=", maxsplit=1)
        mapping[int(scale_str)] = path
    missing = [scale for scale in [2, 3, 4] if scale not in mapping]
    if missing:
        raise ValueError(f'Missing {key_name} for scales: {missing}')
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--train-files', nargs='+', required=True, help='Use format: 2=path 3=path 4=path')
    parser.add_argument('--eval-files', nargs='+', required=True, help='Use format: 2=path 3=path 4=path')
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--run-rgb', action='store_true', help='Run RGB perceptual and joint experiments if RGB H5 exists')
    args = parser.parse_args()

    args.train_files = parse_scale_map(args.train_files, 'train file')
    args.eval_files = parse_scale_map(args.eval_files, 'eval file')
    os.makedirs(args.outputs_dir, exist_ok=True)

    # Stage 1: Baseline reproduction x2/x3/x4
    for scale in [2, 3, 4]:
        run_command(
            build_train_command(
                args=args,
                scale=scale,
                model_name='srcnn_baseline',
                attention_type='none',
                loss_type='mse',
                channels=1,
                exp_name='baseline_repro',
            )
        )

    # Stage 3: Attention extension (SE and CBAM)
    for attention_type in ['se', 'cbam']:
        run_command(
            build_train_command(
                args=args,
                scale=3,
                model_name='srcnn_attention',
                attention_type=attention_type,
                loss_type='mse',
                channels=1,
                exp_name=f'attention_{attention_type}',
            )
        )

    # Stage 4: Perceptual loss quick compatibility test in Y space
    run_command(
        build_train_command(
            args=args,
            scale=3,
            model_name='srcnn_baseline',
            attention_type='none',
            loss_type='mse+perceptual',
            channels=1,
            exp_name='perceptual_y_compat',
            extra_args=['--perceptual-weight', '0.01', '--perceptual-layer', 'relu3_3'],
        )
    )

    if args.run_rgb:
        train_rgb = os.path.join(os.path.dirname(args.train_files[3]), 'train_x3_rgb.h5')
        eval_rgb = os.path.join(os.path.dirname(args.eval_files[3]), 'eval_x3_rgb.h5')
        if not os.path.exists(train_rgb) or not os.path.exists(eval_rgb):
            raise FileNotFoundError('RGB H5 not found. Expected train_x3_rgb.h5 and eval_x3_rgb.h5 next to Y H5 files.')

        # Stage 4b: Perceptual loss RGB mainline
        run_command(
            [
                sys.executable,
                'train.py',
                '--train-file',
                train_rgb,
                '--eval-file',
                eval_rgb,
                '--outputs-dir',
                args.outputs_dir,
                '--experiment-name',
                'perceptual_rgb',
                '--scale',
                '3',
                '--model-name',
                'srcnn_baseline',
                '--attention-type',
                'none',
                '--loss-type',
                'mse+perceptual',
                '--num-channels',
                '3',
                '--perceptual-weight',
                '0.01',
                '--perceptual-layer',
                'relu3_3',
                '--lr',
                str(args.lr),
                '--batch-size',
                str(args.batch_size),
                '--num-epochs',
                str(args.num_epochs),
                '--num-workers',
                str(args.num_workers),
                '--seed',
                str(args.seed),
            ]
        )

        # Stage 5: Joint model (Attention + Perceptual) on RGB
        run_command(
            [
                sys.executable,
                'train.py',
                '--train-file',
                train_rgb,
                '--eval-file',
                eval_rgb,
                '--outputs-dir',
                args.outputs_dir,
                '--experiment-name',
                'joint_rgb',
                '--scale',
                '3',
                '--model-name',
                'srcnn_attention',
                '--attention-type',
                'se',
                '--attention-position',
                'after_conv2',
                '--loss-type',
                'mse+perceptual',
                '--num-channels',
                '3',
                '--perceptual-weight',
                '0.01',
                '--perceptual-layer',
                'relu3_3',
                '--lr',
                str(args.lr),
                '--batch-size',
                str(args.batch_size),
                '--num-epochs',
                str(args.num_epochs),
                '--num-workers',
                str(args.num_workers),
                '--seed',
                str(args.seed),
            ]
        )
    else:
        print('\nTip: add --run-rgb to include perceptual RGB and joint runs.')


if __name__ == '__main__':
    main()
