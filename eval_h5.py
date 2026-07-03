import argparse
import csv
import json
import os

import h5py
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from datasets import EvalDataset, TrainDataset
from models import create_model
from train import run_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--prefetch-factor', type=int, default=4)
    parser.add_argument('--persistent-workers', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config_file, 'r', encoding='utf-8') as handle:
        config = json.load(handle)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        model_name=config.get('model_name', 'srcnn_baseline'),
        num_channels=int(config.get('num_channels', 1)),
        attention_type=config.get('attention_type', 'none'),
        attention_position=config.get('attention_position', 'after_conv2'),
        kernel_sizes=tuple(config.get('kernel_sizes', [9, 5, 5])),
    ).to(device)
    model.load_state_dict(torch.load(args.weights_file, map_location=device))

    with h5py.File(args.eval_file, 'r') as handle:
        is_group_eval = hasattr(handle['lr'], 'keys')
    dataset = EvalDataset(args.eval_file) if is_group_eval else TrainDataset(args.eval_file)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1 if is_group_eval else args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
        loader_kwargs['persistent_workers'] = args.persistent_workers
    dataloader = DataLoader(**loader_kwargs)

    eval_psnr, eval_ssim = run_eval(model, dataloader, device, calc_ssim_flag=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'eval_file',
                'weights_file',
                'scale',
                'kernel_sizes',
                'parameter_count',
                'eval_psnr',
                'eval_ssim',
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                'eval_file': args.eval_file,
                'weights_file': args.weights_file,
                'scale': config.get('scale'),
                'kernel_sizes': '-'.join(str(size) for size in config.get('kernel_sizes', [9, 5, 5])),
                'parameter_count': config.get('parameter_count'),
                'eval_psnr': eval_psnr,
                'eval_ssim': eval_ssim,
            }
        )
    print(f'eval psnr: {eval_psnr:.4f}, eval ssim: {eval_ssim:.4f}')
    print(f'Saved eval to {args.output_path}')


if __name__ == '__main__':
    main()
