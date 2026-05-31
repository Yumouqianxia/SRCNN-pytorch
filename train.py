import argparse
import copy
import csv
import json
import os
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets import EvalDataset, TrainDataset
from losses import CombinedLoss
from models import create_model
from utils import AverageMeter, calc_psnr, calc_ssim, count_parameters


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, default='baseline')
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--model-name', type=str, default='srcnn_baseline', choices=['srcnn_baseline', 'srcnn_attention'])
    parser.add_argument('--attention-type', type=str, default='none', choices=['none', 'se', 'cbam'])
    parser.add_argument('--attention-position', type=str, default='after_conv2', choices=['after_conv1', 'after_conv2'])
    parser.add_argument('--num-channels', type=int, default=1, choices=[1, 3])
    parser.add_argument('--loss-type', type=str, default='mse', choices=['mse', 'mse+perceptual'])
    parser.add_argument('--perceptual-weight', type=float, default=0.01)
    parser.add_argument('--perceptual-layer', type=str, default='relu3_3', choices=['relu2_2', 'relu3_3', 'relu4_3'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--ssim-interval', type=int, default=1)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training on CUDA')
    return parser.parse_args()


def get_optimizer(args, model):
    return optim.Adam(
        [
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), 'lr': args.lr * 0.1},
        ],
        lr=args.lr,
    )


def run_train_epoch(model, dataloader, criterion, optimizer, scaler, device, batch_size, epoch, total_epochs, use_amp):
    model.train()
    epoch_losses = AverageMeter()
    epoch_mse = AverageMeter()
    epoch_perceptual = AverageMeter()
    expected_total = len(dataloader.dataset) - len(dataloader.dataset) % batch_size

    with tqdm(total=expected_total) as progress:
        progress.set_description(f'epoch: {epoch}/{total_epochs - 1}')
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                preds = model(inputs)
                loss, details = criterion(preds, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.update(loss.item(), len(inputs))
            epoch_mse.update(details["mse"], len(inputs))
            epoch_perceptual.update(details.get("perceptual", 0.0), len(inputs))
            progress.set_postfix(loss=f'{epoch_losses.avg:.6f}')
            progress.update(len(inputs))

    return epoch_losses.avg, epoch_mse.avg, epoch_perceptual.avg


def run_eval(model, dataloader, device, calc_ssim_flag=True):
    model.eval()
    epoch_psnr = AverageMeter()
    ssim_scores = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, labels).item(), len(inputs))
            if calc_ssim_flag:
                ssim_scores.append(calc_ssim(preds, labels))
    epoch_ssim = float(sum(ssim_scores) / len(ssim_scores)) if ssim_scores else None
    return epoch_psnr.avg, epoch_ssim


def main():
    args = get_args()
    run_name = (
        f'{args.experiment_name}_x{args.scale}_{args.model_name}_{args.attention_type}_'
        f'{args.loss_type}_c{args.num_channels}'
    )
    args.outputs_dir = os.path.join(args.outputs_dir, run_name)
    os.makedirs(args.outputs_dir, exist_ok=True)

    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.amp and device.type == 'cuda')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = create_model(
        model_name=args.model_name,
        num_channels=args.num_channels,
        attention_type=args.attention_type,
        attention_position=args.attention_position,
    ).to(device)
    criterion = CombinedLoss(
        loss_type=args.loss_type,
        perceptual_weight=args.perceptual_weight,
        perceptual_layer=args.perceptual_layer,
        device=device,
    ).to(device)
    optimizer = get_optimizer(args, model)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    train_dataset = TrainDataset(args.train_file)
    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    if args.num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = args.prefetch_factor
        train_loader_kwargs['persistent_workers'] = args.persistent_workers
    train_dataloader = DataLoader(**train_loader_kwargs)
    eval_dataset = EvalDataset(args.eval_file)
    eval_loader_kwargs = dict(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        pin_memory=(device.type == 'cuda'),
    )
    if eval_loader_kwargs['num_workers'] > 0:
        eval_loader_kwargs['prefetch_factor'] = args.prefetch_factor
        eval_loader_kwargs['persistent_workers'] = args.persistent_workers
    eval_dataloader = DataLoader(**eval_loader_kwargs)

    config = vars(args).copy()
    config["parameter_count"] = count_parameters(model)
    config["started_at"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(args.outputs_dir, 'config.json'), 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    metrics_path = os.path.join(args.outputs_dir, 'metrics.csv')
    with open(metrics_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['epoch', 'train_loss', 'train_mse', 'train_perceptual', 'eval_psnr', 'eval_ssim'],
        )
        writer.writeheader()

        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_psnr = 0.0

        for epoch in range(args.num_epochs):
            train_loss, train_mse, train_perceptual = run_train_epoch(
                model=model,
                dataloader=train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                batch_size=args.batch_size,
                epoch=epoch,
                total_epochs=args.num_epochs,
                use_amp=use_amp,
            )
            should_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch == args.num_epochs - 1)
            eval_psnr = None
            eval_ssim = None
            if should_eval:
                calc_ssim_flag = ((epoch + 1) % args.ssim_interval == 0) or (epoch == args.num_epochs - 1)
                eval_psnr, eval_ssim = run_eval(
                    model=model,
                    dataloader=eval_dataloader,
                    device=device,
                    calc_ssim_flag=calc_ssim_flag,
                )
            writer.writerow(
                {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_mse': train_mse,
                    'train_perceptual': train_perceptual,
                    'eval_psnr': '' if eval_psnr is None else eval_psnr,
                    'eval_ssim': '' if eval_ssim is None else eval_ssim,
                }
            )
            csv_file.flush()

            if should_eval:
                if eval_ssim is None:
                    print(f'eval psnr: {eval_psnr:.2f}, eval ssim: skipped')
                else:
                    print(f'eval psnr: {eval_psnr:.2f}, eval ssim: {eval_ssim:.4f}')
            else:
                print('eval: skipped')
            if (epoch + 1) % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.outputs_dir, f'epoch_{epoch}.pth'))

            if should_eval and eval_psnr > best_psnr:
                best_epoch = epoch
                best_psnr = eval_psnr
                best_weights = copy.deepcopy(model.state_dict())

    print(f'best epoch: {best_epoch}, psnr: {best_psnr:.2f}')
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))


if __name__ == '__main__':
    main()
