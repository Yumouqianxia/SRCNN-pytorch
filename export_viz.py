import argparse
import json
import os
from pathlib import Path

import numpy as np
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn

from models import create_model
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True, help='Experiment directory containing best.pth and config.json')
    parser.add_argument('--images', nargs='+', required=True, help='Input image paths for visualization')
    parser.add_argument('--scale', type=int, default=None, help='Override scale in config')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory. Default: <run-dir>/viz')
    return parser.parse_args()


def load_run_config(run_dir):
    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config.json not found under {run_dir}')
    with open(config_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def load_model(weights_file, config, device):
    model = create_model(
        model_name=config.get('model_name', 'srcnn_baseline'),
        num_channels=int(config.get('num_channels', 1)),
        attention_type=config.get('attention_type', 'none'),
        attention_position=config.get('attention_position', 'after_conv2'),
        kernel_sizes=tuple(config.get('kernel_sizes', [9, 5, 5])),
    ).to(device)

    state_dict = model.state_dict()
    loaded = torch.load(weights_file, map_location=lambda storage, loc: storage)
    for name, param in loaded.items():
        if name in state_dict:
            state_dict[name].copy_(param)
        else:
            raise KeyError(name)
    model.eval()
    return model


def preprocess_bicubic(image, scale):
    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    bicubic = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    bicubic = bicubic.resize((bicubic.width * scale, bicubic.height * scale), resample=pil_image.BICUBIC)
    return bicubic


def run_model(model, image_rgb_np, num_channels, device):
    if num_channels == 1:
        ycbcr = convert_rgb_to_ycbcr(image_rgb_np.astype(np.float32))
        model_input = torch.from_numpy(ycbcr[..., 0] / 255.0).to(device).unsqueeze(0).unsqueeze(0)
    else:
        rgb = image_rgb_np.astype(np.float32) / 255.0
        model_input = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).to(device).unsqueeze(0)
        ycbcr = None

    with torch.no_grad():
        preds = model(model_input).clamp(0.0, 1.0)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0)

    if num_channels == 1:
        output = np.array([preds[0], ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    else:
        output = np.transpose(preds, (1, 2, 0))
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    return output


def build_triplet_canvas(original, bicubic, sr):
    width, height = original.size
    canvas = pil_image.new('RGB', (width * 3, height))
    canvas.paste(original, (0, 0))
    canvas.paste(bicubic, (width, 0))
    canvas.paste(sr, (width * 2, 0))
    return canvas


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    weights_path = os.path.join(run_dir, 'best.pth')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'best.pth not found under {run_dir}')

    config = load_run_config(run_dir)
    scale = args.scale if args.scale is not None else int(config.get('scale', 3))
    num_channels = int(config.get('num_channels', 1))
    output_dir = args.output_dir if args.output_dir else os.path.join(run_dir, 'viz')
    os.makedirs(output_dir, exist_ok=True)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(weights_path, config, device)

    for image_path in args.images:
        image_path = os.path.abspath(image_path)
        image_name = Path(image_path).stem
        image_ext = Path(image_path).suffix
        original = pil_image.open(image_path).convert('RGB')
        bicubic = preprocess_bicubic(original, scale)
        sr_np = run_model(model, np.array(bicubic), num_channels, device)
        sr_image = pil_image.fromarray(sr_np)

        bicubic_path = os.path.join(output_dir, f'{image_name}_bicubic_x{scale}{image_ext}')
        sr_path = os.path.join(output_dir, f'{image_name}_sr_x{scale}{image_ext}')
        triplet_path = os.path.join(output_dir, f'{image_name}_triplet_x{scale}.png')

        bicubic.save(bicubic_path)
        sr_image.save(sr_path)
        build_triplet_canvas(original.resize(sr_image.size), bicubic, sr_image).save(triplet_path)
        print(f'exported: {sr_path}')

    print(f'Visualization outputs saved to: {output_dir}')


if __name__ == '__main__':
    main()
