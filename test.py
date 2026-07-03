import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import create_model
from utils import calc_psnr, calc_ssim, convert_rgb_to_ycbcr, convert_ycbcr_to_rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--model-name', type=str, default='srcnn_baseline', choices=['srcnn_baseline', 'srcnn_attention'])
    parser.add_argument('--kernel-sizes', type=int, nargs=3, default=[9, 5, 5], metavar=('K1', 'K2', 'K3'))
    parser.add_argument('--attention-type', type=str, default='none', choices=['none', 'se', 'cbam'])
    parser.add_argument('--attention-position', type=str, default='after_conv2', choices=['after_conv1', 'after_conv2'])
    parser.add_argument('--num-channels', type=int, default=1, choices=[1, 3])
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = create_model(
        model_name=args.model_name,
        num_channels=args.num_channels,
        attention_type=args.attention_type,
        attention_position=args.attention_position,
        kernel_sizes=tuple(args.kernel_sizes),
    ).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    bicubic = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = bicubic.resize((bicubic.width * args.scale, bicubic.height * args.scale), resample=pil_image.BICUBIC)

    stem, ext = os.path.splitext(args.image_file)
    bicubic_path = f'{stem}_bicubic_x{args.scale}{ext}'
    bicubic.save(bicubic_path)

    image = np.array(bicubic).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    if args.num_channels == 1:
        model_input = torch.from_numpy(ycbcr[..., 0] / 255.0).to(device).unsqueeze(0).unsqueeze(0)
    else:
        rgb = image / 255.0
        model_input = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).to(device).unsqueeze(0)

    with torch.no_grad():
        preds = model(model_input).clamp(0.0, 1.0)

    psnr = calc_psnr(model_input, preds).item()
    ssim = calc_ssim(model_input, preds)
    print('PSNR: {:.2f}, SSIM: {:.4f}'.format(psnr, ssim))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0)
    if args.num_channels == 1:
        output = np.array([preds[0], ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    else:
        output = np.transpose(preds, (1, 2, 0))
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)

    output = pil_image.fromarray(output)
    output.save(f'{stem}_srcnn_x{args.scale}{ext}')
