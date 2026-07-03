import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--swinir-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--training-patch-size", type=int, default=48)
    args = parser.parse_args()

    sys.path.insert(0, str(args.swinir_dir.resolve()))
    from models.network_swinir import SwinIR

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SwinIR(
        upscale=args.scale,
        in_chans=3,
        img_size=args.training_patch_size,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    pretrained = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(pretrained["params"] if "params" in pretrained else pretrained, strict=True)
    model.eval().to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_size = 8
    for image_path in sorted(args.input_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, h_old, w_old = tensor.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([tensor, torch.flip(tensor, [2])], 2)[:, :, : h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, : w_old + w_pad]
            output = model(inp)
            output = output[..., : h_old * args.scale, : w_old * args.scale]

        output = output.squeeze(0).clamp(0.0, 1.0).cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        Image.fromarray(output).save(args.output_dir / f"{image_path.stem}_swinir_x{args.scale}.png")

    print(f"Saved practical SwinIR outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
