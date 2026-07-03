import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import create_model
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb


def load_model(weights_file: Path, config_file: Path, device: torch.device):
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model = create_model(
        model_name=config.get("model_name", "srcnn_baseline"),
        num_channels=int(config.get("num_channels", 1)),
        attention_type=config.get("attention_type", "none"),
        attention_position=config.get("attention_position", "after_conv2"),
        kernel_sizes=tuple(config.get("kernel_sizes", [9, 5, 5])),
    ).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()
    return model


def run_image(model, image_path: Path, output_path: Path, scale: int, device: torch.device):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    upscaled = image.resize((image.width * scale, image.height * scale), Image.BICUBIC)

    arr = np.array(upscaled).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(arr)
    y = torch.from_numpy(ycbcr[..., 0] / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_y = model(y).clamp(0.0, 1.0)

    pred_y = pred_y.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.stack([pred_y, ycbcr[..., 1], ycbcr[..., 2]], axis=2)
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(output).save(output_path)
    return upscaled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--weights-955", type=Path, required=True)
    parser.add_argument("--config-955", type=Path, required=True)
    parser.add_argument("--weights-915", type=Path, required=True)
    parser.add_argument("--config-915", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_955 = load_model(args.weights_955, args.config_955, device)
    model_915 = load_model(args.weights_915, args.config_915, device)

    scale_tag = f"x{args.scale}"
    bicubic_dir = args.output_dir / f"bicubic_{scale_tag}"
    srcnn955_dir = args.output_dir / f"srcnn_9-5-5_{scale_tag}"
    srcnn915_dir = args.output_dir / f"srcnn_9-1-5_{scale_tag}"
    for directory in (bicubic_dir, srcnn955_dir, srcnn915_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(args.input_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        stem = image_path.stem
        bicubic = run_image(
            model_955,
            image_path,
            srcnn955_dir / f"{stem}_srcnn_955_{scale_tag}.png",
            args.scale,
            device,
        )
        bicubic.save(bicubic_dir / f"{stem}_bicubic_{scale_tag}.png")
        run_image(
            model_915,
            image_path,
            srcnn915_dir / f"{stem}_srcnn_915_{scale_tag}.png",
            args.scale,
            device,
        )

    print(f"Saved practical SRCNN outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
