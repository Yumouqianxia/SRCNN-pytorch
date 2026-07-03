import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import convert_rgb_to_y


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    return float(10.0 * np.log10(1.0 / mse))


def eval_pair(sr_path: Path, gt_path: Path) -> tuple[float, float]:
    sr = np.array(Image.open(sr_path).convert("RGB")).astype(np.float32)
    gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.float32)
    if sr.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {sr_path.name}: {sr.shape} vs {gt.shape}")

    sr_y = convert_rgb_to_y(sr) / 255.0
    gt_y = convert_rgb_to_y(gt) / 255.0
    return psnr(sr_y, gt_y), float(structural_similarity(sr_y, gt_y, data_range=1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--sr-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for gt_path in sorted(args.gt_dir.glob("*")):
        if gt_path.suffix.lower() not in {".png", ".bmp", ".jpg", ".jpeg"}:
            continue
        sr_path = args.sr_dir / f"{gt_path.stem}_SwinIR.png"
        if not sr_path.exists():
            raise FileNotFoundError(f"Missing SwinIR output for {gt_path.name}: {sr_path}")
        eval_psnr, eval_ssim = eval_pair(sr_path, gt_path)
        rows.append(
            {
                "dataset": args.dataset,
                "scale": args.scale,
                "image": gt_path.stem,
                "psnr_y_nocrop": eval_psnr,
                "ssim_y_nocrop": eval_ssim,
            }
        )

    avg_psnr = float(np.mean([row["psnr_y_nocrop"] for row in rows]))
    avg_ssim = float(np.mean([row["ssim_y_nocrop"] for row in rows]))
    rows.append(
        {
            "dataset": args.dataset,
            "scale": args.scale,
            "image": "AVERAGE",
            "psnr_y_nocrop": avg_psnr,
            "ssim_y_nocrop": avg_ssim,
        }
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "scale", "image", "psnr_y_nocrop", "ssim_y_nocrop"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{args.dataset} x{args.scale}: PSNR_Y={avg_psnr:.4f}, SSIM_Y={avg_ssim:.4f}")
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
