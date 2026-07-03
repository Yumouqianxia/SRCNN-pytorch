import argparse
from pathlib import Path

from PIL import Image


def prepare_split(input_dir: Path, output_root: Path, split_name: str, scales: list[int]) -> None:
    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    lr_dirs = {}
    hr_dirs = {}
    for scale in scales:
        hr_dir = output_root / split_name / "HR" / f"X{scale}"
        hr_dir.mkdir(parents=True, exist_ok=True)
        hr_dirs[scale] = hr_dir
        lr_dir = output_root / split_name / "LR_bicubic" / f"X{scale}"
        lr_dir.mkdir(parents=True, exist_ok=True)
        lr_dirs[scale] = lr_dir

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        stem = image_path.stem
        if stem.endswith("_GT"):
            stem = stem[:-3]

        for scale in scales:
            hr_width = (image.width // scale) * scale
            hr_height = (image.height // scale) * scale
            hr = image.resize((hr_width, hr_height), resample=Image.BICUBIC)
            lr = hr.resize((hr_width // scale, hr_height // scale), resample=Image.BICUBIC)
            hr.save(hr_dirs[scale] / f"{stem}.png")
            lr.save(lr_dirs[scale] / f"{stem}x{scale}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set5-dir", type=Path, required=True)
    parser.add_argument("--set14-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()

    prepare_split(args.set5_dir, args.output_root, "Set5", args.scales)
    prepare_split(args.set14_dir, args.output_root, "Set14", args.scales)


if __name__ == "__main__":
    main()
