import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_font(size: int):
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit_canvas(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(img.convert("RGB"), size, Image.Resampling.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def make_grid(stem: str, paths: list[tuple[str, Path]], out_path: Path, scale: int) -> Image.Image:
    images = []
    for label, path in paths:
        img = Image.open(path).convert("RGB")
        if label.startswith("LR"):
            img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        images.append((label, img))

    target_w = max(img.width for _, img in images)
    target_h = max(img.height for _, img in images)
    label_h = 44
    pad = 16
    font = load_font(18)
    small = load_font(13)
    title_h = 38

    width = len(images) * target_w + (len(images) + 1) * pad
    height = title_h + label_h + target_h + pad * 2
    grid = Image.new("RGB", (width, height), "#f5f7f8")
    draw = ImageDraw.Draw(grid)
    draw.text((pad, 8), stem, fill="#15333d", font=font)

    x = pad
    y_img = title_h + label_h
    for label, img in images:
        draw.rounded_rectangle(
            (x - 4, title_h + 4, x + target_w + 4, height - pad + 4),
            radius=8,
            fill="white",
            outline="#d7e0e4",
        )
        draw.text((x, title_h + 14), label, fill="#087f99", font=small)
        canvas = fit_canvas(img, (target_w, target_h))
        grid.paste(canvas, (x, y_img))
        x += target_w + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    grid_dir = args.output_root / "comparisons"
    all_grids = []
    for input_path in sorted(args.input_dir.iterdir()):
        if input_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        stem = input_path.stem
        scale_tag = f"x{args.scale}"
        paths = [
            (f"LR input (displayed x{args.scale})", input_path),
            (f"Bicubic x{args.scale}", args.output_root / f"bicubic_{scale_tag}" / f"{stem}_bicubic_{scale_tag}.png"),
            ("SRCNN 9-5-5", args.output_root / f"srcnn_9-5-5_{scale_tag}" / f"{stem}_srcnn_955_{scale_tag}.png"),
            ("SRCNN 9-1-5", args.output_root / f"srcnn_9-1-5_{scale_tag}" / f"{stem}_srcnn_915_{scale_tag}.png"),
            ("SwinIR", args.output_root / f"swinir_{scale_tag}" / f"{stem}_swinir_{scale_tag}.png"),
        ]
        missing = [str(path) for _, path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing outputs for {stem}: {missing}")
        grid = make_grid(stem, paths, grid_dir / f"{stem}_comparison_{scale_tag}.png", args.scale)
        all_grids.append(grid)

    if all_grids:
        width = max(grid.width for grid in all_grids)
        height = sum(grid.height for grid in all_grids) + 20 * (len(all_grids) - 1)
        sheet = Image.new("RGB", (width, height), "#e9eef1")
        y = 0
        for grid in all_grids:
            x = (width - grid.width) // 2
            sheet.paste(grid, (x, y))
            y += grid.height + 20
        sheet.save(grid_dir / f"all_practical_comparisons_{scale_tag}.png")
        print(grid_dir / f"all_practical_comparisons_{scale_tag}.png")


if __name__ == "__main__":
    main()
