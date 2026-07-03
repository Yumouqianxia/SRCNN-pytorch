import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from export_viz import load_model, preprocess_bicubic, run_model
from utils import convert_rgb_to_y


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets" / "final_comparison"


SRCNN_RUNS = {
    ("9-5-5", 2): ROOT / "outputs" / "repro_paper_full_x2_srcnn_baseline_none_mse_c1",
    ("9-5-5", 3): ROOT / "outputs" / "repro_paper_full_x3_srcnn_baseline_none_mse_c1",
    ("9-5-5", 4): ROOT / "outputs" / "repro_paper_full_x4_srcnn_baseline_none_mse_c1",
    ("9-1-5", 2): ROOT / "outputs" / "paper_915_fast_x2_srcnn_baseline_none_mse_c1_k9-1-5",
    ("9-1-5", 3): ROOT / "outputs" / "paper_915_fast_x3_srcnn_baseline_none_mse_c1_k9-1-5",
    ("9-1-5", 4): ROOT / "outputs" / "paper_915_fast_x4_srcnn_baseline_none_mse_c1_k9-1-5",
}


BENCHMARK_SAMPLES = [
    {
        "dataset": "Set5",
        "scale": 2,
        "image": "butterfly",
        "gt": ROOT / "data" / "swinir_testsets" / "Set5" / "HR" / "X2" / "butterfly.png",
        "swinir": ROOT / "outputs" / "swinir" / "set5_x2_outputs" / "butterfly_SwinIR.png",
    },
    {
        "dataset": "Set5",
        "scale": 3,
        "image": "butterfly",
        "gt": ROOT / "data" / "swinir_testsets" / "Set5" / "HR" / "X3" / "butterfly.png",
        "swinir": ROOT / "outputs" / "swinir" / "set5_x3_outputs" / "butterfly_SwinIR.png",
    },
    {
        "dataset": "Set5",
        "scale": 4,
        "image": "butterfly",
        "gt": ROOT / "data" / "swinir_testsets" / "Set5" / "HR" / "X4" / "butterfly.png",
        "swinir": ROOT / "outputs" / "swinir" / "set5_x4_outputs" / "butterfly_SwinIR.png",
    },
    {
        "dataset": "Set14",
        "scale": 3,
        "image": "zebra",
        "gt": ROOT / "data" / "swinir_testsets" / "Set14" / "HR" / "X3" / "zebra.png",
        "swinir": ROOT / "outputs" / "swinir" / "set14_x3_outputs" / "zebra_SwinIR.png",
    },
]


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(img.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def psnr_y(sr: Image.Image, gt: Image.Image) -> float:
    sr_y = convert_rgb_to_y(np.asarray(sr.convert("RGB")).astype(np.float32)) / 255.0
    gt_y = convert_rgb_to_y(np.asarray(gt.convert("RGB")).astype(np.float32)) / 255.0
    mse = np.mean((sr_y - gt_y) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim_y(sr: Image.Image, gt: Image.Image) -> float:
    sr_y = convert_rgb_to_y(np.asarray(sr.convert("RGB")).astype(np.float32)) / 255.0
    gt_y = convert_rgb_to_y(np.asarray(gt.convert("RGB")).astype(np.float32)) / 255.0
    return float(structural_similarity(sr_y, gt_y, data_range=1.0))


def load_srcnn(run_dir: Path, device: torch.device):
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model = load_model(str(run_dir / "best.pth"), config, device)
    return model, int(config.get("num_channels", 1))


def make_benchmark_grid(title: str, panels: list[tuple[str, Image.Image, str]], out_path: Path) -> None:
    cell_w = max(img.width for _, img, _ in panels)
    cell_h = max(img.height for _, img, _ in panels)
    cell_w = min(max(cell_w, 180), 260)
    cell_h = min(max(cell_h, 160), 230)
    pad = 16
    title_h = 42
    label_h = 52
    width = len(panels) * cell_w + (len(panels) + 1) * pad
    height = title_h + label_h + cell_h + pad * 2

    canvas = Image.new("RGB", (width, height), "#f4f7f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), title, fill="#14343d", font=font(20))

    x = pad
    for label, img, metric in panels:
        draw.rounded_rectangle(
            (x - 4, title_h + 2, x + cell_w + 4, height - pad + 4),
            radius=6,
            fill="white",
            outline="#cfd9dd",
        )
        draw.text((x, title_h + 8), label, fill="#087f99", font=font(14))
        if metric:
            draw.text((x, title_h + 28), metric, fill="#3b4c52", font=font(12))
        canvas.paste(fit(img, (cell_w, cell_h)), (x, title_h + label_h))
        x += cell_w + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_training_grid() -> None:
    train_dir = ROOT / "data" / "benchmark_raw" / "set14_srcnn_repo" / "Train"
    paths = sorted(train_dir.glob("*.bmp"))[:12]
    if not paths:
        return

    thumb = (140, 140)
    pad = 14
    cols = 6
    rows = int(np.ceil(len(paths) / cols))
    title_h = 42
    width = cols * thumb[0] + (cols + 1) * pad
    height = title_h + rows * (thumb[1] + 28) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "#f4f7f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), "91-image training-set samples (HR source images)", fill="#14343d", font=font(20))

    for idx, path in enumerate(paths):
        row, col = divmod(idx, cols)
        x = pad + col * (thumb[0] + pad)
        y = title_h + pad + row * (thumb[1] + 28 + pad)
        img = Image.open(path).convert("RGB")
        canvas.paste(fit(img, thumb), (x, y))
        draw.text((x, y + thumb[1] + 6), path.name, fill="#3b4c52", font=font(12))

    out_path = OUT / "images" / "training_samples" / "training_hr_samples_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_training_degradation_examples(scale: int = 3) -> None:
    train_dir = ROOT / "data" / "benchmark_raw" / "set14_srcnn_repo" / "Train"
    paths = sorted(train_dir.glob("*.bmp"))[:6]
    if not paths:
        return

    out_dir = OUT / "images" / "training_degradation_examples"
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    grids = []
    for path in paths:
        hr = Image.open(path).convert("RGB")
        width = (hr.width // scale) * scale
        height = (hr.height // scale) * scale
        hr = hr.resize((width, height), Image.Resampling.BICUBIC)
        lr = hr.resize((width // scale, height // scale), Image.Resampling.BICUBIC)
        bicubic_input = lr.resize((width, height), Image.Resampling.BICUBIC)

        sample_dir = panel_dir / path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        hr.save(sample_dir / "hr_ground_truth.png")
        lr.save(sample_dir / f"synthetic_lr_x{scale}.png")
        bicubic_input.save(sample_dir / f"bicubic_input_x{scale}.png")

        panels = [
            ("HR ground truth", hr, ""),
            (f"Synthetic LR x{scale}", lr.resize((width, height), Image.Resampling.NEAREST), "displayed with nearest zoom"),
            (f"Bicubic input x{scale}", bicubic_input, "SRCNN model input"),
        ]
        grid_path = out_dir / f"{path.stem}_degradation_x{scale}.png"
        make_benchmark_grid(f"Training degradation example: {path.name}", panels, grid_path)
        grids.append(Image.open(grid_path).convert("RGB"))

    if grids:
        width = max(grid.width for grid in grids)
        height = sum(grid.height for grid in grids) + 18 * (len(grids) - 1)
        sheet = Image.new("RGB", (width, height), "#e9eef1")
        y = 0
        for grid in grids:
            sheet.paste(grid, ((width - grid.width) // 2, y))
            y += grid.height + 18
        sheet.save(out_dir / f"all_training_degradation_examples_x{scale}.png")


def copy_tables() -> None:
    table_dir = OUT / "data"
    table_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / "report_assets" / "data" / "summary_swinir_vs_srcnn_y_nocrop.csv",
        ROOT / "report_assets" / "data" / "summary_repro_paper_915_fast.csv",
        ROOT / "report_assets" / "data" / "set14_x3_955_fullimage.csv",
        ROOT / "report_assets" / "data" / "set14_x3_915_fullimage.csv",
    ]
    for src in sources:
        if src.exists():
            shutil.copy2(src, table_dir / src.name)


def copy_practical_grids() -> None:
    out_dir = OUT / "images" / "practical_no_gt_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / "report_assets" / "images" / "all_practical_comparisons_x4.png",
        ROOT / "report_assets" / "images" / "anime_avatar_comparison_x4.png",
        ROOT / "report_assets" / "images" / "cartoon_character_comparison_x4.png",
        ROOT / "report_assets" / "images" / "building_lowres_comparison_x4.png",
        ROOT / "report_assets" / "images" / "portrait_clearer_comparison_x4.png",
    ]
    for src in sources:
        if src.exists():
            shutil.copy2(src, out_dir / src.name)


def build_benchmark_comparisons() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache = {}
    rows = []
    panel_root = OUT / "images" / "benchmark_gt_comparisons" / "panels"

    for sample in BENCHMARK_SAMPLES:
        dataset = sample["dataset"]
        scale = sample["scale"]
        image_name = sample["image"]
        sample_tag = f"{dataset.lower()}_{image_name}_x{scale}"

        gt = Image.open(sample["gt"]).convert("RGB")
        bicubic = preprocess_bicubic(gt, scale)
        swinir = Image.open(sample["swinir"]).convert("RGB")

        model_outputs = {}
        for model_name in ("9-5-5", "9-1-5"):
            cache_key = (model_name, scale)
            if cache_key not in cache:
                cache[cache_key] = load_srcnn(SRCNN_RUNS[cache_key], device)
            model, num_channels = cache[cache_key]
            sr = run_model(model, np.asarray(bicubic), num_channels, device)
            model_outputs[model_name] = Image.fromarray(sr)

        panels = [
            ("Ground truth", gt, ""),
            ("Bicubic", bicubic, f"{psnr_y(bicubic, gt):.2f} / {ssim_y(bicubic, gt):.4f}"),
            ("SRCNN 9-5-5", model_outputs["9-5-5"], f"{psnr_y(model_outputs['9-5-5'], gt):.2f} / {ssim_y(model_outputs['9-5-5'], gt):.4f}"),
            ("SRCNN 9-1-5", model_outputs["9-1-5"], f"{psnr_y(model_outputs['9-1-5'], gt):.2f} / {ssim_y(model_outputs['9-1-5'], gt):.4f}"),
            ("SwinIR", swinir, f"{psnr_y(swinir, gt):.2f} / {ssim_y(swinir, gt):.4f}"),
        ]

        sample_panel_dir = panel_root / sample_tag
        sample_panel_dir.mkdir(parents=True, exist_ok=True)
        gt.save(sample_panel_dir / "ground_truth.png")
        bicubic.save(sample_panel_dir / "bicubic.png")
        model_outputs["9-5-5"].save(sample_panel_dir / "srcnn_9-5-5.png")
        model_outputs["9-1-5"].save(sample_panel_dir / "srcnn_9-1-5.png")
        swinir.save(sample_panel_dir / "swinir.png")

        make_benchmark_grid(
            f"{dataset} {image_name} x{scale} (PSNR / SSIM on Y channel)",
            panels,
            OUT / "images" / "benchmark_gt_comparisons" / f"{sample_tag}_comparison.png",
        )

        for label, img, _ in panels[1:]:
            rows.append(
                {
                    "dataset": dataset,
                    "scale": f"x{scale}",
                    "image": image_name,
                    "method": label,
                    "psnr_y_nocrop": f"{psnr_y(img, gt):.4f}",
                    "ssim_y_nocrop": f"{ssim_y(img, gt):.4f}",
                }
            )

    csv_path = OUT / "data" / "selected_image_metrics_y_nocrop.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "scale", "image", "method", "psnr_y_nocrop", "ssim_y_nocrop"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_readme() -> None:
    text = """# Final Comparison Assets

All paths are relative to the repository root.

## Data

- `data/summary_swinir_vs_srcnn_y_nocrop.csv`: final dataset-level quantitative table.
- `data/selected_image_metrics_y_nocrop.csv`: per-image metrics for the visual examples.
- `data/summary_repro_paper_915_fast.csv`: Set5 paper-faithful `9-1-5` reproduction results.
- `data/set14_x3_955_fullimage.csv` and `data/set14_x3_915_fullimage.csv`: Set14 x3 full-image SRCNN evaluation.

## Images

- `images/training_samples/training_hr_samples_grid.png`: HR source samples from the 91-image training set.
- `images/training_degradation_examples/`: examples showing HR ground truth, synthetic LR, and bicubic-upsampled SRCNN input.
- `images/benchmark_gt_comparisons/`: benchmark examples with ground truth, bicubic, SRCNN `9-5-5`, SRCNN `9-1-5`, and SwinIR.
- `images/benchmark_gt_comparisons/panels/`: individual panels for each benchmark example.
- `images/practical_no_gt_comparisons/`: real low-resolution examples. These do not have ground truth, so PSNR/SSIM should not be reported for them.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    copy_tables()
    make_training_grid()
    make_training_degradation_examples(scale=3)
    build_benchmark_comparisons()
    copy_practical_grids()
    write_readme()
    print(f"Built report package: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
