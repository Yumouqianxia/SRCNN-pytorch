# SRCNN Reproduction and Modern SR Comparison

This repository contains a course final project for Digital Image and Signal Processing.
The project started as a reproduction of SRCNN and was extended with:

- paper-faithful SRCNN `9-1-5` reproduction,
- comparison with the initial `9-5-5` SRCNN variant,
- attention and perceptual-loss ablations,
- pretrained SwinIR comparison,
- qualitative practical tests on real low-resolution images.

The current repository keeps source code, lightweight scripts, summary CSV files, and selected report figures.
Large generated artifacts such as H5 datasets, trained `.pth` checkpoints, full `outputs/`, and SwinIR pretrained weights are intentionally not tracked.
All paths below are relative to the repository root after cloning.

## Repository Structure

```text
.
|-- datasets.py
|-- eval_h5.py
|-- export_viz.py
|-- losses.py
|-- models.py
|-- prepare.py
|-- run_experiments.py
|-- summarize_results.py
|-- test.py
|-- train.py
|-- utils.py
|-- scripts/
|   |-- prepare_swinir_testsets.py
|   |-- eval_swinir_outputs.py
|   |-- run_practical_srcnn.py
|   |-- run_practical_swinir.py
|   `-- make_practical_comparison_grid.py
`-- report_assets/
    |-- data/
    `-- images/
```

## Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`requirements.txt` includes the extra dependencies needed for SwinIR inference and report scripts:
`opencv-python`, `timm`, `requests`, and `pypdf`.

## Data Preparation

The SRCNN pipeline uses HR images to synthesize LR/HR pairs through bicubic downsampling and bicubic upsampling.

Training H5:

```powershell
python prepare.py `
  --images-dir "<train_images_dir>" `
  --output-path "data/h5_paper/91-image_x3.h5" `
  --scale 3 `
  --patch-size 33 `
  --stride 14 `
  --color-space y
```

Evaluation H5:

```powershell
python prepare.py `
  --images-dir "<eval_images_dir>" `
  --output-path "data/h5_paper/Set5_x3.h5" `
  --scale 3 `
  --color-space y `
  --eval
```

## SRCNN Reproduction

### 9-5-5 Variant

The initial implementation uses `9-5-5` kernels:

```powershell
python train.py `
  --train-file "data/h5_paper/91-image_x3.h5" `
  --eval-file "data/h5_paper/Set5_x3.h5" `
  --outputs-dir "outputs" `
  --experiment-name "repro_paper_full" `
  --scale 3 `
  --model-name srcnn_baseline `
  --attention-type none `
  --loss-type mse `
  --num-channels 1 `
  --kernel-sizes 9 5 5 `
  --num-epochs 400
```

### Paper-Faithful 9-1-5

The original ECCV 2014 SRCNN architecture uses `9-1-5` kernels.
This project adds direct support for arbitrary kernel sizes:

```powershell
python train.py `
  --train-file "data/h5_paper/91-image_x3.h5" `
  --eval-file "data/h5_paper/Set5_x3.h5" `
  --outputs-dir "outputs" `
  --experiment-name "paper_915_fast" `
  --scale 3 `
  --model-name srcnn_baseline `
  --attention-type none `
  --loss-type mse `
  --num-channels 1 `
  --kernel-sizes 9 1 5 `
  --num-epochs 400
```

The same command pattern was used for scales `x2`, `x3`, and `x4`.

## Set14 Evaluation

Evaluate a trained checkpoint on another H5 file:

```powershell
python eval_h5.py `
  --eval-file "data/h5_paper/Set14_x3_eval.h5" `
  --weights-file "outputs/repro_paper_full_x3_srcnn_baseline_none_mse_c1/best.pth" `
  --config-file "outputs/repro_paper_full_x3_srcnn_baseline_none_mse_c1/config.json" `
  --output-path "outputs/repro_paper_full_set14_x3_eval_fullimage.csv"
```

## SwinIR Modern Comparison

SwinIR is used as a modern transformer-based comparison method from the course-provided super-resolution paper list.
It is **not trained from scratch** in this project. We use official pretrained SwinIR weights for inference.

Clone SwinIR separately:

```powershell
mkdir external
git clone https://github.com/JingyunLiang/SwinIR.git external/SwinIR
```

Download or auto-download official classical SR weights:

```text
external/SwinIR/model_zoo/swinir/
|-- 001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth
|-- 001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth
`-- 001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth
```

Prepare image-folder testsets for SwinIR:

```powershell
python scripts/prepare_swinir_testsets.py `
  --set5-dir "data/benchmark_raw/set14_srcnn_repo/Set5" `
  --set14-dir "data/benchmark_raw/Set14" `
  --output-root "data/swinir_testsets" `
  --scales 2 3 4
```

Run official SwinIR inference, for example Set5 x3:

```powershell
cd external/SwinIR
..\..\.venv\Scripts\python.exe main_test_swinir.py `
  --task classical_sr `
  --scale 3 `
  --training_patch_size 48 `
  --model_path "model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth" `
  --folder_lq "../../data/swinir_testsets/Set5/LR_bicubic/X3" `
  --folder_gt "../../data/swinir_testsets/Set5/HR/X3"
cd ../..
```

Recompute SwinIR outputs with the same Y-channel/no-crop convention used for the SRCNN table:

```powershell
python scripts/eval_swinir_outputs.py `
  --dataset Set5 `
  --scale 3 `
  --sr-dir "outputs/swinir/set5_x3_outputs" `
  --gt-dir "data/swinir_testsets/Set5/HR/X3" `
  --output-csv "outputs/swinir/set5_x3_y_nocrop.csv"
```

## Practical Low-Resolution Images

Real low-resolution images do not have ground-truth HR references, so PSNR/SSIM is not meaningful.
They are used only for qualitative comparison.

Run SRCNN on practical images:

```powershell
python scripts/run_practical_srcnn.py `
  --input-dir "data/practical_images_x4" `
  --output-dir "outputs/practical_sr_x4_all" `
  --scale 4 `
  --weights-955 "outputs/repro_paper_full_x4_srcnn_baseline_none_mse_c1/best.pth" `
  --config-955 "outputs/repro_paper_full_x4_srcnn_baseline_none_mse_c1/config.json" `
  --weights-915 "outputs/paper_915_fast_x4_srcnn_baseline_none_mse_c1_k9-1-5/best.pth" `
  --config-915 "outputs/paper_915_fast_x4_srcnn_baseline_none_mse_c1_k9-1-5/config.json"
```

Run SwinIR on the same practical images:

```powershell
python scripts/run_practical_swinir.py `
  --input-dir "data/practical_images_x4" `
  --output-dir "outputs/practical_sr_x4_all/swinir_x4" `
  --swinir-dir "external/SwinIR" `
  --model-path "external/SwinIR/model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth" `
  --scale 4 `
  --training-patch-size 48
```

Create comparison grids:

```powershell
python scripts/make_practical_comparison_grid.py `
  --input-dir "data/practical_images_x4" `
  --output-root "outputs/practical_sr_x4_all" `
  --scale 4
```

## Final Summary Results

Selected final report assets are tracked under `report_assets/`.

For report writing, the most complete packaged version is:

```text
report_assets/final_comparison/
```

It contains the final CSV tables, training-set sample images, benchmark visual comparisons with ground truth, and practical no-ground-truth examples.

### SRCNN and SwinIR Quantitative Comparison

File: `report_assets/data/summary_swinir_vs_srcnn_y_nocrop.csv`

| Dataset | Scale | SRCNN 9-5-5 | SRCNN 9-1-5 | SwinIR pretrained |
|---|---:|---:|---:|---:|
| Set5 | x2 | 36.5129 / 0.9588 | 36.0641 / 0.9560 | 38.3341 / 0.9666 |
| Set5 | x3 | 33.1732 / 0.9303 | 32.7022 / 0.9226 | 35.4955 / 0.9508 |
| Set5 | x4 | 30.1749 / 0.8680 | 29.9150 / 0.8595 | 32.6230 / 0.9111 |
| Set14 | x3 | 29.5050 / 0.8522 | 29.2290 / 0.8457 | 31.0948 / 0.8797 |

### Practical Qualitative Comparison

The x4 practical comparison figure is available at:

```text
report_assets/images/all_practical_comparisons_x4.png
```

These examples compare:

```text
LR input -> Bicubic -> SRCNN 9-5-5 -> SRCNN 9-1-5 -> SwinIR
```

No PSNR/SSIM is reported for these real low-resolution examples because ground-truth HR images are unavailable.

## What Is Not Tracked

The following files are intentionally excluded from git:

- `.venv/`
- `external/SwinIR/`
- official SwinIR `.pth` weights,
- generated H5 datasets,
- full training outputs and logs,
- full PowerPoint/PDF drafts,
- full practical image output folders.

This keeps the GitHub repository small while preserving the code and final report evidence needed to reproduce the process.
