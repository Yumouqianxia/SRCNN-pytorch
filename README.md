# SRCNN Reproduction + Extension Toolkit

This repo now supports a full project workflow for:

1. Baseline SRCNN reproduction (`x2/x3/x4`, Y channel, MSE)
2. Structural extension (attention modules: `SE` or `CBAM`)
3. Objective extension (perceptual loss on top of MSE)
4. Joint experiment template (`Attention + Perceptual`)

## 1) Environment Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2) Dataset Preparation

You can use official H5 files or generate your own.

### Generate training H5

```bash
python prepare.py --images-dir "<train_images_dir>" --output-path "<train_x3_y.h5>" --scale 3 --patch-size 33 --stride 14 --color-space y
```

### Generate eval H5

```bash
python prepare.py --images-dir "<eval_images_dir>" --output-path "<eval_x3_y.h5>" --scale 3 --color-space y --eval
```

For RGB experiments, set `--color-space rgb`.

## 3) Baseline Reproduction

```bash
python train.py \
  --train-file "<train_x3_y.h5>" \
  --eval-file "<eval_x3_y.h5>" \
  --outputs-dir "outputs" \
  --experiment-name "baseline_repro" \
  --scale 3 \
  --model-name srcnn_baseline \
  --attention-type none \
  --loss-type mse \
  --num-channels 1 \
  --num-epochs 400
```

Run similarly for `--scale 2` and `--scale 4`.

## 4) Attention Extension

### SE attention

```bash
python train.py \
  --train-file "<train_x3_y.h5>" \
  --eval-file "<eval_x3_y.h5>" \
  --outputs-dir "outputs" \
  --experiment-name "attention_se" \
  --scale 3 \
  --model-name srcnn_attention \
  --attention-type se \
  --attention-position after_conv2 \
  --loss-type mse \
  --num-channels 1
```

### CBAM attention

Replace `--attention-type se` with `--attention-type cbam`.

## 5) Perceptual Loss Extension

### Y-channel compatibility test

```bash
python train.py \
  --train-file "<train_x3_y.h5>" \
  --eval-file "<eval_x3_y.h5>" \
  --outputs-dir "outputs" \
  --experiment-name "perceptual_y_compat" \
  --scale 3 \
  --model-name srcnn_baseline \
  --loss-type mse+perceptual \
  --num-channels 1 \
  --perceptual-weight 0.01 \
  --perceptual-layer relu3_3
```

### RGB mainline (recommended)

```bash
python train.py \
  --train-file "<train_x3_rgb.h5>" \
  --eval-file "<eval_x3_rgb.h5>" \
  --outputs-dir "outputs" \
  --experiment-name "perceptual_rgb" \
  --scale 3 \
  --model-name srcnn_baseline \
  --loss-type mse+perceptual \
  --num-channels 3 \
  --perceptual-weight 0.01 \
  --perceptual-layer relu3_3
```

## 6) Joint Model (Attention + Perceptual)

```bash
python train.py \
  --train-file "<train_x3_rgb.h5>" \
  --eval-file "<eval_x3_rgb.h5>" \
  --outputs-dir "outputs" \
  --experiment-name "joint_rgb" \
  --scale 3 \
  --model-name srcnn_attention \
  --attention-type se \
  --loss-type mse+perceptual \
  --num-channels 3 \
  --perceptual-weight 0.01
```

## 7) Batch Experiment Launcher

`run_experiments.py` can launch baseline + attention + perceptual(Y compatibility) jobs:

```bash
python run_experiments.py \
  --outputs-dir outputs \
  --train-files 2=<train_x2_y.h5> 3=<train_x3_y.h5> 4=<train_x4_y.h5> \
  --eval-files 2=<eval_x2_y.h5> 3=<eval_x3_y.h5> 4=<eval_x4_y.h5> \
  --num-epochs 200
```

## 8) Inference

```bash
python test.py \
  --weights-file "<outputs/.../best.pth>" \
  --image-file "<test_image_path>" \
  --scale 3 \
  --model-name srcnn_attention \
  --attention-type se \
  --num-channels 1
```

The script saves bicubic and SR outputs next to the input image.

## 8.1) Export Visual Comparison After Training

Use `export_viz.py` to export images into a run-specific visualization folder:

```bash
python export_viz.py \
  --run-dir "outputs/baseline_repro_x3_srcnn_baseline_none_mse_c1" \
  --images "data/butterfly_GT.bmp" "data/zebra.bmp" "data/ppt3.bmp"
```

It will create `<run-dir>/viz/` containing:

- `*_bicubic_x<scale>.*`
- `*_sr_x<scale>.*`
- `*_triplet_x<scale>.png` (original / bicubic / SR side by side)

## 9) Logged Artifacts

Each experiment directory contains:

- `config.json`: full configuration and parameter count
- `metrics.csv`: per-epoch train/eval metrics (loss, PSNR, SSIM)
- `best.pth`: best checkpoint by PSNR
- optional `epoch_*.pth`: intermediate checkpoints
