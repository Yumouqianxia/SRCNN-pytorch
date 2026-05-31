# Experiment Tracker

Use this sheet to keep all experiments comparable and reproducible.

## Baseline Reproduction

- [ ] `x2` baseline (`srcnn_baseline`, `mse`, `Y`)
- [ ] `x3` baseline (`srcnn_baseline`, `mse`, `Y`)
- [ ] `x4` baseline (`srcnn_baseline`, `mse`, `Y`)
- [ ] Repeat runs with different seeds (at least 2-3 seeds)

## Attention Ablation

- [ ] `x3` baseline (reference)
- [ ] `x3 + SE`
- [ ] `x3 + CBAM`
- [ ] Record parameter count and speed difference

## Perceptual Loss Ablation

- [ ] `x3 + mse+perceptual` on Y-compatibility setup
- [ ] `x3 + mse+perceptual` on RGB mainline
- [ ] Try `lambda_perc` in `{0.01, 0.05, 0.1}`

## Joint Model

- [ ] `x3 + Attention + Perceptual` (RGB)
- [ ] Compare with all single-change variants

## Result Logging Format

For each run, log:

1. Command
2. Random seed
3. Best epoch
4. Best PSNR
5. Best SSIM
6. Train time
7. Inference speed
8. Visual notes (edge quality, texture fidelity, artifacts)
