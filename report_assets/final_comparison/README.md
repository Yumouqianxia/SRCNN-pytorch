# Final Comparison Assets

All paths are relative to the repository root.

## Data

- `data/summary_swinir_vs_srcnn_y_nocrop.csv`: final dataset-level quantitative table.
- `data/selected_image_metrics_y_nocrop.csv`: per-image metrics for the visual examples.
- `data/summary_repro_paper_915_fast.csv`: Set5 paper-faithful `9-1-5` reproduction results.
- `data/set14_x3_955_fullimage.csv` and `data/set14_x3_915_fullimage.csv`: Set14 x3 full-image SRCNN evaluation.

## Images

- `images/training_samples/training_hr_samples_grid.png`: HR source samples from the 91-image training set.
- `images/training_degradation_examples/`: all 91 training examples showing HR ground truth, synthetic LR, and bicubic-upsampled SRCNN input.
- `images/benchmark_gt_comparisons/`: benchmark examples with ground truth, bicubic, SRCNN `9-5-5`, SRCNN `9-1-5`, and SwinIR.
- `images/benchmark_gt_comparisons/panels/`: individual panels for each benchmark example.
- `images/practical_no_gt_comparisons/`: real low-resolution examples. These do not have ground truth, so PSNR/SSIM should not be reported for them.
