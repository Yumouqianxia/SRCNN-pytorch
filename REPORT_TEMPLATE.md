# Final Report Template (8-12 pages)

## 1. Abstract and Objectives

- Problem: single-image super-resolution
- Paper selected: SRCNN (ECCV 2014)
- Objectives:
  - Reproduce baseline performance
  - Extend architecture with attention
  - Extend objective with perceptual loss
  - Analyze trade-offs (quality/speed/complexity)

## 2. Background and Related Work

- Classical SR and sparse coding
- SRCNN motivation and architecture
- Follow-up SR methods (VDSR, EDSR, SRGAN/ESRGAN, SwinIR)
- Why attention and perceptual losses are reasonable extensions

## 3. Methodology

### 3.1 Baseline Reproduction

- Dataset setup
- Preprocessing pipeline
- Training settings (optimizer/lr/epochs/seed)
- Evaluation protocol (PSNR/SSIM, visual comparisons)

### 3.2 Structural Extension

- Attention design (SE/CBAM placement and rationale)
- Complexity analysis (parameter count and speed)

### 3.3 Loss Extension

- MSE + perceptual objective
- VGG layer choice and lambda tuning
- Y-compatibility trial vs RGB mainline

## 4. Results and Analysis

- Baseline reproducibility table (x2/x3/x4)
- Attention ablation table
- Perceptual ablation table
- Joint model comparison table
- Visual comparisons (bicubic, baseline, extensions)
- Failure cases and possible reasons

## 5. Discussion

- What worked
- What did not work
- Why some metrics improved while others did not
- Deployment considerations (speed, memory, model size)

## 6. Conclusion and Future Work

- Summary of key findings
- Lessons learned from replication
- Next potential directions (deeper backbone, GAN-based loss, transformer SR)

## 7. Individual Contributions (for team report)

Add a table with:

- Member name
- Primary responsibility
- Specific deliverables
