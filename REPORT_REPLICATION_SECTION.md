## 4. Paper Replication: SRCNN (ECCV 2014)

### 4.1 复现目标与设置

本项目选择复现 Dong 等人在 ECCV 2014 提出的 SRCNN（Super-Resolution Convolutional Neural Network）方法。复现目标为：在与原文一致的任务设定下（单幅图像超分辨率，放大倍率 `x2`、`x3`、`x4`），验证模型在标准测试集上的重建性能，并与论文报告结果进行对比。

本次正式复现采用如下设置：

- **模型结构**：SRCNN 基线网络（3 层卷积，9-5-5，通道数 64-32-1）
- **训练目标**：MSE 损失（仅 Y 通道）
- **训练数据**：`91-image_x2/x3/x4.h5`
- **测试数据**：`Set5_x2/x3/x4.h5`
- **倍率策略**：每个放大倍率单独训练一个模型
- **训练轮次**：400 epochs
- **实现框架**：PyTorch（GPU 训练）

### 4.2 定量结果

正式复现实验结果如下（Set5）：

| Scale | Reproduced PSNR (dB) | Reproduced SSIM | Paper/Reference PSNR (dB) | Difference |
|---|---:|---:|---:|---:|
| x2 | 36.51 | 0.9588 | 36.66 | -0.15 |
| x3 | 33.19 | 0.9303 | 32.75 | +0.44 |
| x4 | 30.17 | 0.8680 | 30.49 | -0.32 |

从结果可见，复现值与参考值整体接近，误差在合理范围内；其中 `x3` 设置下结果略高于参考值，说明实现与训练流程能够稳定复现 SRCNN 的核心性能。

### 4.3 可视化结果

除定量指标外，我们导出了重建可视化结果（Original / Bicubic / SRCNN）进行主观对比。结果显示 SRCNN 相较双三次插值在边缘清晰度与纹理恢复上具有明显优势，视觉效果与论文描述一致。可视化文件位于：

- `outputs/repro_paper_full_x2_srcnn_baseline_none_mse_c1/viz`
- `outputs/repro_paper_full_x3_srcnn_baseline_none_mse_c1/viz`
- `outputs/repro_paper_full_x4_srcnn_baseline_none_mse_c1/viz`

其中 `*_triplet_x*.png` 为三图拼接对比图。

### 4.4 复现结论与偏差分析

本工作成功复现了 SRCNN 在 Set5 上的主要性能趋势，结果与论文报告值高度一致，说明代码实现、数据处理流程与训练策略整体有效。与参考结果仍存在小幅偏差，可能来源于：

1. 训练实现细节差异（优化器、学习率调度、评估频率等）；
2. 随机初始化与训练随机性；
3. 数据预处理与边界处理的具体实现差异。

总体上，本次复现满足“可重复、可验证、可对比”的课程要求，并为后续结构改进（注意力机制）与损失函数扩展（感知损失）提供了可靠基线。
