# Marine Oil-Spill Segmentation in Sentinel-1 SAR Imagery

Deep-learning pipeline that detects and segments marine oil spills in Sentinel-1 SAR satellite imagery — B.Sc. graduation project (graded **A+**) at the British University in Egypt.

## Highlights

- **Five custom U-Net variants** designed and trained in PyTorch: residual attention, shape-transformer, ASPP + CBAM, and a dual-attention **Mixture-of-Experts (MoE)** decoder.
- **0.70 oil-class IoU / 0.78 F1** on a 450-tile held-out test set with a **5.2M-parameter** MoE model — outperforming a baseline **2.2× larger**.
- **Extreme class imbalance (~1% oil pixels)** handled with a weighted tile sampler, validation-tuned decision thresholds, and 95% bootstrap confidence intervals.
- **Fault-tolerant training at scale**: 127 GB dataset trained on Google Colab with per-epoch checkpointing + automatic resume, gradient checkpointing, mixed precision, and half-precision tile caching.
- **Controlled ablations**: classifier gating and a two-model ensemble with test-time augmentation (0.72 IoU) — the simpler single model was selected for delivery after a cost/benefit analysis.

## Results (held-out test set, 450 tiles)

| Model | Params | Oil IoU | Oil F1 |
|---|---|---|---|
| Dual-attention MoE U-Net (selected) | 5.2M | **0.70** | **0.78** |
| Two-model ensemble + TTA (ablation) | — | 0.72 | — |
| Larger single baseline | 11.6M | lower | — |

Full methodology, architecture diagrams, and evaluation are in [`docs/thesis.pdf`](docs/thesis.pdf); a condensed overview is in [`docs/presentation.pdf`](docs/presentation.pdf).

## Repository contents

- `sar_oil_spill_segmentation.ipynb` — full pipeline: preprocessing, augmentation, caching, five architectures, training loops, evaluation, and ablations (all outputs preserved).
- `docs/` — thesis and defense presentation.

## Running

The notebook targets Google Colab (Drive-mounted dataset). To reproduce: place the SAR tile dataset in your Drive, adjust the config cell paths, and run top-to-bottom. Checkpointing resumes automatically after Colab disconnects.

```bash
pip install -r requirements.txt
```

*The 127 GB Sentinel-1 dataset is not redistributed here.*
