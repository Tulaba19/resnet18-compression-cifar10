# Reference Results

Headline metrics from the reference run (ResNet-18 / CIFAR-10, baseline trained
to 94.42% top-1). Latency is single-image CPU inference; peak GPU memory is the
secondary metric and is not available for the CPU-only quantized models.

| Config         | Acc (%) | Latency (ms) | Disk (MB) | Params (M) | Peak GPU (MB) |
|----------------|--------:|-------------:|----------:|-----------:|--------------:|
| baseline_fp32  |   94.42 |        14.12 |     42.70 |      11.17 |         61.13 |
| structured_10  |   94.19 |         6.82 |     34.48 |       9.02 |         43.60 |
| structured_30  |   93.61 |         5.07 |     20.90 |       5.46 |         30.02 |
| structured_50  |   92.71 |         3.16 |     10.74 |       2.80 |         19.85 |
| structured_70  |   90.67 |         1.64 |      3.86 |       1.00 |         13.00 |
| structured_90  |   73.35 |         0.95 |      0.47 |       0.11 |          9.61 |
| ptq_int8       |   94.42 |         2.35 |     10.79 |          — |           N/A |
| qat_int8       |   94.41 |         2.33 |     10.79 |          — |           N/A |

Notes:
- Quantized models report ~0 parameters because FX packs the weights; for them
  `size_disk_mb` is the source of truth.
- The full per-run CSV (including latency std/median) is produced at
  `outputs/results.csv` when you run the benchmark.

Hardware/software for the reference run: _fill in your CPU, GPU, OS, Python and
PyTorch versions here so others can interpret the latency numbers._

## Note on the peak GPU memory metric

Peak GPU memory is measured in an **isolated subprocess**, one per model
(`measure_gpu.py`). This is deliberate: measuring it inside the long-running
pipeline gives a fresh CUDA allocator no chance to be contaminated by the
preceding training or fine-tuning, which would otherwise inflate the reading.
The single `python run.py` command handles this automatically.

As with latency, the absolute MB values are hardware- and driver-specific, so a
different GPU will not reproduce the exact numbers. The *trend* reproduces:
pruned models use less peak memory than the baseline, decreasing as more
channels are removed. Quantized models are CPU-only and report `N/A`.
