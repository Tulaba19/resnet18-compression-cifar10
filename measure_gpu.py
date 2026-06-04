"""
Measure peak GPU memory for a single model, in isolation.

This script is meant to be launched as a SEPARATE process (by evaluate.py).
Running it in a fresh process guarantees a clean CUDA allocator, so the peak
memory reading reflects only this model and is never contaminated by earlier
training or fine-tuning in a long-running pipeline.

It prints exactly one line to stdout:
    PEAK_MB <value>     e.g. "PEAK_MB 52.502"
    PEAK_MB NA          if no GPU, or the model is quantized (CPU-only)

Usage:
    python measure_gpu.py <model_path>
"""
import sys

import torch
import torch.nn as nn

from evaluate import load_model


def measure(model_path):
    if not torch.cuda.is_available():
        return None
    try:
        model = load_model(model_path)
    except Exception:
        return None
    # Quantized models are TorchScript / CPU-only -> not applicable.
    if isinstance(model, torch.jit.ScriptModule):
        return None
    if any("quantized" in type(m).__module__ for m in model.modules()):
        return None
    try:
        model = model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = model.to("cuda")
        dummy = torch.randn(1, 3, 32, 32, device="cuda")
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("PEAK_MB NA")
        sys.exit(0)
    peak = measure(sys.argv[1])
    print(f"PEAK_MB {peak:.3f}" if peak is not None else "PEAK_MB NA")
