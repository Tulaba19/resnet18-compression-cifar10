"""
Evaluate any saved model on CIFAR-10 across the four thesis metrics:
    - Top-1 accuracy
    - CPU inference latency (mean / std / median, batch size 1)
    - Model size on disk
    - Peak GPU memory during one forward pass (secondary metric)

Also records total parameter count. Appends one row per model to results.csv.

Standalone use:
    python evaluate.py --config config.yaml --model models/baseline_fp32.pth --name baseline_fp32 --plots
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from model import build_model, CLASSES
from data import get_test_loader

CSV_COLUMNS = [
    "name", "accuracy_pct",
    "latency_cpu_mean_ms", "latency_cpu_std_ms", "latency_cpu_median_ms",
    "size_disk_mb", "total_params_m", "peak_gpu_memory_mb",
]


def load_model(path):
    """
    Load a model from disk, handling three formats:
      1. checkpoint dict {"model": state_dict, ...}  (training output)
      2. full pickled nn.Module                       (structured-pruned models)
      3. bare state_dict
    TorchScript archives (quantized models) are loaded via torch.jit.load.
    """
    try:
        return _load_torchscript(path)
    except Exception:
        pass

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        model = build_model()
        model.load_state_dict(obj["model"])
    elif isinstance(obj, nn.Module):
        model = obj
    else:
        model = build_model()
        model.load_state_dict(obj)
    model.eval()
    return model


def _load_torchscript(path):
    m = torch.jit.load(path, map_location="cpu")
    m.eval()
    return m


def measure_accuracy(model, data_dir, return_predictions=False):
    loader = get_test_loader(data_dir, batch_size=256)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(1).numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = (all_preds == all_targets).mean() * 100
    if return_predictions:
        return acc, all_preds, all_targets
    return acc


def measure_latency(model, warmup, runs):
    """Single-image CPU latency, batch size 1, after warm-up."""
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    times_ms = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
    times_ms = np.array(times_ms)
    return float(times_ms.mean()), float(times_ms.std()), float(np.median(times_ms))


def measure_disk_size(path):
    return os.path.getsize(path) / (1024 * 1024)


def measure_parameter_count(model):
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0  # TorchScript packed-weight modules report 0 here


def measure_gpu_peak_memory(model_path):
    """
    Measure peak GPU memory in an ISOLATED subprocess.

    Running the measurement in a fresh process gives it a clean CUDA allocator,
    so the reading reflects only this model and cannot be inflated by earlier
    training or fine-tuning in the same pipeline run. Returns None if there is
    no GPU or the model is quantized (CPU-only), in which cases the helper
    prints "PEAK_MB NA".
    """
    if not torch.cuda.is_available():
        return None
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure_gpu.py")
    try:
        out = subprocess.run(
            [sys.executable, helper, model_path],
            capture_output=True, text=True, timeout=300,
        )
    except Exception:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("PEAK_MB"):
            val = line.split(maxsplit=1)[1].strip()
            if val == "NA":
                return None
            try:
                return float(val)
            except ValueError:
                return None
    return None


def append_to_csv(row, csv_path):
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def save_curves(history_path, out_path):
    if not Path(history_path).exists():
        print(f"  ({history_path} not found, skipping curves)")
        return
    with open(history_path) as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, h["train_loss"], label="train")
    ax[0].plot(epochs, h["test_loss"], label="test")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(True, alpha=.3)
    ax[1].plot(epochs, h["train_acc"], label="train")
    ax[1].plot(epochs, h["test_acc"], label="test")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy (%)")
    ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"  Saved {out_path}")


def save_confusion_matrix(preds, targets, out_path):
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"  Saved {out_path}")


def evaluate_model(model_path, name, cfg, plots=False):
    """Evaluate one model and append a row to results.csv."""
    output_dir = cfg["output_dir"]
    data_dir = cfg["data_dir"]
    ec = cfg["evaluate"]
    csv_path = os.path.join(output_dir, "results.csv")

    # Force CPU for accuracy + latency, mirroring the original setup.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    print(f"\n=== Evaluating: {name} ===")
    print(f"Model file: {model_path}")
    model = load_model(model_path)

    if plots:
        acc, preds, targets = measure_accuracy(model, data_dir, return_predictions=True)
    else:
        acc = measure_accuracy(model, data_dir)
    print(f"  Accuracy: {acc:.2f}%")

    lat_mean, lat_std, lat_median = measure_latency(model, ec["warmup_runs"], ec["timing_runs"])
    print(f"  Latency: {lat_mean:.2f} ms +/- {lat_std:.2f} (median {lat_median:.2f} ms)")

    size_mb = measure_disk_size(model_path)
    print(f"  Size on disk: {size_mb:.2f} MB")

    total_p = measure_parameter_count(model)
    print(f"  Parameters: {total_p/1e6:.2f}M")

    gpu_peak = measure_gpu_peak_memory(model_path)
    print(f"  Peak GPU memory: {gpu_peak:.2f} MB" if gpu_peak is not None
          else "  Peak GPU memory: N/A (CPU-only model or no GPU)")

    row = {
        "name": name,
        "accuracy_pct": round(acc, 3),
        "latency_cpu_mean_ms": round(lat_mean, 3),
        "latency_cpu_std_ms": round(lat_std, 3),
        "latency_cpu_median_ms": round(lat_median, 3),
        "size_disk_mb": round(size_mb, 3),
        "total_params_m": round(total_p / 1e6, 4),
        "peak_gpu_memory_mb": round(gpu_peak, 3) if gpu_peak is not None else "--",
    }
    append_to_csv(row, csv_path)
    print(f"  Appended row to {csv_path}")

    if plots:
        print("  Saving plots...")
        save_curves(os.path.join(output_dir, "history.json"),
                    os.path.join(output_dir, "curves.png"))
        save_confusion_matrix(preds, targets,
                              os.path.join(output_dir, "confusion_matrix.png"))
        print(classification_report(targets, preds, target_names=CLASSES, digits=3))
    return row


if __name__ == "__main__":
    import argparse
    from config_utils import load_config, ensure_dirs

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    evaluate_model(args.model, args.name, cfg, plots=args.plots)
