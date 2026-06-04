"""
Post-training static INT8 quantization via PyTorch FX graph mode.

Static (not dynamic) quantization is used because ResNet-18 is almost entirely
Conv2d, which dynamic quantization leaves in FP32. FX graph mode traces the
model and handles the residual connections automatically. The quantized model
is saved as TorchScript (torch.jit.save), which evaluate.py loads directly.

Standalone use:
    python quantize_ptq.py --config config.yaml --in models/baseline_fp32.pth
"""
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from model import build_model
from data import get_calibration_loader


def _load_baseline(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()
    return model


def apply_ptq(baseline_path, cfg):
    """Quantize the baseline to INT8 and return the output path."""
    qc = cfg["quantization_ptq"]
    out_path = os.path.join(cfg["models_dir"], "ptq_int8.pth")

    model = _load_baseline(baseline_path)
    fp32_mb = os.path.getsize(baseline_path) / (1024 * 1024)
    print(f"Baseline size on disk: {fp32_mb:.2f} MB")

    qconfig_mapping = get_default_qconfig_mapping(qc.get("backend", "fbgemm"))
    example_inputs = (torch.randn(1, 3, 32, 32),)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs)

    calib_loader = get_calibration_loader(
        cfg["data_dir"], qc["calib_batches"], qc.get("calib_batch_size", 32))
    print("Calibrating...", end="", flush=True)
    with torch.no_grad():
        for x, _ in calib_loader:
            prepared(x)
            print(".", end="", flush=True)
    print(" done")

    quantized = convert_fx(prepared)
    scripted = torch.jit.script(quantized)
    torch.jit.save(scripted, out_path)

    int8_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved {out_path}  ({int8_mb:.2f} MB, {fp32_mb/int8_mb:.2f}x smaller)")
    return out_path


if __name__ == "__main__":
    import argparse
    from config_utils import load_config, ensure_dirs

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--in", dest="in_path", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    apply_ptq(args.in_path, cfg)
