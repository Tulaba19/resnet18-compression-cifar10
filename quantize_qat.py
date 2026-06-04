"""
Quantization-Aware Training (QAT) at INT8 via FX graph mode.

QAT inserts fake-quant layers and fine-tunes for a few epochs so the weights
learn to round well to INT8, then converts to a real INT8 model saved as
TorchScript. Typically recovers slightly more accuracy than PTQ, at the cost
of a short training run.

Standalone use:
    python quantize_qat.py --config config.yaml --in models/baseline_fp32.pth
"""
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

from model import build_model
from data import get_loaders


def _load_baseline(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    return model


def _eval(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, 100.0 * correct / total


def apply_qat(baseline_path, cfg, device):
    """Fine-tune with fake-quant, convert to INT8, return output path."""
    qc = cfg["quantization_qat"]
    out_path = os.path.join(cfg["models_dir"], "qat_int8.pth")

    fp32 = _load_baseline(baseline_path)
    fp32_mb = os.path.getsize(baseline_path) / (1024 * 1024)
    print(f"Baseline size on disk: {fp32_mb:.2f} MB")

    train_loader, test_loader = get_loaders(
        cfg["data_dir"], qc["batch_size"], qc["num_workers"])

    fp32.train()  # BatchNorm stats must update during fine-tuning
    qconfig_mapping = get_default_qat_qconfig_mapping(qc.get("backend", "fbgemm"))
    example_inputs = (torch.randn(1, 3, 32, 32),)
    prepared = prepare_qat_fx(fp32, qconfig_mapping, example_inputs).to(device)
    print("Inserted fake-quant layers (prepare_qat_fx)")

    optimizer = optim.SGD(prepared.parameters(), lr=qc["lr"], momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=qc["epochs"])
    criterion = nn.CrossEntropyLoss()

    print(f"QAT fine-tuning for {qc['epochs']} epochs (lr {qc['lr']})...")
    for epoch in range(1, qc["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = _train_epoch(prepared, train_loader, optimizer, criterion, device)
        te_acc = _eval(prepared, test_loader, device)
        scheduler.step()
        print(f"  epoch {epoch}/{qc['epochs']} | train_loss={tr_loss:.3f} "
              f"train_acc={tr_acc:5.2f}% test_acc={te_acc:5.2f}% "
              f"lr={scheduler.get_last_lr()[0]:.5f} {time.time()-t0:5.1f}s")

    print("Converting to INT8...")
    prepared.cpu().eval()
    quantized = convert_fx(prepared)
    scripted = torch.jit.script(quantized)
    torch.jit.save(scripted, out_path)

    int8_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved {out_path}  ({int8_mb:.2f} MB, {fp32_mb/int8_mb:.2f}x smaller)")
    return out_path


if __name__ == "__main__":
    import argparse
    from config_utils import load_config, resolve_device, ensure_dirs

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--in", dest="in_path", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    apply_qat(args.in_path, cfg, resolve_device(cfg))
