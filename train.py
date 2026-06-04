"""
Train the ResNet-18 / CIFAR-10 baseline.

Mirrors the original train.py exactly (SGD + Nesterov, cosine schedule, mixed
precision), but reads all hyperparameters from the config instead of module
constants. Saves baseline_fp32.pth + last_model.pth into models/ and history.json into the
output directory.

Standalone use:
    python train.py --config config.yaml
"""
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from model import build_model
from data import get_loaders


def run_epoch(loader, model, optimizer, scaler, criterion, device, train):
    """One pass over `loader`. Returns (mean_loss, accuracy_pct)."""
    model.train(train)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, 100.0 * correct / total


def train_baseline(cfg, device):
    """Train the baseline and return the path to baseline_fp32.pth."""
    tc = cfg["train"]
    models_dir = cfg["models_dir"]
    output_dir = cfg["output_dir"]
    data_dir = cfg["data_dir"]
    train_loader, test_loader = get_loaders(
        data_dir, tc["batch_size"], tc["num_workers"])

    model = build_model(cfg["model"]["num_classes"]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=tc["lr"], momentum=tc["momentum"],
                          weight_decay=tc["weight_decay"], nesterov=tc.get("nesterov", True))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}
    best_acc = 0.0
    best_path = os.path.join(models_dir, "baseline_fp32.pth")
    last_path = os.path.join(models_dir, "last_model.pth")

    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, model, optimizer, scaler, criterion, device, True)
        te_loss, te_acc = run_epoch(test_loader, model, optimizer, scaler, criterion, device, False)
        scheduler.step()

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss); history["test_acc"].append(te_acc)
        history["lr"].append(scheduler.get_last_lr()[0])

        print(f"Epoch {epoch:3d}/{tc['epochs']} | "
              f"train {tr_loss:.3f}/{tr_acc:5.2f}% | "
              f"test {te_loss:.3f}/{te_acc:5.2f}% | "
              f"lr {scheduler.get_last_lr()[0]:.4f} | {time.time()-t0:5.1f}s")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"epoch": epoch, "model": model.state_dict(), "test_acc": te_acc}, best_path)

    torch.save({"epoch": tc["epochs"], "model": model.state_dict(), "test_acc": te_acc}, last_path)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nBest test accuracy: {best_acc:.2f}%  (saved to {best_path})")
    return best_path


if __name__ == "__main__":
    import argparse
    from config_utils import load_config, resolve_device, ensure_dirs

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device = resolve_device(cfg)
    print(f"Using device: {device}")
    train_baseline(cfg, device)
