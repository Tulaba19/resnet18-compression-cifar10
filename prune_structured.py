"""
Structured (filter/channel) magnitude pruning via torch-pruning (DepGraph).

Whole output channels are removed, so tensors physically shrink and disk size
and latency actually drop. ResNet's residual connections couple channels across
layers; torch-pruning's dependency graph propagates removals automatically.

For each target ratio a FRESH baseline is loaded (structured pruning is
destructive and not cleanly composable), pruned in a single step, fine-tuned,
and saved as a FULL model object (channel counts change, so a state_dict reload
would not match). evaluate.py loads these via its nn.Module branch.

Install:  pip install torch-pruning
Standalone use:
    python prune_structured.py --config config.yaml --in models/baseline_fp32.pth
"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp

from model import build_model
from data import get_loaders
from train import run_epoch  # identical AMP training loop


def _finetune(model, train_loader, test_loader, epochs, lr, device):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=0.0005, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    final_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        _, _ = run_epoch(train_loader, model, optimizer, scaler, criterion, device, True)
        _, te_acc = run_epoch(test_loader, model, optimizer, scaler, criterion, device, False)
        scheduler.step()
        final_acc = te_acc
        print(f"    epoch {epoch:2d}/{epochs} | test {te_acc:5.2f}% | {time.time()-t0:4.1f}s")
    return final_acc


def _build_pruner(model, ratio, example_inputs):
    importance = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return tp.pruner.MagnitudePruner(
        model, example_inputs, importance=importance,
        pruning_ratio=ratio, ignored_layers=ignored_layers)


def apply_structured_pruning(baseline_path, cfg, device):
    """Prune to each ratio, fine-tune, save. Returns list of (name, path)."""
    pc = cfg["pruning_structured"]
    train_loader, test_loader = get_loaders(
        cfg["data_dir"], pc["batch_size"], pc["num_workers"])
    example_inputs = torch.randn(1, 3, 32, 32).to(device)

    outputs = []
    base_params = None
    for ratio in pc["ratios"]:
        pct = int(round(ratio * 100))
        print(f"\n=== Structured pruning: {pct}% channels removed ===")
        ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
        model = build_model()
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        model.to(device)

        if base_params is None:
            base_params = sum(p.numel() for p in model.parameters())

        pruner = _build_pruner(model, ratio, example_inputs)
        pruner.step()
        pruned_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {base_params/1e6:.2f}M -> {pruned_params/1e6:.2f}M "
              f"({100*(1-pruned_params/base_params):.1f}% removed)")

        model.eval()
        with torch.no_grad():
            assert model(example_inputs).shape == (1, 10)

        print(f"Fine-tuning for {pc['finetune_epochs']} epochs...")
        acc = _finetune(model, train_loader, test_loader,
                        pc["finetune_epochs"], pc["lr"], device)

        out_path = os.path.join(cfg["models_dir"], f"structured_{pct}.pth")
        model.cpu().eval()
        torch.save(model, out_path)
        print(f"Saved {out_path}  (test acc {acc:.2f}%, {pruned_params/1e6:.2f}M params)")
        outputs.append((f"structured_{pct}", out_path))
    return outputs


if __name__ == "__main__":
    import argparse
    from config_utils import load_config, resolve_device, ensure_dirs

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--in", dest="in_path", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    apply_structured_pruning(args.in_path, cfg, resolve_device(cfg))
