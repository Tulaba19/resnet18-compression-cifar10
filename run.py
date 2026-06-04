"""
Single entry point for the whole benchmark.

    python run.py --config config.yaml

Reads config.yaml, runs each enabled stage in order, evaluates every model it
produces, and writes all metrics to outputs/results.csv. Trained and compressed
models are saved under models/.

Options:
    --skip-train   Use an existing models/baseline_fp32.pth instead of training.

Set "pause_between_stages: true" in the config to wait for Enter between the
big stages (handy if the machine gets hot and you want to let it cool down).
"""
import argparse
import os

from config_utils import load_config, resolve_device, ensure_dirs
from train import train_baseline
from evaluate import evaluate_model
from quantize_ptq import apply_ptq
from quantize_qat import apply_qat


def maybe_pause(cfg, upcoming):
    if cfg.get("pause_between_stages", False):
        try:
            input(f"\n[paused] Next stage: {upcoming}. "
                  f"Press Enter to continue (Ctrl-C to stop)... ")
        except EOFError:
            pass  # non-interactive run: just continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip-train", action="store_true",
                    help="Use an existing models/baseline_fp32.pth instead of training.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device = resolve_device(cfg)
    print(f"Device: {device}")

    baseline_path = os.path.join(cfg["models_dir"], "baseline_fp32.pth")

    # --- Stage 1: train baseline ---
    if cfg["train"]["enabled"] and not args.skip_train:
        baseline_path = train_baseline(cfg, device)
    elif not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"No baseline at {baseline_path}. Enable training, or place a "
            f"baseline there and use --skip-train.")

    # Evaluate the baseline (with plots).
    evaluate_model(baseline_path, "baseline_fp32", cfg, plots=True)

    # --- Stage 2: PTQ ---
    if cfg["quantization_ptq"]["enabled"]:
        maybe_pause(cfg, "post-training quantization (PTQ)")
        p = apply_ptq(baseline_path, cfg)
        evaluate_model(p, "ptq_int8", cfg)

    # --- Stage 3: QAT ---
    if cfg["quantization_qat"]["enabled"]:
        maybe_pause(cfg, "quantization-aware training (QAT)")
        p = apply_qat(baseline_path, cfg, device)
        evaluate_model(p, "qat_int8", cfg)

    # --- Stage 4: structured pruning ---
    if cfg["pruning_structured"]["enabled"]:
        maybe_pause(cfg, "structured pruning")
        from prune_structured import apply_structured_pruning  # lazy: imports torch_pruning
        for name, path in apply_structured_pruning(baseline_path, cfg, device):
            evaluate_model(path, name, cfg)

    print(f"\nDone. Metrics written to {os.path.join(cfg['output_dir'], 'results.csv')}")


if __name__ == "__main__":
    main()
