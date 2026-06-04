"""
Small helpers for loading the YAML config and resolving common settings.
"""
import os
import torch
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(cfg):
    choice = str(cfg.get("device", "auto")).lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(cfg):
    os.makedirs(cfg["data_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg.get("models_dir", "./models"), exist_ok=True)


def results_csv_path(cfg):
    return os.path.join(cfg["output_dir"], "results.csv")
