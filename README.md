# Benchmarking Quantization and Pruning on ResNet-18 / CIFAR-10

A small, reproducible benchmark that trains a ResNet-18 baseline on CIFAR-10 and
compares two compression techniques — **INT8 quantization** (post-training and
quantization-aware) and **structured (channel) pruning** — across four
deployment-relevant metrics:

- Top-1 accuracy
- CPU inference latency (batch size 1)
- Model size on disk
- Peak GPU memory during inference (secondary)

Everything is driven by a single `config.yaml` and one command. You can train
your own model and tweak epochs, learning rate, pruning ratios, and so on
without editing any code.

> Scope: this benchmark targets ResNet-18 on CIFAR-10. Hyperparameters are
> fully configurable, but the model and dataset are fixed by design, so the
> reported results stay directly comparable.

## 1. Setup

Requires Python Python 3.11. A CUDA GPU is recommended for training (an 8 GB card is
plenty); everything also runs on CPU, just slower.

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows:  .\.venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# 1) Install PyTorch with the right CUDA build from the official selector:
#    https://pytorch.org/get-started/locally/
#    (example for CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) Install the remaining dependencies
pip install -r requirements.txt
```

Quick GPU check (should print `True`):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Run the whole benchmark

```bash
python run.py --config config.yaml
```

This trains the baseline, applies PTQ, QAT, and structured pruning at each
configured ratio, evaluates every resulting model, and writes all metrics to
`outputs/results.csv`. Models and plots are also saved under `outputs/`.

Already have a trained baseline at `models/baseline_fp32.pth`? Skip training:

```bash
python run.py --config config.yaml --skip-train
```

## 3. Configure

Open `config.yaml` and edit any value: number of epochs, learning rate, batch
size, quantization backend, the list of pruning ratios, evaluation timing runs,
and so on. Set `enabled: false` on any stage you want to skip.

## 4. Run a single stage (optional)

Each stage also runs on its own, which is handy for experimenting:

```bash
python train.py            --config config.yaml
python quantize_ptq.py     --config config.yaml --in models/baseline_fp32.pth
python quantize_qat.py     --config config.yaml --in models/baseline_fp32.pth
python prune_structured.py --config config.yaml --in models/baseline_fp32.pth
python evaluate.py         --config config.yaml --model models/baseline_fp32.pth --name baseline_fp32 --plots
```

Note: every compression stage needs a trained baseline as input (your own, or
a released one from `models/`). Training is the only stage that runs on its own.

## 5. Project layout

```
config.yaml          all hyperparameters
run.py               single entry point
model.py             shared ResNet-18 (CIFAR variant) + constants
data.py              CIFAR-10 loaders
train.py             baseline training
quantize_ptq.py      post-training INT8 quantization
quantize_qat.py      quantization-aware training INT8
prune_structured.py  structured channel pruning (torch-pruning / DepGraph)
evaluate.py          four-metric evaluation -> outputs/results.csv
measure_gpu.py       isolated GPU peak-memory probe (run per model in a subprocess)
models/              trained + released models (baseline_fp32.pth, ptq_int8.pth, ...)
outputs/             results.csv, history.json, plots
RESULTS.md           headline results from the reference run
```

Trained and compressed models are written to `models/` (the baseline is saved
as `baseline_fp32.pth`, matching its row name in `results.csv`). Run artifacts
(metrics CSV, training-curve and confusion-matrix plots) go to `outputs/`.

If the machine runs hot doing train -> PTQ -> QAT -> pruning back to back, set
`pause_between_stages: true` in the config and the run will wait for you to
press Enter before each stage, so you can let it cool down.

## Notes

- Quantized models are saved as TorchScript and run on **CPU only** (a PyTorch
  limitation). `evaluate.py` detects this and reports peak GPU memory as `N/A`
  for them.
- Structured-pruned models are saved as full objects (channel counts change),
  so they are loaded directly rather than via `state_dict`.
- Latency and accuracy are always measured on CPU for a consistent comparison.

## License

MIT (see LICENSE).
