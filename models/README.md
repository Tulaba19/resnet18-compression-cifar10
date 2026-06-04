# Pre-trained models

This folder holds the released models so others can reproduce the numbers in
`RESULTS.md` or use the compressed models directly, without retraining.

Expected files (add your own here before committing):

- `baseline_fp32.pth`     FP32 baseline (checkpoint dict with "model" key)
- `ptq_int8.pth`          post-training INT8 (TorchScript)
- `qat_int8.pth`          QAT INT8 (TorchScript)
- `structured_10.pth` ... `structured_90.pth`   structured-pruned (full objects)

## How to use a released model

Evaluate any of them directly:

```bash
python evaluate.py --config config.yaml --model models/baseline_fp32.pth --name baseline_fp32 --plots
python evaluate.py --config config.yaml --model models/ptq_int8.pth   --name ptq_int8
```

To run only compression on the released baseline (no training), make sure
`models/baseline_fp32.pth` is present, then:

```bash
python run.py --config config.yaml --skip-train
```

## Important: these are large files — use Git LFS

The `.pth` files can be tens of MB. This repo's `.gitattributes` already tracks
`*.pth` with Git LFS. Install it once before committing the models:

```bash
git lfs install
git add .gitattributes
git add models/*.pth
git commit -m "Add pre-trained models via LFS"
```

Without LFS, GitHub will reject files over 100 MB and warn above 50 MB.
