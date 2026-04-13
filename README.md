# Machine Fault Recognition (Deep Learning)

This repository is organized for a 4-student team and an end-to-end deep learning workflow:

1. Preprocessing
2. Feature extraction
3. Training
4. Analysis and evaluation

## Recommended Project Structure

```text
Machine-Fault-Recognition/
|-- configs/                       # YAML/JSON experiment configs
|-- data/
|   |-- raw/                       # Immutable original data
|   |-- external/                  # Third-party data
|   |-- interim/                   # Temporary outputs between stages
|   `-- processed/                 # Final model-ready datasets
|-- docs/                          # Documentation and team coordination notes
|-- models/
|   |-- checkpoints/               # Training checkpoints (.pt, .ckpt)
|   `-- exports/                   # Final exported models (ONNX, TorchScript)
|-- notebooks/
|   |-- 01_preprocessing/
|   |-- 02_feature_extraction/
|   |-- 03_training/
|   `-- 04_analysis/
|-- reports/
|   |-- figures/                   # Plots and visual outputs
|   `-- tables/                    # Metrics tables and summaries
|-- scripts/                       # CLI scripts for stage-by-stage runs
|-- src/
|   |-- preprocessing/             # Student 1 ownership
|   |-- feature_extraction/        # Student 2 ownership
|   |-- training/                  # Student 3 ownership
|   |-- analysis/                  # Student 4 ownership
|   `-- utils/                     # Shared helpers (I/O, logging, seeds)
|-- tests/
|   |-- preprocessing/
|   |-- feature_extraction/
|   |-- training/
|   `-- analysis/
|-- .gitignore
`-- LICENSE
```

## Team Ownership

- Student 1: `src/preprocessing/`, `tests/preprocessing/`, `notebooks/01_preprocessing/`
- Student 2: `src/feature_extraction/`, `tests/feature_extraction/`, `notebooks/02_feature_extraction/`
- Student 3: `src/training/`, `tests/training/`, `notebooks/03_training/`
- Student 4: `src/analysis/`, `tests/analysis/`, `notebooks/04_analysis/`

Shared ownership:

- `configs/`, `src/utils/`, `reports/`, `docs/`

## Best-Practice Workflow

- Keep `data/raw/` unchanged after initial import.
- Promote outputs stage-by-stage: `raw -> interim -> processed`.
- Drive experiments from `configs/` to keep runs reproducible.
- Save checkpoints during training and only commit lightweight artifacts.
- Keep notebooks for exploration; move stable logic into `src/`.
- Add tests per module in `tests/`.

## Suggested Naming Conventions

- Configs: `configs/exp_<model>_<dataset>.yaml`
- Checkpoints: `models/checkpoints/<exp_name>/epoch_<n>.pt`
- Reports: `reports/figures/<exp_name>/...`
