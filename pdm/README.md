# Predictive Maintenance on NASA C-MAPSS

End-to-end Remaining Useful Life (RUL) prediction using LSTM/BiLSTM on the C-MAPSS turbofan degradation dataset.

## Features

- supports `FD001`, `FD002`, `FD003`, `FD004`
- engine-aware train/validation split to reduce leakage
- LSTM and BiLSTM architectures
- early stopping + LR scheduling + best-model checkpointing
- single-seed and multi-seed evaluation
- automatic strategy selection among:
  - best single seed
  - mean ensemble
  - weighted ensemble (by validation loss)
- exports reproducible experiment artifacts:
  - `results/metrics.json`
  - `results/predictions.csv`

## Project Structure

```text
pdm/
├─ predictive_maintenance.py
├─ requirements.txt
├─ README.md
└─ CMAPSS_Dataset/
```

## Setup

```powershell
cd C:\AI_Work\pdm
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quick Start

```powershell
python predictive_maintenance.py --model-type bilstm --epochs 60 --batch-size 128 --top-k-corr 0 --seeds 42
```

## Recommended Reproducible Run

```powershell
python predictive_maintenance.py --subset FD001 --model-type bilstm --sequence-length 50 --rul-cap 125 --epochs 60 --batch-size 128 --patience 10 --val-size 0.2 --learning-rate 0.001 --top-k-corr 0 --seeds 42
```

## Key CLI Options

- `--subset {FD001,FD002,FD003,FD004}`
- `--model-type {lstm,bilstm}`
- `--seeds 42,52,62` (comma-separated)
- `--top-k-corr 0` (`0` keeps all selected features)
- `--fd001-preset` (optional sensor preset for FD001)
- `--metrics-out results/metrics.json`
- `--predictions-out results/predictions.csv`

## Outputs

Terminal output includes train/val/test shapes, feature count, per-seed metrics, ensemble metrics, and the final selected strategy.

Saved files:

- `results/metrics.json`: full config + RMSE/PHM metrics
- `results/predictions.csv`: per-engine true/predicted RUL + absolute error
- `best_model_seed*.keras`: best checkpoint for each seed
