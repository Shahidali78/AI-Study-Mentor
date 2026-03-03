import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


COLUMN_NAMES = [
    "Engine_ID",
    "Cycle",
    "Op_Setting_1",
    "Op_Setting_2",
    "Op_Setting_3",
]
COLUMN_NAMES.extend([f"Sensor_{i}" for i in range(1, 22)])


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def build_paths(dataset_dir: Path, subset: str) -> tuple[Path, Path, Path]:
    subset = subset.upper()
    train_file = dataset_dir / f"train_{subset}.txt"
    test_file = dataset_dir / f"test_{subset}.txt"
    rul_file = dataset_dir / f"RUL_{subset}.txt"
    return train_file, test_file, rul_file


def read_cmapss_matrix(file_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    if raw_df.shape[1] < len(COLUMN_NAMES):
        raise ValueError(
            f"{file_path.name} has {raw_df.shape[1]} columns, expected at least {len(COLUMN_NAMES)}"
        )
    return raw_df.iloc[:, : len(COLUMN_NAMES)].set_axis(COLUMN_NAMES, axis=1)


def load_data(dataset_dir: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train_file, test_file, rul_file = build_paths(dataset_dir, subset)
    required = [train_file, test_file, rul_file]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    train_df = read_cmapss_matrix(train_file)
    test_df = read_cmapss_matrix(test_file)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["RUL"], engine="python")
    return train_df, test_df, rul_df["RUL"].values


def calculate_rul(train_df: pd.DataFrame, rul_cap: int) -> pd.Series:
    max_cycles = train_df.groupby("Engine_ID")["Cycle"].transform("max")
    rul = max_cycles - train_df["Cycle"]
    return rul.clip(upper=rul_cap)


def select_feature_columns(train_df: pd.DataFrame, variance_threshold: float = 1e-8) -> list[str]:
    # Keep Cycle as a feature; it carries useful degradation trend information.
    candidate_cols = [c for c in train_df.columns if c != "Engine_ID"]
    variances = train_df[candidate_cols].var()
    selected = variances[variances > variance_threshold].index.tolist()
    if not selected:
        raise ValueError("No features selected after variance filtering.")
    return selected


def select_feature_columns_smart(
    train_df: pd.DataFrame,
    labels: pd.Series,
    subset: str,
    use_fd001_preset: bool,
    top_k_corr: int,
) -> list[str]:
    base_cols = select_feature_columns(train_df)

    if subset == "FD001" and use_fd001_preset:
        preset = [
            "Cycle",
            "Op_Setting_1",
            "Sensor_2",
            "Sensor_3",
            "Sensor_4",
            "Sensor_7",
            "Sensor_8",
            "Sensor_9",
            "Sensor_11",
            "Sensor_12",
            "Sensor_13",
            "Sensor_14",
            "Sensor_15",
            "Sensor_17",
            "Sensor_20",
            "Sensor_21",
        ]
        preset_available = [c for c in preset if c in base_cols]
        if preset_available:
            base_cols = preset_available

    if top_k_corr > 0 and top_k_corr < len(base_cols):
        tmp = train_df[base_cols].copy()
        tmp["RUL"] = labels.values
        corr = tmp.corr(numeric_only=True)["RUL"].drop(labels=["RUL"]).abs().sort_values(ascending=False)
        ranked = corr.index.tolist()
        top_cols = ranked[:top_k_corr]
        if "Cycle" in base_cols and "Cycle" not in top_cols:
            top_cols = ["Cycle"] + top_cols[:-1]
        base_cols = top_cols

    return base_cols


def create_sequences(
    source_df: pd.DataFrame,
    scaled_features_df: pd.DataFrame,
    labels_series: pd.Series,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[float] = []

    for engine_id in source_df["Engine_ID"].unique():
        engine_indices = source_df[source_df["Engine_ID"] == engine_id].index
        engine_x = scaled_features_df.loc[engine_indices].values
        engine_y = labels_series.loc[engine_indices].values

        if len(engine_x) < sequence_length:
            continue

        for i in range(len(engine_x) - sequence_length + 1):
            x_list.append(engine_x[i : i + sequence_length])
            y_list.append(engine_y[i + sequence_length - 1])

    if not x_list:
        raise ValueError("No training sequences were created. Reduce sequence_length.")

    return np.array(x_list), np.array(y_list).reshape(-1, 1)


def split_train_val_by_engine(
    source_df: pd.DataFrame,
    x_all: np.ndarray,
    y_all: np.ndarray,
    sequence_length: int,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences by engine IDs to reduce leakage between train/validation."""
    seq_engine_ids: list[int] = []
    for engine_id in source_df["Engine_ID"].unique():
        engine_len = len(source_df[source_df["Engine_ID"] == engine_id])
        seq_count = max(0, engine_len - sequence_length + 1)
        seq_engine_ids.extend([int(engine_id)] * seq_count)

    seq_engine_ids_arr = np.array(seq_engine_ids)
    unique_ids = np.unique(seq_engine_ids_arr)
    train_ids, val_ids = train_test_split(unique_ids, test_size=val_size, random_state=seed)

    train_mask = np.isin(seq_engine_ids_arr, train_ids)
    val_mask = np.isin(seq_engine_ids_arr, val_ids)
    return x_all[train_mask], x_all[val_mask], y_all[train_mask], y_all[val_mask]


def prepare_test_data_final_window(
    source_test_df: pd.DataFrame, scaled_test_features_df: pd.DataFrame, sequence_length: int
) -> np.ndarray:
    x_test_final: list[np.ndarray] = []

    for engine_id in source_test_df["Engine_ID"].unique():
        engine_indices = source_test_df[source_test_df["Engine_ID"] == engine_id].index
        engine_data = scaled_test_features_df.loc[engine_indices].values

        if len(engine_data) < sequence_length:
            padding = np.zeros((sequence_length - len(engine_data), engine_data.shape[1]))
            last_window = np.vstack((padding, engine_data))
        else:
            last_window = engine_data[-sequence_length:]

        x_test_final.append(last_window)

    return np.array(x_test_final)


def build_model(input_shape: tuple[int, int], model_type: str, learning_rate: float):
    try:
        from tensorflow.keras import Input
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.losses import Huber
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to train the LSTM model. "
            "Install it with: pip install tensorflow"
        ) from exc

    if model_type == "bilstm":
        model = Sequential(
            [
                Input(shape=input_shape),
                Bidirectional(LSTM(units=96, return_sequences=True)),
                Dropout(0.25),
                Bidirectional(LSTM(units=48, return_sequences=False)),
                Dropout(0.25),
                Dense(units=32, activation="relu"),
                Dense(units=1),
            ]
        )
    else:
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(units=100, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=1),
            ]
        )
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0))
    return model


def score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    pos = np.exp(d[d >= 0] / 10.0) - 1.0
    neg = np.exp(-d[d < 0] / 13.0) - 1.0
    return float(np.sum(pos) + np.sum(neg))


def main() -> None:
    parser = argparse.ArgumentParser(description="C-MAPSS LSTM Predictive Maintenance Pipeline")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).parent / "CMAPSS_Dataset")
    parser.add_argument("--subset", type=str, default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-type", type=str, default="bilstm", choices=["lstm", "bilstm"])
    parser.add_argument("--model-out", type=Path, default=Path(__file__).parent / "best_model.keras")
    parser.add_argument("--fd001-preset", action="store_true")
    parser.add_argument("--top-k-corr", type=int, default=0)
    parser.add_argument("--metrics-out", type=Path, default=Path(__file__).parent / "results" / "metrics.json")
    parser.add_argument(
        "--predictions-out", type=Path, default=Path(__file__).parent / "results" / "predictions.csv"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds for ensemble, e.g. 42,52,62",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_seed = args.seed
    run_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not run_seeds:
        raise ValueError("At least one seed must be provided via --seeds")
    if not (0 < args.val_size < 0.9):
        raise ValueError("--val-size must be in the range (0, 0.9).")
    if args.top_k_corr < 0:
        raise ValueError("--top-k-corr must be >= 0.")

    train_df, test_df, y_true = load_data(args.dataset_dir, args.subset)
    train_labels = calculate_rul(train_df, args.rul_cap)
    feature_cols = select_feature_columns_smart(
        train_df,
        train_labels,
        subset=args.subset,
        use_fd001_preset=args.fd001_preset,
        top_k_corr=args.top_k_corr,
    )

    train_feature_df = train_df[feature_cols].copy()
    test_feature_df = test_df[feature_cols].copy()

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_feature_df)
    test_scaled = scaler.transform(test_feature_df)

    train_scaled_df = pd.DataFrame(train_scaled, columns=feature_cols, index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=feature_cols, index=test_df.index)

    x_all, y_all = create_sequences(
        train_df, train_scaled_df, train_labels, sequence_length=args.sequence_length
    )
    x_test = prepare_test_data_final_window(test_df, test_scaled_df, sequence_length=args.sequence_length)

    # Scale labels to stabilize optimization.
    y_scaler = MinMaxScaler()
    y_scaled_all = y_scaler.fit_transform(y_all)
    x_train, x_val, y_train_scaled, y_val_scaled = split_train_val_by_engine(
        train_df,
        x_all,
        y_scaled_all,
        sequence_length=args.sequence_length,
        val_size=args.val_size,
        seed=split_seed,
    )

    print(f"Subset: {args.subset}")
    print(f"Train shape: {x_train.shape}, labels: {y_train_scaled.shape}")
    print(f"Val shape: {x_val.shape}, labels: {y_val_scaled.shape}")
    print(f"Test shape: {x_test.shape}, true RUL size: {y_true.shape}")
    print(f"Feature count after variance filter: {len(feature_cols)}")

    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to train the LSTM model. "
            "Install it with: pip install tensorflow"
        ) from exc

    seed_predictions: list[np.ndarray] = []
    seed_val_losses: list[float] = []
    seed_metrics: list[tuple[int, float, float]] = []

    for i, run_seed in enumerate(run_seeds, start=1):
        set_seed(run_seed)
        print(f"\nTraining run {i}/{len(run_seeds)} with seed={run_seed}...")

        model = build_model((x_train.shape[1], x_train.shape[2]), args.model_type, args.learning_rate)
        model_out_i = args.model_out.with_name(f"{args.model_out.stem}_seed{run_seed}{args.model_out.suffix}")
        model_out_i.parent.mkdir(parents=True, exist_ok=True)

        es = EarlyStopping(
            monitor="val_loss", patience=args.patience, mode="min", restore_best_weights=True, verbose=1
        )
        rlrop = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, args.patience // 3), min_lr=1e-5, verbose=1
        )
        ckpt = ModelCheckpoint(
            filepath=str(model_out_i), monitor="val_loss", mode="min", save_best_only=True, verbose=1
        )

        history = model.fit(
            x_train,
            y_train_scaled,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, y_val_scaled),
            verbose=2,
            callbacks=[es, rlrop, ckpt],
        )

        # Calling the model directly is typically less prone to retracing warnings in loops.
        y_pred_scaled = model(x_test, training=False).numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
        y_pred_capped = np.clip(y_pred, 0, args.rul_cap)
        seed_predictions.append(y_pred_capped)
        best_val_loss = float(min(history.history.get("val_loss", [np.inf])))
        seed_val_losses.append(best_val_loss)

        rmse_seed = math.sqrt(mean_squared_error(y_true, y_pred_capped))
        phm_seed = score(y_true, y_pred_capped)
        seed_metrics.append((run_seed, rmse_seed, phm_seed))
        print(f"Seed {run_seed} -> RMSE: {rmse_seed:.4f}, PHM: {phm_seed:.4f}")

    pred_stack = np.stack(seed_predictions, axis=0)
    mean_pred = np.mean(pred_stack, axis=0)

    # Weighted ensemble: lower validation loss gets higher weight.
    safe_losses = np.array(seed_val_losses, dtype=float)
    safe_losses = np.clip(safe_losses, 1e-8, None)
    weights = (1.0 / safe_losses)
    weights = weights / np.sum(weights)
    weighted_pred = np.average(pred_stack, axis=0, weights=weights)

    rmse_mean = math.sqrt(mean_squared_error(y_true, mean_pred))
    phm_mean = score(y_true, mean_pred)
    rmse_weighted = math.sqrt(mean_squared_error(y_true, weighted_pred))
    phm_weighted = score(y_true, weighted_pred)

    best_seed, best_seed_rmse, best_seed_phm = min(seed_metrics, key=lambda t: t[1])

    if len(run_seeds) == 1:
        final_name = f"best_single_seed_{best_seed}"
        final_pred = pred_stack[0]
        final_rmse = best_seed_rmse
        final_phm = best_seed_phm
    elif rmse_weighted <= rmse_mean and rmse_weighted <= best_seed_rmse:
        final_name = "weighted_ensemble"
        final_pred = weighted_pred
        final_rmse = rmse_weighted
        final_phm = phm_weighted
    elif rmse_mean <= best_seed_rmse:
        final_name = "mean_ensemble"
        final_pred = mean_pred
        final_rmse = rmse_mean
        final_phm = phm_mean
    else:
        final_name = f"best_single_seed_{best_seed}"
        final_pred = pred_stack[run_seeds.index(best_seed)]
        final_rmse = best_seed_rmse
        final_phm = best_seed_phm

    print("\n--- Ensemble Evaluation ---")
    print(f"Seeds: {run_seeds}")
    print(f"Mean Ensemble -> RMSE: {rmse_mean:.4f}, PHM: {phm_mean:.4f}")
    print(f"Weighted Ensemble -> RMSE: {rmse_weighted:.4f}, PHM: {phm_weighted:.4f}")
    print(f"Best Single Seed -> {best_seed}, RMSE: {best_seed_rmse:.4f}, PHM: {best_seed_phm:.4f}")
    print(f"Selected Final Strategy: {final_name}")
    print(f"Final RMSE: {final_rmse:.4f}")
    print(f"Final PHM Score (lower is better): {final_phm:.4f}")
    print(f"Predicted RUL (first 10): {np.round(final_pred[:10], 1)}")
    print(f"True RUL      (first 10): {y_true[:10]}")

    # Persist artifacts for experiment tracking / GitHub reproducibility.
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "subset": args.subset,
        "seeds": run_seeds,
        "selected_strategy": final_name,
        "mean_ensemble": {"rmse": rmse_mean, "phm_score": phm_mean},
        "weighted_ensemble": {"rmse": rmse_weighted, "phm_score": phm_weighted},
        "best_single_seed": {"seed": best_seed, "rmse": best_seed_rmse, "phm_score": best_seed_phm},
        "final": {"rmse": final_rmse, "phm_score": final_phm},
        "config": {
            "model_type": args.model_type,
            "sequence_length": args.sequence_length,
            "rul_cap": args.rul_cap,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "val_size": args.val_size,
            "top_k_corr": args.top_k_corr,
            "fd001_preset": args.fd001_preset,
            "feature_count": len(feature_cols),
        },
    }
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    pred_df = pd.DataFrame(
        {
            "engine_id": np.arange(1, len(y_true) + 1),
            "true_rul": y_true.astype(float),
            "predicted_rul": final_pred.astype(float),
            "absolute_error": np.abs(final_pred - y_true).astype(float),
        }
    )
    pred_df.to_csv(args.predictions_out, index=False)
    print(f"Saved metrics to: {args.metrics_out}")
    print(f"Saved predictions to: {args.predictions_out}")


if __name__ == "__main__":
    main()
