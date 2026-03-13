from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_returns
from src.models.return_model_lstm import ReturnLSTM
from src.models.lstm_features import build_lstm_feature_frame, select_feature_columns
from src.models.sequence_utils import build_lstm_sequences_multifeature

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "lstm_lab" / "runs"
MODEL_ROOT = ROOT / "models_artifacts" / "returns_lstm" / "runs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def temporal_split(X, y, train_frac=0.70, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:],
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def run_lstm_lab(
    selected_features: list[str],
    seq_len: int = 20,
    forecast_horizon: int = 5,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    tickers: list[str] | None = None,
):
    returns = load_returns()

    if tickers:
        returns = returns[[c for c in tickers if c in returns.columns]]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = OUTPUT_ROOT / run_id
    run_model_dir = MODEL_ROOT / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    pred_rows = []

    for ticker in returns.columns:
        series = returns[ticker].dropna()

        feat_df = build_lstm_feature_frame(series)
        feat_df = select_feature_columns(feat_df, selected_features)

        common_idx = feat_df.index.intersection(series.index)
        feat_df = feat_df.loc[common_idx]
        target_series = series.loc[common_idx]

        X, y = build_lstm_sequences_multifeature(
            feature_df=feat_df,
            target_series=target_series,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
        )

        if len(X) < 250:
            continue

        X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y)

        if min(len(X_train), len(X_val), len(X_test)) == 0:
            continue

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=False,
        )

        model = ReturnLSTM(
            input_size=X.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            dense_size=32,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()

        best_val = np.inf
        best_state = None
        train_losses, val_losses = [], []

        for _ in range(epochs):
            model.train()
            batch_losses = []

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_losses.append(loss.item())

            train_loss = float(np.mean(batch_losses))
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t.to(DEVICE))
                val_loss = float(criterion(val_pred, y_val_t.to(DEVICE)).item())

            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_t.to(DEVICE)).cpu().numpy()
            last_pred = float(model(torch.tensor(X[-1:], dtype=torch.float32).to(DEVICE)).cpu().numpy()[0])

        hist_mean = float(series.mean())
        hist_std = float(series.std())
        lower = hist_mean - 2.5 * hist_std
        upper = hist_mean + 2.5 * hist_std
        last_pred = float(np.clip(last_pred, lower, upper))
        last_pred = float(np.clip(last_pred, -0.02, 0.02))

        torch.save(model.state_dict(), run_model_dir / f"{ticker}_lstm.pt")

        with open(run_model_dir / f"{ticker}_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ticker": ticker,
                    "selected_features": selected_features,
                    "seq_len": seq_len,
                    "forecast_horizon": forecast_horizon,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "lr": lr,
                    "metrics": {
                        "rmse_test": rmse(y_test, y_pred_test),
                        "mae_test": mae(y_test, y_pred_test),
                        "best_val_loss": best_val,
                    },
                },
                f,
                indent=2,
            )

        summary_rows.append(
            {
                "ticker": ticker,
                "rmse_test": rmse(y_test, y_pred_test),
                "mae_test": mae(y_test, y_pred_test),
                "best_val_loss": best_val,
            }
        )
        pred_rows.append(
            {
                "ticker": ticker,
                "mu_pred": last_pred,
            }
        )

    predictions_df = pd.DataFrame(pred_rows).sort_values("ticker").reset_index(drop=True)
    metrics_df = pd.DataFrame(summary_rows).sort_values("ticker").reset_index(drop=True)

    predictions_df.to_csv(run_output_dir / "predictions.csv", index=False)
    metrics_df.to_csv(run_output_dir / "metrics.csv", index=False)

    with open(run_output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "selected_features": selected_features,
                "seq_len": seq_len,
                "forecast_horizon": forecast_horizon,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "tickers": list(returns.columns),
            },
            f,
            indent=2,
        )

    return {
        "run_id": run_id,
        "predictions_path": str(run_output_dir / "predictions.csv"),
        "metrics_path": str(run_output_dir / "metrics.csv"),
        "model_dir": str(run_model_dir),
        "predictions": predictions_df,
        "metrics": metrics_df,
    }
