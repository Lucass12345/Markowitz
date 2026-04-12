from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_returns
from src.models.sequence_utils import build_lstm_sequences
from src.models.return_model_lstm import ReturnLSTM

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "models_artifacts" / "returns_lstm"
OUTPUT_DIR = ROOT / "outputs" / "predictions"
LOSS_DIR = OUTPUT_DIR / "lstm_loss_curves"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def temporal_split(X, y, train_frac=0.70, val_frac=0.15):
    n = len(X)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_all_return_models_lstm(
    seq_len: int = 20,
    forecast_horizon: int = 5,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    patience: int = 10,
):
    returns = load_returns()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOSS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for ticker in returns.columns:
        series = returns[ticker].dropna()

        X, y = build_lstm_sequences(
            series,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon
        )

        if len(X) < 250:
            continue

        X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y)

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            continue

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)

        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

        model = ReturnLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            dense_size=32
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = np.inf
        best_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_losses = []

            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses))
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t.to(DEVICE))
                val_loss = float(criterion(val_pred, y_val_t.to(DEVICE)).item())

            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()

        with torch.no_grad():
            y_pred_test = model(X_test_t.to(DEVICE)).cpu().numpy()
            last_pred = float(
                model(torch.tensor(X[-1:], dtype=torch.float32).to(DEVICE)).cpu().numpy()[0]
            )

        # Clip defensivo del retorno acumulado futuro a 5 días
        last_pred = max(min(last_pred, 0.02), -0.02)

        metrics = {
            "mse_test": float(np.mean((y_test - y_pred_test) ** 2)),
            "rmse_test": rmse(y_test, y_pred_test),
            "mae_test": mae(y_test, y_pred_test),
            "best_val_loss": float(best_val_loss),
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "epochs_trained": int(len(train_losses)),
        }

        model_path = ARTIFACTS_DIR / f"{ticker}_lstm.pt"
        torch.save(model.state_dict(), model_path)

        meta_path = ARTIFACTS_DIR / f"{ticker}_lstm_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "ticker": ticker,
                "seq_len": seq_len,
                "forecast_horizon": forecast_horizon,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "metrics": metrics
            }, f, indent=2)

        loss_df = pd.DataFrame({
            "epoch": range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss": val_losses
        })
        loss_df.to_csv(LOSS_DIR / f"{ticker}_loss.csv", index=False)

        rows.append({
            "ticker": ticker,
            "mse_test": metrics["mse_test"],
            "rmse_test": metrics["rmse_test"],
            "mae_test": metrics["mae_test"],
            "best_val_loss": metrics["best_val_loss"],
            "epochs_trained": metrics["epochs_trained"],
            "train_size": metrics["train_size"],
            "val_size": metrics["val_size"],
            "test_size": metrics["test_size"],
            "mu_pred": last_pred
        })

    df_out = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    df_out.to_csv(OUTPUT_DIR / "return_predictions_lstm.csv", index=False)

    summary = {
        "n_assets": int(len(df_out)),
        "mean_mse_test": float(df_out["mse_test"].mean()) if len(df_out) else None,
        "mean_rmse_test": float(df_out["rmse_test"].mean()) if len(df_out) else None,
        "mean_mae_test": float(df_out["mae_test"].mean()) if len(df_out) else None,
        "mean_best_val_loss": float(df_out["best_val_loss"].mean()) if len(df_out) else None,
    }

    with open(OUTPUT_DIR / "lstm_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(df_out.head())
    print("\nSummary:")
    print(summary)

    return df_out
