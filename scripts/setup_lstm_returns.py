from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]

files = {
    "src/models/return_model_lstm.py": r'''
import torch
import torch.nn as nn


class ReturnLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        dense_size: int = 32
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)

        # Último timestep
        last_hidden = out[:, -1, :]

        y = self.head(last_hidden)
        return y.squeeze(-1)
''',

    "src/models/sequence_utils.py": r'''
import numpy as np
import pandas as pd


def build_lstm_sequences(
    series: pd.Series,
    seq_len: int = 20,
    forecast_horizon: int = 5
):
    """
    Input:
        ventana de retornos pasados de largo seq_len

    Target:
        retorno acumulado futuro de los próximos forecast_horizon días
    """
    s = series.dropna().astype(float).values

    X, y = [], []

    for i in range(seq_len, len(s) - forecast_horizon + 1):
        x_window = s[i - seq_len:i]
        y_future = s[i:i + forecast_horizon].sum()

        X.append(x_window.reshape(-1, 1))
        y.append(y_future)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y
''',

    "src/models/train_returns_lstm.py": r'''
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def train_all_return_models_lstm(
    seq_len: int = 20,
    forecast_horizon: int = 5,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2
):
    returns = load_returns()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for ticker in returns.columns:
        series = returns[ticker].dropna()

        X, y = build_lstm_sequences(
            series,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon
        )

        if len(X) < 200:
            continue

        split = int(len(X) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
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

        best_loss = np.inf
        best_state = None
        patience = 8
        patience_counter = 0

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

            if train_loss < best_loss:
                best_loss = train_loss
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

        metrics = {
            "mse": float(np.mean((y_test - y_pred_test) ** 2)),
            "rmse": rmse(y_test, y_pred_test),
            "mae": mae(y_test, y_pred_test)
        }

        # Clip defensivo del retorno acumulado futuro
        last_pred = max(min(last_pred, 0.05), -0.05)

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

        rows.append({
            "ticker": ticker,
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mu_pred": last_pred
        })

    df_out = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    df_out.to_csv(OUTPUT_DIR / "return_predictions_lstm.csv", index=False)

    print(df_out.head())
    return df_out
''',

    "scripts/run_train_returns_lstm.py": r'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.train_returns_lstm import train_all_return_models_lstm

if __name__ == "__main__":
    train_all_return_models_lstm()
    print("\nLSTM return models trained.")
''',
}


def main():
    for relative_path, content in files.items():
        path = ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        print(f"created: {relative_path}")

    print("\nAll LSTM files created successfully.")


if __name__ == "__main__":
    main()