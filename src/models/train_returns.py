from pathlib import Path
import pandas as pd

from src.data.load_data import load_returns
from src.data.features import build_features_for_ticker
from src.models.return_model_mlp import ReturnMLPModel

ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = ROOT / "models_artifacts" / "returns"
OUTPUT_DIR = ROOT / "outputs" / "predictions"


def train_all_return_models():
    returns = load_returns()

    rows = []

    for ticker in returns.columns:
        df = build_features_for_ticker(returns[ticker])

        if len(df) < 200:
            continue

        X = df.drop(columns=["ret", "target"])
        y = df["target"]

        split = int(len(df) * 0.8)

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        X_test = X.iloc[split:]
        y_test = y.iloc[split:]

        model = ReturnMLPModel()
        model.fit(X_train, y_train)

        metrics = model.evaluate(X_test, y_test)

        pred = float(model.predict(X.tail(1))[0])

        # Escalar la predicción a magnitud comparable con el histórico
        hist_std = float(returns[ticker].std())
        pred = pred * hist_std

        # Clip defensivo para evitar retornos diarios absurdos
        pred = max(min(pred, 0.02), -0.02)

        model_path = ARTIFACTS_DIR / f"{ticker}_mlp.joblib"
        model.save(model_path)

        rows.append({
            "ticker": ticker,
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mu_pred": pred
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_DIR / "return_predictions.csv", index=False)

    print(df_out.head())

    return df_out