"""Production uses the same validated multifeature pipeline as the laboratory."""
from pathlib import Path
import json
from src.models.train_returns_lstm_lab import run_lstm_lab
from src.models.lstm_features import AVAILABLE_FEATURES


def train_all_return_models_lstm(seq_len=40, forecast_horizon=5, epochs=80,
                                 batch_size=32, lr=1e-3, hidden_size=64,
                                 num_layers=2, dropout=0.2, patience=10,
                                 selected_features=None, candidates=None):
    result = run_lstm_lab(selected_features=selected_features or AVAILABLE_FEATURES,
                          seq_len=seq_len, forecast_horizon=forecast_horizon,
                          epochs=epochs, batch_size=batch_size, lr=lr,
                          hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, patience=patience, candidates=candidates,
                          start_date="full", persist_db=False, feature_search=selected_features is None)
    predictions = result["predictions"].merge(result["metrics"].drop(columns=["mu_pred"], errors="ignore"), on="ticker")
    if predictions.empty:
        raise RuntimeError(f"No LSTM forecasts generated; inspect {result['metrics_path']}")
    output = Path(__file__).resolve().parents[2] / "outputs" / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "return_predictions_lstm.csv", index=False)
    (output / "lstm_production.json").write_text(json.dumps({"run_id": result["run_id"],
        "run_output_dir": str(Path(result["predictions_path"]).parent)}, indent=2), encoding="utf-8")
    return predictions
