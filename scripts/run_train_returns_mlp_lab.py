from __future__ import annotations

from src.models.train_returns_mlp_lab import run_mlp_lab

if __name__ == "__main__":
    result = run_mlp_lab(
        start_date="2022-01-01",
        forecast_horizon=5,
        target_aggregation="mean",
        lags=20,
        short_window=5,
        long_window=20,
        hidden_layers="256,128,64",
        activation="relu",
        solver="adam",
        scaler="standard",
        alpha=1e-4,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        batch_size="auto",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        clip_pred=0.02,
        winsorize_quantile=0.01,
        include_abs_features=True,
        include_square_features=True,
        include_sign_features=True,
    )
    print("RUN_ID:", result["run_id"])
    print("PREDICTIONS:", result["predictions_path"])
    print("METRICS:", result["metrics_path"])
