"""MLP selection must never consult test results or historical test-based scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app import mlp_lab_storage as storage
from src.models.train_returns_mlp_lab import build_composite_score


def validation_rows():
    return pd.DataFrame({"ticker": ["A", "B"], "mu_pred": [.001, .002],
        "dir_acc_val": [.8, .4], "corr_val": [.7, -.1], "rmse_val": [.01, .03],
        "positive_precision_val": [.8, .3], "dir_acc_test": [0., 1.],
        "corr_test": [-1., 1.], "rmse_test": [10., 0.],
        "positive_precision_test": [0., 1.], "score": [0., 100.]})


def test_composite_keeps_weights_but_is_invariant_to_test_metrics():
    rows = validation_rows()
    score = build_composite_score(rows)
    # Two eligible rows: A's mu rank=.5 and four validation terms favor A.
    assert score.iloc[0] == pytest.approx(.35 * .5 + .25 + .15 + .15 * .5 + .10)
    changed = rows.copy()
    changed[["dir_acc_test", "corr_test", "rmse_test", "positive_precision_test", "score"]] = 999.
    pd.testing.assert_series_equal(score, build_composite_score(changed))


@pytest.mark.parametrize("policy", [
    {"policy_type": "all_assets"}, {"policy_type": "dir_acc_min", "dir_acc_min": .6},
    {"policy_type": "corr_and_rmse", "corr_min": .2, "rmse_max": .02},
    {"policy_type": "composite_min_score", "min_score": .7},
    {"policy_type": "composite_top_n", "top_n": 1},
    {"policy_type": "hybrid_quality", "dir_acc_min": .6, "corr_min": .2, "min_score": .7},
])
def test_every_policy_uses_validation_and_recomputes_old_scores(policy):
    rows = validation_rows()
    result = storage._apply_selection_policy(rows, policy)
    changed = rows.copy()
    changed[["dir_acc_test", "corr_test", "rmse_test", "positive_precision_test", "score"]] = -999.
    other = storage._apply_selection_policy(changed, policy)
    pd.testing.assert_series_equal(result.use_for_optimizer, other.use_for_optimizer)
    pd.testing.assert_series_equal(result.score, other.score)
    assert result.selection_split.eq("validation").all()
    if policy["policy_type"] != "all_assets":
        assert result.use_for_optimizer.tolist() == [True, False]


def test_missing_validation_cannot_be_rescued_by_perfect_test_or_old_score():
    rows = validation_rows().drop(columns="positive_precision_val")
    rows[["dir_acc_test", "corr_test", "positive_precision_test", "score"]] = 1.
    rows.rmse_test = 0.
    result = storage._apply_selection_policy(rows, {"policy_type": "all_assets"})
    assert not result.use_for_optimizer.any()
    assert result.selection_reason.eq("missing_validation_evidence").all()
    assert result.score.isna().all()


def test_storage_and_loader_restore_validation_precision_from_files(tmp_path, monkeypatch):
    from src.optimization.mlp_lab_loader import load_active_mlp_predictions
    rows = validation_rows()
    active = {"run_id": "validation-run", "run_output_dir": str(tmp_path),
              "selection_policy_json": {"policy_type": "composite_top_n", "top_n": 1}}
    rows.drop(columns="score").to_csv(tmp_path / "metrics.csv", index=False)
    pd.DataFrame({"ticker": rows.ticker, "dataset_version": "pinned-v1", "base_currency": "USD",
                  "as_of": "2024-01-08", "prediction_units": "daily_log_return"}).to_csv(tmp_path / "predictions.csv", index=False)
    db_rows = rows.drop(columns="positive_precision_val")
    monkeypatch.setattr(storage, "get_active_mlp_prediction_run", lambda: active)
    monkeypatch.setattr(storage, "get_mlp_run_tickers_df", lambda run_id: db_rows.copy())
    public = storage.load_active_mlp_predictions_df()
    loaded = load_active_mlp_predictions(validate_current=False)
    assert public.use_for_optimizer.tolist() == loaded.use_for_optimizer.tolist() == [True, False]
    assert loaded.model_run_id.eq("validation-run").all()
    assert loaded.positive_precision_val.tolist() == [.8, .3]
    assert loaded.as_of.eq("2024-01-08").all()
    assert loaded.selection_policy.tolist() == [active["selection_policy_json"]] * 2


def test_lab_exports_daily_sum_forecast_and_actual_inference_date(tmp_path, monkeypatch):
    from src.models import train_returns_mlp_lab as lab
    dates = pd.bdate_range("2023-01-01", periods=270)
    returns = pd.DataFrame({"NEW": np.random.default_rng(2).normal(0, .01, len(dates))}, index=dates)
    inferred_dates = []
    class Model:
        def __init__(self, steps): pass
        def fit(self, X, y): pass
        def predict(self, X):
            if len(X) == 1:
                inferred_dates.append(X.index[-1])
            return np.full(len(X), .05)
    monkeypatch.setattr(lab, "active_data_dir", lambda: tmp_path / "snapshot")
    monkeypatch.setattr(lab, "load_returns", lambda **kwargs: returns)
    monkeypatch.setattr(lab, "Pipeline", Model)
    monkeypatch.setattr(lab.joblib, "dump", lambda *args: None)
    monkeypatch.setattr(storage, "persist_mlp_lab_run", lambda **kwargs: None)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(lab, "MODEL_ROOT", tmp_path / "models")
    result = lab.run_mlp_lab(start_date="full", target_aggregation="sum", forecast_horizon=5)
    row = result["predictions"].iloc[0]
    assert row.mu_pred == pytest.approx(.01)
    assert row.as_of == str(pd.Timestamp(inferred_dates[-1]).date())
    assert row.prediction_units == "daily_log_return"
    assert row.model_run_id == result["run_id"]
    assert "positive_precision_val" in result["metrics"]
    config = json.loads((Path(result["run_output_dir"]) / "config.json").read_text())
    assert config["selection_split"] == "validation"


@pytest.mark.parametrize("aggregation", ["mean", "sum"])
def test_targets_require_all_five_future_observations(aggregation):
    from src.models.train_returns_mlp_lab import build_target
    series = pd.Series(np.arange(12., dtype=float))
    target = build_target(series, 5, aggregation)
    assert target.iloc[-5:].isna().all()
    assert target.iloc[0] == pytest.approx(sum(range(1, 6)) if aggregation == "sum" else np.mean(range(1, 6)))
    assert target.iloc[-6] == pytest.approx(sum(range(7, 12)) if aggregation == "sum" else np.mean(range(7, 12)))


def test_temporal_split_purges_overlapping_future_targets():
    from src.models.train_returns_mlp_lab import temporal_split
    dates = pd.bdate_range("2024-01-01", periods=105)
    frame = pd.DataFrame({"value": np.arange(100)}, index=dates[:100])
    train, val, test = temporal_split(frame, forecast_horizon=5)
    def target_dates(origins):
        return {dates[dates.get_loc(origin) + offset] for origin in origins for offset in range(1, 6)}
    assert not target_dates(train.index).intersection(target_dates(val.index))
    assert not target_dates(val.index).intersection(target_dates(test.index))
    # Horizon-one callers retain the previous 70/15/15 split lengths.
    assert [len(part) for part in temporal_split(frame)] == [70, 15, 15]


def test_winsor_limits_do_not_depend_on_future_validation_or_test():
    from src.models.train_returns_mlp_lab import training_winsorization_bounds
    series = pd.Series(np.random.default_rng(4).normal(0, .01, 150),
                       index=pd.bdate_range("2024-01-01", periods=150))
    train_end = series.index[89]
    before = training_winsorization_bounds(series, train_end, .01)
    series.iloc[90:] = np.arange(60) * 1000 + 1000
    after = training_winsorization_bounds(series, train_end, .01)
    assert before == after
