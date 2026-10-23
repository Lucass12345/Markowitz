import json
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.yahoo_dataset import refresh_dataset, classify_asset
from src.models.sequence_utils import build_lstm_sequences_multifeature
from src.models.train_returns_lstm_lab import temporal_split, scale_3d_train, run_lstm_lab
from src.models.return_model_lstm import ReturnLSTM
from src.models.lstm_features import build_lstm_feature_frame, AVAILABLE_FEATURES
from src.models.lstm_search import default_candidates


def test_target_is_exactly_next_five_sessions():
    s = pd.Series(np.arange(30, dtype=float))
    X, mean = build_lstm_sequences_multifeature(s.to_frame("return"), s, seq_len=4)
    _, total = build_lstm_sequences_multifeature(s.to_frame("return"), s, seq_len=4, target_aggregation="sum")
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3])
    assert mean[0] == np.mean([4, 5, 6, 7, 8])
    assert total[-1] == sum(range(25, 30))


def test_purged_split_has_disjoint_target_intervals():
    X = np.arange(100)[:, None, None]
    y = np.arange(100)
    parts = temporal_split(X, y, y, forecast_horizon=5)
    assert parts[1][-1] + 4 < parts[4][0]
    assert parts[4][-1] + 4 < parts[7][0]


def test_scaling_uses_training_only():
    train = np.array([1., 3.]).reshape(2, 1, 1)
    _, val, _, mean, std = scale_3d_train(train, np.full((1, 1, 1), 100.), train)
    assert mean[0] == 2 and std[0] == 1 and val.item() == 98


def test_features_do_not_use_future():
    s = pd.Series(np.random.default_rng(1).normal(size=100))
    before = build_lstm_feature_frame(s)
    s.iloc[70:] = 999
    after = build_lstm_feature_frame(s)
    pd.testing.assert_frame_equal(before.iloc[:70], after.iloc[:70])
    assert set(AVAILABLE_FEATURES) <= set(after.columns)


@pytest.mark.parametrize("sizes", [[64, 64], [120, 64]])
@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu"])
def test_architectures(sizes, activation):
    model = ReturnLSTM(input_size=3, hidden_sizes=sizes, activation=activation)
    result = model(torch.zeros(2, 10, 3))
    assert result.shape == (2,) and torch.isfinite(result).all()


class FakeTicker:
    info = {"quoteType": "EQUITY", "country": "Argentina", "sector": "Energy", "currency": "USD"}

    def __init__(self, ticker):
        self.ticker = ticker

    def history(self, **kwargs):
        if self.ticker == "BAD":
            return pd.DataFrame()
        dates = pd.date_range("2024-01-01", periods=5, tz="America/New_York")
        if self.ticker == "SHORT":
            dates = dates.delete(2)
        return pd.DataFrame({"Close": np.arange(len(dates)) + 100.}, index=dates)


def test_dataset_atomic_activation_metadata_and_market_gaps(tmp_path):
    manifest = refresh_dataset(["NEW", "SHORT"], data_dir=tmp_path, provider=FakeTicker)
    version = manifest["version"]
    directory = tmp_path / "dynamic" / version
    returns = pd.read_csv(directory / "returns.csv", index_col=0)
    assert list(returns) == ["NEW", "SHORT"]
    assert np.isclose(returns.loc["2024-01-04", "SHORT"], np.log(102 / 101))
    assert pd.isna(returns.loc["2024-01-03", "SHORT"])
    meta = pd.read_csv(directory / "assets_metadata.csv")
    assert meta.iloc[0].country == "Argentina" and meta.iloc[0].sector == "Energy"
    with pytest.raises(ValueError):
        refresh_dataset(["BAD"], data_dir=tmp_path, provider=FakeTicker)
    assert json.loads((tmp_path / "active_dataset.json").read_text())["version"] == version


def test_fund_metadata_does_not_invent_country():
    result = classify_asset("BND", {"quoteType": "ETF", "category": "Intermediate Core Bond"})
    assert result["asset_type"] == "bond_fund" and result["country"] is None
    assert classify_asset("TLT", {"quoteType": "ETF", "category": "Long Government"})["block"] == "FUNDS_BONDS"


def test_latest_forecast_uses_latest_data_and_saves_reproducible_model(tmp_path, monkeypatch):
    import src.models.train_returns_lstm_lab as lab
    dates = pd.bdate_range("2023-01-01", periods=260)
    returns = pd.DataFrame({"TEST": np.random.default_rng(2).normal(0, .01, len(dates))}, index=dates)
    monkeypatch.setattr(lab, "load_returns", lambda **kwargs: returns)
    monkeypatch.setattr(lab, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(lab, "MODEL_ROOT", tmp_path / "models")
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    result = run_lstm_lab(["return"], seq_len=20, start_date="full", epochs=2,
                          candidates=[default_candidates()[0]], persist_db=False)
    assert len(result["predictions"]) == 1, result["metrics"].to_dict()
    pred = result["predictions"].iloc[0]
    assert pred.as_of == str(dates[-1].date())
    meta = json.loads((__import__('pathlib').Path(result["model_dir"]) / "TEST_meta.json").read_text())
    model = ReturnLSTM(input_size=1, hidden_sizes=meta["selected_config"]["hidden_sizes"],
                       activation=meta["selected_config"]["activation"], dropout=meta["dropout"])
    model.load_state_dict(torch.load(__import__('pathlib').Path(result["model_dir"]) / "TEST_lstm.pt", weights_only=True))
    model.eval()
    latest = returns.iloc[-20:].values.astype(np.float32)
    latest = (latest - np.array(meta["x_mean"])) / np.array(meta["x_std"])
    with torch.no_grad():
        expected = model(torch.tensor(latest[None], dtype=torch.float32)).item() * meta["y_std"] + meta["y_mean"]
    assert pred.mu_pred == pytest.approx(expected, abs=1e-8)
    assert pred.forecast_log_return_5d == pytest.approx(pred.mu_pred * 5)


def test_garch_diagnostics_and_scale():
    from src.models.vol_model_garch import diagnose_garch
    s = pd.Series(np.random.default_rng(42).standard_t(8, size=320) * .01)
    result = diagnose_garch(s, test_size=3)
    assert result["converged"] and len(result["variance_by_day"]) == 5
    assert 0 < result["variance_by_day"][0] < .01
    assert result["backtest"]["n"] == 3
    assert result["variance_sum_5d"] == pytest.approx(sum(result["variance_by_day"]))
    assert all(0 <= result[k] <= 1 for k in ["arch_lm_p", "ljung_box_resid_p", "ljung_box_squared_resid_p"])


def test_feature_subset_is_saved_in_model_config():
    from src.models.lstm_search import fit_candidates
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 5, 3)).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    config = dict(default_candidates()[0], feature_indices=[0])
    result = fit_candidates(x[:20], y[:20], x[20:], y[20:], [config], epochs=1)
    assert result["model"].layers[0].input_size == 1
    assert result["model"](torch.tensor(x[20:])).shape == (10,)


def test_stale_predictions_rejected(tmp_path, monkeypatch):
    from src.data import load_data
    monkeypatch.setattr(load_data, "DATA_DIR", tmp_path)
    (tmp_path / "active_dataset.json").write_text(json.dumps({"version": "new"}))
    folder = tmp_path / "dynamic" / "new"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"base_currency": "USD"}))
    with pytest.raises(ValueError, match="dataset activo"):
        load_data.assert_current_dataset(pd.DataFrame({"dataset_version": ["old"]}))
    load_data.assert_current_dataset(pd.DataFrame({"dataset_version": ["new"]}))


def test_automated_predictions_work_without_database_and_use_validation_gate(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from src.optimization import lstm_lab_loader as loader
    from src.data.load_data import active_data_dir
    def offline():
        raise ConnectionError("Database unavailable")
    monkeypatch.setitem(sys.modules, "app.lstm_lab_storage", SimpleNamespace(get_active_prediction_run=offline))
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    output = tmp_path / "outputs" / "predictions"
    output.mkdir(parents=True)
    run = tmp_path / "run"
    (run / "per_ticker").mkdir(parents=True)
    pd.DataFrame({"ticker": ["NEW"], "mu_pred": [.001], "dir_acc_val": [.6],
                  "dataset_version": [active_data_dir().name]}).to_csv(output / "return_predictions_lstm.csv", index=False)
    (output / "lstm_production.json").write_text(json.dumps({"run_output_dir": str(run)}))
    pd.DataFrame({"split": ["val", "val", "test", "test"],
                  "y_true": [0., 2., 0., 2.], "y_pred": [0., 1., 0., 100.]}).to_csv(run / "per_ticker" / "NEW_predictions.csv", index=False)
    result = loader.load_active_lstm_predictions_df()
    assert result.iloc[0].amp_ratio == pytest.approx(.5)
    assert result.iloc[0].dir_acc_val == .6
