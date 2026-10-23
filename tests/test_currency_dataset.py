"""Publishing/training must see USD prices and returns, with local prices auditable."""
import json
import numpy as np
import pandas as pd
import pytest
from src.data.yahoo_dataset import refresh_dataset
from src.data import load_data


class MixedAssets:
    def __init__(self, ticker):
        self.ticker = ticker
        self.info = {"quoteType": "EQUITY", "currency": "ARS" if ticker == "LOCAL.BA" else "USD",
                     "country": "Argentina", "sector": "Financial Services"}

    def history(self, **kwargs):
        closes = [10000., 11000.] if self.ticker == "LOCAL.BA" else [10., 11.]
        return pd.DataFrame({"Close": closes}, index=pd.to_datetime(["2024-01-05", "2024-01-08"]))


def test_usd_dataset_contains_correct_prices_returns_and_audit(tmp_path, monkeypatch):
    fx = pd.Series([1000., 1200.], index=pd.to_datetime(["2024-01-05", "2024-01-08"]))
    manifest = refresh_dataset(["LOCAL.BA", "GGAL"], data_dir=tmp_path, provider=MixedAssets, ccl_rates=fx)
    folder = tmp_path / "dynamic" / manifest["version"]
    assert manifest["base_currency"] == "USD"
    monkeypatch.setattr(load_data, "DATA_DIR", tmp_path)
    result = load_data.load_returns()
    assert result["LOCAL.BA"].iloc[0] == pytest.approx(np.log(1.1 / 1.2))
    assert result["GGAL"].iloc[0] == pytest.approx(np.log(1.1))
    prices = load_data.load_prices()
    assert prices["LOCAL.BA"].iloc[-1] == pytest.approx(11000 / 1200)
    native = pd.read_csv(folder / "prices_native.csv")
    assert native["LOCAL.BA"].iloc[-1] == 11000
    audit = pd.read_csv(folder / "fx_rates.csv")
    assert audit.ticker.unique().tolist() == ["LOCAL.BA"]
    assert audit.rate.tolist() == [1000., 1200.]
    metadata = load_data.load_metadata().set_index("ticker")
    assert metadata.loc["LOCAL.BA", "original_currency"] == "ARS"
    assert metadata.loc["LOCAL.BA", "base_currency"] == "USD"
    assert metadata.loc["GGAL", "fx_method"] == "already_usd"


def test_stale_currency_does_not_publish_over_valid_dataset(tmp_path):
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    first = refresh_dataset(["GGAL"], data_dir=tmp_path, provider=MixedAssets)
    stale = pd.Series([1000.], index=pd.to_datetime(["2023-12-01"]))
    with pytest.raises(ValueError, match="desactualizado"):
        refresh_dataset(["LOCAL.BA"], data_dir=tmp_path, provider=MixedAssets, ccl_rates=stale)
    assert json.loads((tmp_path / "active_dataset.json").read_text())["version"] == first["version"]


def test_old_mixed_currency_snapshot_cannot_feed_models(tmp_path, monkeypatch):
    folder = tmp_path / "dynamic" / "old"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"return_type": "log"}))
    (tmp_path / "active_dataset.json").write_text(json.dumps({"version": "old"}))
    monkeypatch.setattr(load_data, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="USD"):
        load_data.load_returns()
    with pytest.raises(ValueError, match="USD"):
        load_data.load_prices()


def test_legacy_dataset_rejects_predictions_from_ignored_dynamic_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(load_data, "DATA_DIR", tmp_path / "Data")
    with pytest.raises(ValueError, match="dataset activo"):
        load_data.assert_current_dataset(pd.DataFrame({"ticker": ["NEW"], "dataset_version": ["usd-v1"], "base_currency": ["USD"]}))


def test_legacy_dataset_accepts_its_own_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(load_data, "DATA_DIR", tmp_path / "Data")
    load_data.assert_current_dataset(pd.DataFrame({"ticker": ["NEW"], "dataset_version": ["Data"], "base_currency": ["USD"]}))
