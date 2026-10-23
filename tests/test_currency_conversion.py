"""Economic and chronology checks for a common USD return denomination."""
import numpy as np
import pandas as pd
import pytest

from src.data.currency import (
    align_fx_rates,
    convert_prices_to_usd,
    normalize_currency,
)


DATES = pd.to_datetime(["2024-01-05", "2024-01-08"])


class FakeFXProvider:
    """Yahoo-shaped deterministic FX provider; unexpected requests fail locally."""

    def __init__(self, pairs=None):
        self.pairs = pairs or {}
        self.requested = []

    def __call__(self, ticker):
        self.requested.append(ticker)
        if ticker not in self.pairs:
            raise ValueError(f"Unsupported FX ticker: {ticker}")
        prices = self.pairs[ticker]

        class Quote:
            def history(self, **kwargs):
                assert kwargs.get("auto_adjust") is True
                return prices.rename("Close").to_frame()

        return Quote()


def prices(values, dates=DATES):
    return pd.Series(values, index=dates, dtype=float)


def rates(values, dates=DATES):
    return pd.DataFrame({"rate": values, "source": "test_CCL"}, index=dates)


def test_ars_returns_include_ccl_depreciation():
    provider = FakeFXProvider()
    converted, metadata, audit = convert_prices_to_usd(
        {"LOCAL.BA": prices([10000, 11000])},
        [{"ticker": "LOCAL.BA", "currency": "ARS", "country": "Argentina"}],
        provider,
        ccl_rates=prices([1000, 1200]),
    )
    np.testing.assert_allclose(converted["LOCAL.BA"], [10., 11000 / 1200])
    usd_return = np.log(converted["LOCAL.BA"].iloc[1] / converted["LOCAL.BA"].iloc[0])
    assert usd_return == pytest.approx(np.log(1.1 / 1.2))
    assert usd_return < 0  # A peso gain need not be a dollar gain.
    assert provider.requested == []  # No silent use of Yahoo's official ARS/USD.
    assert len(metadata) == 1 and not audit.empty


def test_argentine_adr_already_in_usd_is_unchanged():
    provider = FakeFXProvider()
    original = prices([10., 11.])
    converted, _, _ = convert_prices_to_usd(
        {"GGAL": original},
        [{"ticker": "GGAL", "currency": "USD", "country": "Argentina"}],
        provider,
    )
    np.testing.assert_array_equal(converted["GGAL"].values, original.values)
    assert provider.requested == []


def test_euro_prices_multiply_by_usd_per_euro():
    provider = FakeFXProvider({"EURUSD=X": prices([1.1, 1.2])})
    converted, _, audit = convert_prices_to_usd(
        {"EUROPE": prices([100., 110.])},
        [{"ticker": "EUROPE", "currency": "EUR"}],
        provider,
    )
    np.testing.assert_allclose(converted["EUROPE"], [110., 132.])
    assert provider.requested == ["EURUSD=X"]
    assert not audit.empty


def test_pence_are_scaled_before_foreign_exchange_conversion():
    provider = FakeFXProvider({"GBPUSD=X": prices([1.25, 1.30])})
    converted, _, _ = convert_prices_to_usd(
        {"LONDON": prices([10000., 11000.])},
        [{"ticker": "LONDON", "currency": "GBp"}],
        provider,
    )
    np.testing.assert_allclose(converted["LONDON"], [125., 143.])
    assert provider.requested == ["GBPUSD=X"]


@pytest.mark.parametrize("currency, expected", [
    ("USD", ("USD", 1.)),
    ("usd", ("USD", 1.)),
    ("GBP", ("GBP", 1.)),
    ("GBp", ("GBP", .01)),
    ("GBX", ("GBP", .01)),
    ("ILA", ("ILS", .01)),
    ("ZAc", ("ZAR", .01)),
])
def test_currency_units_preserve_case_meaning(currency, expected):
    assert normalize_currency(currency) == expected


def test_fx_uses_only_same_or_prior_date_with_bounded_weekend_carry():
    requested = pd.to_datetime(["2024-01-06", "2024-01-12"])
    aligned = align_fx_rates(rates([1000.], DATES[:1]), requested, max_age_days=7)
    np.testing.assert_allclose(aligned["rate"], [1000., 1000.])
    assert aligned["source"].tolist() == ["test_CCL", "test_CCL"]
    assert aligned["age_days"].tolist() == [1, 7]
    assert pd.to_datetime(aligned["rate_date"]).eq(pd.Timestamp("2024-01-05")).all()
    pd.testing.assert_index_equal(aligned.index, requested, check_names=False)


def test_future_fx_is_never_backfilled():
    with pytest.raises(ValueError):
        align_fx_rates(rates([1200.], DATES[1:]), DATES[:1])


def test_future_rate_does_not_replace_an_earlier_available_rate():
    aligned = align_fx_rates(rates([1000., 1200.]), pd.to_datetime(["2024-01-06"]))
    assert aligned.iloc[0]["rate"] == 1000.
    assert pd.Timestamp(aligned.iloc[0]["rate_date"]) == pd.Timestamp("2024-01-05")


def test_stale_ccl_cannot_extend_forever():
    with pytest.raises(ValueError):
        align_fx_rates(rates([1000.], DATES[:1]), pd.to_datetime(["2024-01-13"]), max_age_days=7)


@pytest.mark.parametrize("invalid_rate", [0., -1., np.nan, np.inf])
def test_invalid_fx_rates_rejected(invalid_rate):
    with pytest.raises(ValueError):
        align_fx_rates(rates([invalid_rate], DATES[:1]), DATES[:1])


@pytest.mark.parametrize("currency", [None, "", np.nan])
def test_missing_quote_currency_is_not_assumed_to_be_usd(currency):
    with pytest.raises(ValueError):
        convert_prices_to_usd(
            {"UNKNOWN": prices([10., 11.])},
            [{"ticker": "UNKNOWN", "currency": currency}],
            FakeFXProvider(),
        )


def test_unsupported_currency_fails_instead_of_retaining_local_prices():
    with pytest.raises(ValueError):
        convert_prices_to_usd(
            {"UNKNOWN": prices([10., 11.])},
            [{"ticker": "UNKNOWN", "currency": "NOT_A_CURRENCY"}],
            FakeFXProvider(),
        )
