"""Versioned Yahoo datasets. Publish only after every requested asset succeeds."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import uuid
import time

import numpy as np
import pandas as pd

from src.data.load_data import DATA_DIR
from src.data.currency import convert_prices_to_usd


def classify_asset(ticker, info):
    quote_type = info.get("quoteType", "UNKNOWN")
    category = info.get("category") or ""
    country = info.get("country")  # Listing exchange is not the country of exposure.
    sector = info.get("sector")
    asset_type = {"EQUITY": "equity", "ETF": "etf", "MUTUALFUND": "mutual_fund",
                  "BOND": "bond", "INDEX": "index", "CURRENCY": "currency",
                  "CRYPTOCURRENCY": "cryptocurrency", "FUTURE": "future"}.get(quote_type, "unknown")
    bond_fund = quote_type in {"ETF", "MUTUALFUND"} and any(
        word in category.lower() for word in ("bond", "treasury", "fixed income", "municipal", "money market", "government", "corporate", "high yield", "inflation-protected", "bank loan", "ultrashort", "securitized"))
    if bond_fund:
        asset_type = "bond_fund"
    block = "OTHER"
    if bond_fund:
        block = "FUNDS_BONDS"
    elif "gold" in category.lower():
        block = "GOLD"
    elif quote_type in {"ETF", "MUTUALFUND"} and category and not any(
            word in category.lower() for word in ("commodit", "currency", "digital asset", "managed futures")):
        block = "FUNDS_EQUITY"
    elif quote_type == "EQUITY":
        block = {"Argentina": "AR", "United States": "US"}.get(country, "OTHER")
    return dict(ticker=ticker, name=info.get("longName") or info.get("shortName"),
                asset_class=asset_type, asset_type=asset_type, quote_type=quote_type,
                country=country, sector=sector, category=category, block=block,
                exchange=info.get("exchange"), currency=info.get("currency"),
                metadata_source="yahoo_finance", classification_basis="quoteType/category", country_status="reported" if country else "unknown",
                sector_status="reported" if sector else "unknown")


def refresh_dataset(tickers, start="2015-01-01", end=None, data_dir=DATA_DIR, provider=None,
                    *, ccl_file=None, ccl_rates=None, max_fx_age_days=7):
    """Adjusted daily closes -> log returns; no forward filling across market holidays.

    end is exclusive. A full refresh also incorporates Yahoo historical revisions.
    Missing country/sector is explicitly unknown, never inferred from domicile.
    """
    if provider is None:
        import yfinance as yf
        provider = yf.Ticker
    symbols = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    if not symbols or any(not re.fullmatch(r"[A-Z0-9^=._-]+", t) for t in symbols):
        raise ValueError("Provide valid Yahoo Finance tickers.")
    prices, metadata = {}, []
    for ticker in symbols:
        asset = provider(ticker)
        for attempt in range(3):
            try:
                hist = asset.history(start=start, end=end, interval="1d", auto_adjust=True, raise_errors=True)
                info = asset.info
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"{ticker}: Yahoo download failed after three attempts") from exc
                time.sleep(2 ** attempt)
        if hist.empty or "Close" not in hist:
            raise ValueError(f"{ticker}: Yahoo returned no price history; dataset not published.")
        close = hist["Close"].astype(float).sort_index()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        close = close[~close.index.duplicated(keep="last")]
        # Exclude the current UTC date so an unfinished daily candle cannot be a label.
        close = close.loc[close.index < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]
        if len(close) < 2 or not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError(f"{ticker}: invalid prices.")
        prices[ticker] = close
        metadata.append(classify_asset(ticker, info))
    native_frame = pd.DataFrame(prices).sort_index()
    prices, metadata, fx_audit = convert_prices_to_usd(
        prices, metadata, provider, ccl_rates=ccl_rates, ccl_file=ccl_file, max_fx_age_days=max_fx_age_days)
    frame = pd.DataFrame(prices).sort_index()
    # Compute per asset BEFORE alignment: different exchanges have different sessions.
    returns = pd.DataFrame({t: np.log(s / s.shift(1)) for t, s in prices.items()}).dropna(how="all")
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    data_dir = Path(data_dir)
    folder = data_dir / "dynamic" / version
    folder.mkdir(parents=True)
    native_frame.to_csv(folder / "prices_native.csv", index_label="date")
    frame.to_csv(folder / "prices.csv", index_label="date")
    fx_audit.to_csv(folder / "fx_rates.csv", index=False)
    returns.to_csv(folder / "returns.csv", index_label="date")
    pd.DataFrame(metadata).to_csv(folder / "assets_metadata.csv", index=False)
    manifest = dict(version=version, tickers=symbols, start=start, end=end,
                    updated_at=datetime.now(timezone.utc).isoformat(), return_type="log", base_currency="USD",
                    currency_policy={"ARS": "CCL vendedor, ARS por USD; precio ARS / CCL",
                                     "other": "Yahoo USD por moneda de cotización; multiplicación",
                                     "alignment": "same_or_previous_date", "max_fx_age_days": max_fx_age_days},
                    current_utc_session_excluded=True,
                    latest_observation={t: str(s.index[-1].date()) for t, s in prices.items()})
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary = data_dir / f".active_dataset_{version}.json"
    temporary.write_text(json.dumps({"version": version}), encoding="utf-8")
    temporary.replace(data_dir / "active_dataset.json")
    return manifest
