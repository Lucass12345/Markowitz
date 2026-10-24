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
    identity = (info.get("longName") or info.get("shortName") or "").lower()
    verified_gold = ticker.upper() == "GLD" and quote_type == "ETF" and "spdr" in identity and "gold" in identity
    if verified_gold:
        # Yahoo currently calls GLD "Commodities Focused". Preserve the original
        # verified SPDR Gold Shares allocation instead of inferring OTHER.
        block = "GOLD"
    return dict(ticker=ticker, name=info.get("longName") or info.get("shortName"),
                asset_class=asset_type, asset_type=asset_type, quote_type=quote_type,
                country=country, sector=sector, category=category, block=block,
                exchange=info.get("exchange"), currency=info.get("currency"),
                metadata_source="yahoo_finance", classification_basis="verified_SPDR_Gold_Shares" if verified_gold else "quoteType/category", country_status="reported" if country else "unknown",
                sector_status="reported" if sector else "unknown")


def _daily_close(history, minimum=2):
    if history.empty or "Close" not in history:
        raise ValueError("Yahoo returned no daily price history")
    close = history["Close"].astype(float).sort_index()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")]
    close = close.loc[close.index < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]
    if len(close) < minimum or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Yahoo returned invalid daily prices")
    return close


def _download(asset, ticker, start, end, check_cancelled=None, include_info=True):
    for attempt in range(3):
        if check_cancelled:
            check_cancelled()
        try:
            history = asset.history(start=start, end=end, interval="1d", auto_adjust=True, raise_errors=True)
            info = asset.info if include_info else None
            return history, info
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"{ticker}: Yahoo download failed after three attempts") from exc
            time.sleep(2 ** attempt)


def _corporate_actions(history, ticker):
    rows = []
    for column in ("Dividends", "Stock Splits", "Capital Gains"):
        if column in history:
            for date, value in history[column].items():
                if pd.notna(value) and value != 0:
                    rows.append({"ticker": ticker, "date": pd.Timestamp(date).tz_localize(None).normalize(),
                                 "action": column, "value": float(value)})
    return pd.DataFrame(rows, columns=["ticker", "date", "action", "value"])


def _action_signature(actions):
    return sorted((pd.Timestamp(row.date).isoformat(), row.action, float(row.value))
                  for row in actions.itertuples())


def refresh_dataset(tickers, start="2015-01-01", end=None, data_dir=DATA_DIR, provider=None,
                    *, activate=True, ccl_file=None, ccl_rates=None, max_fx_age_days=7,
                    previous_dir=None, refresh_mode="full", overlap_days=30, block_overrides=None,
                    check_cancelled=None, on_progress=None):
    """Adjusted daily closes -> log returns; no forward filling across market holidays.

    end is exclusive. A full refresh also incorporates Yahoo historical revisions.
    Missing country/sector is explicitly unknown, never inferred from domicile.
    """
    if activate:
        from src.publication.releases import active_release
        if active_release(data_dir) is not None:
            raise ValueError("Hay un release activo: usá publicación de candidato con reentrenamiento para actualizar datos y modelos juntos.")
    if provider is None:
        import yfinance as yf
        provider = yf.Ticker
    if isinstance(tickers, str) or any(not isinstance(ticker, str) for ticker in tickers):
        raise ValueError("Provide valid Yahoo Finance tickers.")
    symbols = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not symbols or any(not re.fullmatch(r"[A-Z0-9^=._-]+", t) for t in symbols):
        raise ValueError("Provide valid Yahoo Finance tickers.")
    if refresh_mode not in {"full", "incremental"} or not isinstance(overlap_days, int) or overlap_days < 1:
        raise ValueError("Invalid refresh mode or overlap window")
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end is not None else None
    if pd.isna(start_date) or (end_date is not None and (pd.isna(end_date) or end_date <= start_date)):
        raise ValueError("Invalid requested date range")
    overrides = block_overrides or {}
    if set(overrides) - set(symbols) or any(not isinstance(v, str) or not v.strip() for v in overrides.values()):
        raise ValueError("Block overrides require selected tickers and nonempty block names")
    old_prices, old_metadata, previous_manifest = None, None, {}
    previous = Path(previous_dir) if previous_dir else None
    prior_actions = pd.DataFrame(columns=["ticker", "date", "action", "value"])
    if refresh_mode == "incremental" and previous is not None and (previous / "prices_native.csv").exists():
        previous_manifest = json.loads((previous / "manifest.json").read_text())
        if previous_manifest.get("base_currency") != "USD" or previous_manifest.get("return_type") != "log":
            raise ValueError("Incremental parent must contain verified USD log returns")
        old_prices = pd.read_csv(previous / "prices_native.csv", index_col=0, parse_dates=True)
        old_metadata = pd.read_csv(previous / "assets_metadata.csv").set_index("ticker")
        if old_prices.index.has_duplicates or not old_prices.index.is_monotonic_increasing or old_metadata.index.has_duplicates:
            raise ValueError("Incremental parent has duplicate or unsorted observations")
        if (previous / "corporate_actions.csv").exists():
            prior_actions = pd.read_csv(previous / "corporate_actions.csv", parse_dates=["date"])
    refresh_details, action_frames = {}, []
    prices, metadata = {}, []
    for index, ticker in enumerate(symbols):
        if check_cancelled:
            check_cancelled()
        if on_progress:
            on_progress(index / (len(symbols) + 1), "dataset.download", f"Downloading {ticker}")
        asset = provider(ticker)
        old = old_prices[ticker].dropna() if old_prices is not None and ticker in old_prices else None
        if old is not None:
            old = old.loc[old.index >= start_date]
            if end_date is not None:
                old = old.loc[old.index < end_date]
        requested_start = start
        extends_history = bool(old is not None and len(old) and start_date < pd.Timestamp(previous_manifest.get("start", old.index.min())))
        if old is not None and len(old) and not extends_history:
            requested_start = max(pd.Timestamp(start), old.index[-1] - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
        hist, info = _download(asset, ticker, requested_start, end, check_cancelled)
        close = _daily_close(hist, minimum=1 if old is not None and len(old) else 2)
        downloaded_actions = _corporate_actions(hist, ticker)
        details = {"requested_start": requested_start, "mode": "full", "revision_detected": False}
        if old is not None and len(old) and not extends_history:
            common = old.index.intersection(close.index)
            changed = bool(len(common) and not np.allclose(old.loc[common], close.loc[common], rtol=1e-7, atol=1e-10))
            old_overlap = old.loc[(old.index >= close.index.min()) & (old.index <= close.index.max())]
            missing_observations = bool(len(old_overlap.index.difference(close.index)))
            currency_changed = old_metadata is None or ticker not in old_metadata.index or old_metadata.loc[ticker].get("currency") != info.get("currency")
            old_actions = prior_actions.loc[prior_actions.ticker.eq(ticker)]
            action_start = pd.Timestamp(requested_start)
            action_end = min(end_date, close.index.max() + pd.Timedelta(days=1)) if end_date is not None else close.index.max() + pd.Timedelta(days=1)
            action_overlap = old_actions.loc[(old_actions.date >= action_start) & (old_actions.date < action_end)]
            action_revision = _action_signature(action_overlap) != _action_signature(downloaded_actions)
            if changed or missing_observations or currency_changed or action_revision or not len(common):
                # Splits/dividends alter adjusted prices before the event. An
                # overlap revision requires a full asset refresh, not a splice.
                hist, _ = _download(asset, ticker, start, end, check_cancelled, include_info=False)
                close = _daily_close(hist)
                downloaded_actions = _corporate_actions(hist, ticker)
                details.update(mode="full_revision", revision_detected=changed,
                               currency_changed=bool(currency_changed), corporate_actions_revised=action_revision,
                               missing_overlap_observations=missing_observations,
                               requested_start=start)
            else:
                close = pd.concat([old.loc[old.index < close.index.min()], close]).sort_index()
                details["mode"] = "incremental"
                # Replace the downloaded interval, including actions that Yahoo
                # retracted. Historical rows outside the interval stay auditable.
                action_frames.append(old_actions.loc[(old_actions.date < action_start) | (old_actions.date >= action_end)])
        elif extends_history:
            details["mode"] = "full_history_extension"
        close = close.loc[close.index >= pd.Timestamp(start)]
        if end is not None:
            close = close.loc[close.index < pd.Timestamp(end)]
        if len(close) < 2:
            raise ValueError(f"{ticker}: insufficient prices in requested date range")
        action_frames.append(downloaded_actions)
        prices[ticker] = close
        classified = classify_asset(ticker, info)
        if ticker in overrides:
            classified.update(block=overrides[ticker].strip(), block_source="universe_override")
        else:
            classified["block_source"] = classified["classification_basis"]
        metadata.append(classified)
        refresh_details[ticker] = details
    native_frame = pd.DataFrame(prices).sort_index()
    if check_cancelled:
        check_cancelled()
    if on_progress:
        on_progress(len(symbols) / (len(symbols) + 1), "dataset.currency", "Converting observed prices to USD")
    prices, metadata, fx_audit = convert_prices_to_usd(
        prices, metadata, provider, ccl_rates=ccl_rates, ccl_file=ccl_file, max_fx_age_days=max_fx_age_days)
    frame = pd.DataFrame(prices).sort_index()
    # Compute per asset BEFORE alignment: different exchanges have different sessions.
    returns = pd.DataFrame({t: np.log(s / s.shift(1)) for t, s in prices.items()}).dropna(how="all")
    if check_cancelled:
        check_cancelled()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    data_dir = Path(data_dir)
    folder = data_dir / "dynamic" / version
    folder.mkdir(parents=True)
    native_frame.to_csv(folder / "prices_native.csv", index_label="date")
    frame.to_csv(folder / "prices.csv", index_label="date")
    fx_audit.to_csv(folder / "fx_rates.csv", index=False)
    actions = pd.concat(action_frames, ignore_index=True)
    if not actions.empty:
        actions = actions.loc[actions.ticker.isin(symbols) & (actions.date >= start_date)]
        if end_date is not None:
            actions = actions.loc[actions.date < end_date]
        actions = actions.drop_duplicates(["ticker", "date", "action"], keep="last").sort_values(["ticker", "date", "action"])
    actions.to_csv(folder / "corporate_actions.csv", index=False)
    returns.to_csv(folder / "returns.csv", index_label="date")
    pd.DataFrame(metadata).to_csv(folder / "assets_metadata.csv", index=False)
    manifest = dict(version=version, tickers=symbols, start=start, end=end,
                    updated_at=datetime.now(timezone.utc).isoformat(), return_type="log", base_currency="USD",
                    currency_policy={"ARS": "CCL vendedor, ARS por USD; precio ARS / CCL",
                                     "other": "Yahoo USD por moneda de cotización; multiplicación",
                                     "alignment": "same_or_previous_date", "max_fx_age_days": max_fx_age_days},
                    current_utc_session_excluded=True,
                    refresh_mode=refresh_mode, refresh_details=refresh_details,
                    parent_version=previous_manifest.get("version", previous.name if previous else None),
                    overlap_days=overlap_days if refresh_mode == "incremental" else None,
                    incremental_revision_policy="full_asset_refresh_when_adjusted_overlap_changes; periodic_full_refresh_required_for_older_revisions",
                    latest_observation={t: str(s.index[-1].date()) for t, s in prices.items()})
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if check_cancelled:
        check_cancelled()
    if on_progress:
        on_progress(1., "dataset.download.complete", "USD prices, returns and currency audit prepared")
    if not activate:
        return manifest
    temporary = data_dir / f".active_dataset_{version}.json"
    temporary.write_text(json.dumps({"version": version}), encoding="utf-8")
    temporary.replace(data_dir / "active_dataset.json")
    return manifest
