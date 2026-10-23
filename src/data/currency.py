"""Price conversion to USD before return calculation.

ARS uses CCL (ARS per USD); other currencies use Yahoo's USD-per-currency pair.
Rates are matched backward in time with a bounded age. No future backfilling.
"""
from pathlib import Path
import re
import requests

import numpy as np
import pandas as pd

CCL_URL = "https://api.argentinadatos.com/v1/cotizaciones/dolares/contadoconliqui"
LOCAL_CCL = Path(__file__).resolve().parents[2] / "Data" / "fx_CCL.csv"


def normalize_currency(currency):
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("Yahoo no informó la moneda de cotización; no se puede calcular un retorno en USD.")
    code = currency.strip()
    units = {"GBp": ("GBP", .01), "GBX": ("GBP", .01),
             "ILA": ("ILS", .01), "ZAc": ("ZAR", .01)}
    if code in units:
        return units[code]
    code = code.upper()
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise ValueError(f"Moneda de cotización no admitida: {currency}")
    return code, 1.0


def _rate_frame(rates, source="provided_ccl"):
    if isinstance(rates, pd.Series):
        frame = rates.rename("rate").to_frame()
    else:
        frame = rates.copy()
    if "rate" not in frame:
        raise ValueError("La serie de cambio debe contener 'rate'.")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("La serie de cambio tiene fechas duplicadas.")
    if frame.empty or not np.isfinite(frame["rate"].astype(float)).all() or (frame["rate"].astype(float) <= 0).any():
        raise ValueError("Las cotizaciones de cambio deben ser positivas, finitas y no vacías.")
    frame["rate"] = frame["rate"].astype(float)
    if "source" not in frame:
        frame["source"] = source
    return frame[["rate", "source"]]


def read_ccl_file(path):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    date_col = "fecha" if "fecha" in frame else "date"
    rate_col = "cierre" if "cierre" in frame else "rate"
    if date_col not in frame or rate_col not in frame:
        raise ValueError("El CSV CCL requiere fecha,cierre o date,rate; valores ARS por USD.")
    return _rate_frame(frame.set_index(date_col)[rate_col], f"ccl_file:{Path(path).name}")


def load_ccl_rates(ccl_file=None):
    if ccl_file is not None:
        # An explicit file is a reproducible choice, with no hidden remote override.
        return read_ccl_file(ccl_file)
    local = read_ccl_file(LOCAL_CCL) if LOCAL_CCL.exists() else None
    try:
        response = requests.get(CCL_URL, timeout=30)
        response.raise_for_status()
        records = response.json()
        remote = pd.DataFrame(records)
        remote = _rate_frame(remote.set_index("fecha")["venta"], f"{CCL_URL}#venta")
    except Exception as exc:
        if local is None:
            raise ValueError("No se pudo descargar el CCL y no hay historial local.") from exc
        # Conversion still checks every date; a stale local file cannot be carried forever.
        return local
    if local is None:
        return remote
    # Remote vendor series takes precedence; the file only extends earlier history.
    return pd.concat([local.loc[local.index < remote.index.min()], remote]).sort_index()


def align_fx_rates(rates, dates, max_age_days=7):
    if max_age_days < 0:
        raise ValueError("La antigüedad máxima del cambio no puede ser negativa.")
    frame = _rate_frame(rates)
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize()
    positions = frame.index.get_indexer(dates, method="pad")
    if (positions < 0).any():
        missing = dates[positions < 0][0].date()
        raise ValueError(f"Falta tipo de cambio previo o del día {missing}; no se usan datos futuros.")
    aligned = frame.iloc[positions].copy()
    rate_dates = frame.index[positions]
    aligned.index = dates
    aligned["rate_date"] = rate_dates
    aligned["age_days"] = (dates - rate_dates).days
    if (aligned["age_days"] > max_age_days).any():
        invalid = aligned.loc[aligned.age_days > max_age_days].iloc[0]
        raise ValueError(f"Tipo de cambio desactualizado para {invalid.name.date()}: última cotización {invalid.rate_date.date()} (máximo {max_age_days} días).")
    return aligned


def convert_prices_to_usd(prices, metadata, provider, *, ccl_rates=None, ccl_file=None, max_fx_age_days=7):
    if max_fx_age_days < 0:
        raise ValueError("La antigüedad máxima del cambio no puede ser negativa.")
    converted, result_metadata, audit = {}, [], []
    currency_rates = {}
    for row in metadata:
        ticker = row["ticker"]
        code, scale = normalize_currency(row.get("currency"))
        s = prices[ticker].astype(float)
        if code == "USD":
            converted[ticker] = s * scale
            method, pair, sources = "already_usd", None, "quote_currency_USD"
        else:
            if code not in currency_rates:
                if code == "ARS":
                    rates = _rate_frame(ccl_rates) if ccl_rates is not None else load_ccl_rates(ccl_file)
                    pair = "CCL:ARS/USD"
                else:
                    pair = f"{code}USD=X"
                    earliest = min(series.index.min() for series in prices.values()) - pd.Timedelta(days=max_fx_age_days)
                    latest = max(series.index.max() for series in prices.values()) + pd.Timedelta(days=1)
                    history = provider(pair).history(start=earliest.strftime("%Y-%m-%d"), end=latest.strftime("%Y-%m-%d"),
                                                     interval="1d", auto_adjust=True, raise_errors=True)
                    if history.empty or "Close" not in history:
                        raise ValueError(f"Yahoo no devolvió el cambio {pair}; no se puede convertir {ticker} a USD.")
                    rates = _rate_frame(history["Close"], f"yahoo_finance:{pair}")
                currency_rates[code] = (rates, pair)
            rates, pair = currency_rates[code]
            aligned = align_fx_rates(rates, s.index, max_fx_age_days)
            if code == "ARS":
                factor = 1 / aligned["rate"]
                method = "divide_by_ccl_ars_per_usd"
            else:
                factor = aligned["rate"]
                method = "multiply_by_yahoo_usd_per_currency"
            converted[ticker] = s * scale * factor
            sources = ";".join(sorted(aligned.source.astype(str).unique()))
            detail = aligned.copy()
            detail["ticker"] = ticker
            detail["quote_currency"] = code
            detail["usd_per_currency"] = factor
            detail["unit_scale"] = scale
            detail.index.name = "date"
            audit.append(detail.reset_index())
        result_metadata.append({**row, "original_currency": row.get("currency"), "quote_currency": code,
                                "currency_unit_scale": scale, "base_currency": "USD",
                                "fx_method": method, "fx_pair": pair, "fx_source": sources})
    audit_frame = pd.concat(audit, ignore_index=True) if audit else pd.DataFrame(
        columns=["date", "rate", "source", "rate_date", "age_days", "ticker", "quote_currency", "usd_per_currency", "unit_scale"])
    return converted, result_metadata, audit_frame
