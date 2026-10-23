import numpy as np

def port_stats_log(w, mu_log, Sigma_log, rf_log_daily=0.0):
    """
    Stats en escala LOG (daily).
    w: (n,)
    mu_log: (n,) expected daily log returns
    Sigma_log: (n,n) cov daily log returns
    rf_log_daily: risk-free en log daily (normalmente 0)
    """
    ret = float(np.dot(w, mu_log))
    vol = float(np.sqrt(np.dot(w.T, np.dot(Sigma_log, w))))
    sharpe = (ret - rf_log_daily) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

def annualize_log(mu_log_daily, vol_log_daily, periods=252):
    """
    Anualiza log-mean y vol de log-returns.
    """
    mu_a = mu_log_daily * periods
    vol_a = vol_log_daily * np.sqrt(periods)
    return mu_a, vol_a

def log_to_simple_annual(mu_log_annual):
    """
    Convierte retorno log anual a retorno simple equivalente:
    simple = exp(mu_log_annual) - 1
    """
    return float(np.exp(mu_log_annual) - 1.0)

def annualized_sharpe(sharpe_daily, periods=252):
    """Annualized Sharpe for daily log excess returns and daily volatility."""
    if not np.isfinite(periods) or periods <= 0:
        raise ValueError("periods must be finite and positive")
    return float(sharpe_daily * np.sqrt(periods))


def portfolio_statistics(weights, mu, cov, rf_log_daily=0.0, periods=252):
    """New metric semantics are explicit; return_annual remains the log alias."""
    daily_return, daily_vol, daily_sharpe = port_stats_log(weights, mu, cov, rf_log_daily)
    annual_return, annual_vol = annualize_log(daily_return, daily_vol, periods)
    return {
        "return_daily": daily_return, "return_log_daily": daily_return,
        "vol_daily": daily_vol, "sharpe_daily": daily_sharpe,
        "return_annual": float(annual_return), "return_log_annual": float(annual_return),
        "return_simple_annual": log_to_simple_annual(annual_return),
        "vol_annual": float(annual_vol),
        "sharpe_annual": annualized_sharpe(daily_sharpe, periods),
        "periods": periods, "metrics_version": 2, "return_type": "log",
    }
