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