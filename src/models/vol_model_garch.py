"""GJR-GARCH-t forecasts and explicit residual / parameter diagnostics."""
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def fit_gjr(returns, dist="t", mean="Zero", maxiter=300):
    s = returns.dropna().astype(float)
    if len(s) < 252 or not np.isfinite(s).all() or s.std() <= 0:
        raise ValueError("GJR-GARCH requires 252 finite, nonconstant observations")
    res = arch_model(s * 100, mean=mean, vol="GARCH", p=1, o=1, q=1,
                     dist=dist, rescale=False).fit(disp="off", options={"maxiter": maxiter})
    if res.convergence_flag != 0:
        raise ValueError(f"GARCH optimizer did not converge: {res.optimization_result.message}")
    return res


def forecast_gjr_garch_vol(returns, dist="t", mean="Zero", maxiter=300):
    res = fit_gjr(returns, dist, mean, maxiter)
    variance = float(res.forecast(horizon=1).variance.iloc[-1, 0]) / 10000
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Invalid GARCH variance")
    return variance, float(np.sqrt(variance))


def forecast_garch_vol(returns):
    return forecast_gjr_garch_vol(returns)


def diagnose_garch(returns, horizon=5, test_size=20):
    """Refit rolling origins on a held-out tail; realized r² is a noisy variance proxy.

    Forecasts at every origin use past observations only. No test tuning is done.
    """
    s = returns.dropna().astype(float)
    res = fit_gjr(s)
    z = np.asarray(res.std_resid)
    z = z[np.isfinite(z)]
    lag = min(10, len(z) // 5)
    lb = acorr_ljungbox(z, lags=[lag], return_df=True)
    lb2 = acorr_ljungbox(z ** 2, lags=[lag], return_df=True)
    arch_lm = het_arch(z, nlags=lag)
    p = res.params
    persistence = float(p.get("alpha[1]", 0) + 0.5 * p.get("gamma[1]", 0) + p.get("beta[1]", 0))
    variances = res.forecast(horizon=horizon).variance.iloc[-1].to_numpy() / 10000
    if not np.isfinite(variances).all() or (variances <= 0).any():
        raise ValueError("Invalid multi-step GARCH variance")
    out = dict(converged=True, aic=float(res.aic), bic=float(res.bic),
               loglikelihood=float(res.loglikelihood), persistence=persistence,
               covariance_stationary=bool(persistence < 1),
               ljung_box_resid_p=float(lb.lb_pvalue.iloc[0]),
               ljung_box_squared_resid_p=float(lb2.lb_pvalue.iloc[0]),
               arch_lm_p=float(arch_lm[1]), diagnostic_lags=lag,
               residual_tests_pass=bool(lb.lb_pvalue.iloc[0] > .05 and lb2.lb_pvalue.iloc[0] > .05 and arch_lm[1] > .05),
               parameters={name: dict(value=float(value), std_error=float(res.std_err[name]),
                                      p_value=float(res.pvalues[name]),
                                      ci_lower=float(res.conf_int().loc[name].iloc[0]),
                                      ci_upper=float(res.conf_int().loc[name].iloc[1])) for name, value in p.items()},
               forecast_horizon=horizon, variance_by_day=variances.tolist(),
               variance_sum_5d=float(variances.sum()), vol_5d=float(np.sqrt(variances.sum())))
    realized, predicted, baseline = [], [], []
    failures = []
    for i in range(max(252, len(s) - test_size), len(s)):
        train = s.iloc[:i]
        try:
            variance, _ = forecast_garch_vol(train)
            realized.append(float(s.iloc[i] ** 2))
            predicted.append(variance)
            baseline.append(float(train.pow(2).ewm(alpha=.06, adjust=False).mean().iloc[-1]))
        except Exception as exc:
            failures.append(dict(date=str(s.index[i]), error=str(exc)))
    def scores(pred):
        pred = np.asarray(pred)
        actual = np.asarray(realized)
        return dict(qlike=float(np.mean(np.log(pred) + actual / pred)),
                    mse=float(np.mean((actual - pred) ** 2)))
    out["backtest"] = dict(n=len(realized), failures=failures,
                           garch=scores(predicted) if realized else None,
                           ewma=scores(baseline) if realized else None,
                           proxy="squared daily log returns", horizon=1)
    return out
