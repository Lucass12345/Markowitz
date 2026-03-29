from __future__ import annotations

import warnings
from arch import arch_model
import numpy as np


def forecast_gjr_garch_vol(
    returns,
    dist: str = "t",
    mean: str = "Zero",
    maxiter: int = 300,
):
    """
    One-step-ahead variance / volatility forecast using GJR-GARCH(1,1,1).

    Parameters
    ----------
    returns : pd.Series
        Daily returns in original scale (e.g. 0.01 = 1%)
    dist : str
        Innovation distribution. Recommended: "t"
    mean : str
        Mean specification. Recommended: "Zero"
    maxiter : int
        Max iterations for optimizer

    Returns
    -------
    tuple[float, float]
        (var_pred, vol_pred) in original return scale
    """
    s = returns.dropna().astype(float)

    if len(s) < 252:
        raise ValueError("GJR-GARCH requires at least 252 observations.")

    model = arch_model(
        s * 100.0,
        mean=mean,
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist=dist,
        rescale=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off", options={"maxiter": maxiter})

    forecast = res.forecast(horizon=1)
    var_pct2 = float(forecast.variance.values[-1][0])

    # Back to original scale
    var_pred = var_pct2 / 10000.0
    vol_pred = float(np.sqrt(var_pred))

    return var_pred, vol_pred


def forecast_garch_vol(returns):
    """
    Backward-compatible alias used by the rest of the project.
    Internally upgraded to GJR-GARCH with Student-t innovations.
    """
    return forecast_gjr_garch_vol(
        returns=returns,
        dist="t",
        mean="Zero",
        maxiter=300,
    )
