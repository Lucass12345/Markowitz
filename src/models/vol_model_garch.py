from arch import arch_model
import numpy as np


def forecast_garch_vol(returns):

    model = arch_model(
        returns * 100,
        vol="Garch",
        p=1,
        q=1,
        dist="normal"
    )

    res = model.fit(disp="off")

    forecast = res.forecast(horizon=1)

    var = forecast.variance.values[-1][0]

    vol = np.sqrt(var) / 100

    return var / 10000, vol