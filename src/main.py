import pandas as pd
import numpy as np
from markowitz import min_variance, max_sharpe, portfolio_performance

# cargar datos
returns = pd.read_csv("../Data/returns.csv", index_col=0, parse_dates=True)

mean_returns = returns.mean()
cov_matrix = returns.cov()

# Min Variance
w_minvar = min_variance(mean_returns, cov_matrix)
ret_min, vol_min = portfolio_performance(w_minvar, mean_returns, cov_matrix)

print("\nMIN VAR PORTFOLIO")
print("Return:", ret_min)
print("Volatility:", vol_min)

# Max Sharpe
w_sharpe = max_sharpe(mean_returns, cov_matrix)
ret_sh, vol_sh = portfolio_performance(w_sharpe, mean_returns, cov_matrix)

print("\nMAX SHARPE PORTFOLIO")
print("Return:", ret_sh)
print("Volatility:", vol_sh)
