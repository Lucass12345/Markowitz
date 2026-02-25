from pathlib import Path
import pandas as pd
import numpy as np

from markowitz import min_variance, max_sharpe, portfolio_performance

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"

returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)

mean_returns = returns.mean()
cov_matrix = returns.cov()

# Min Variance
w_minvar = min_variance(mean_returns, cov_matrix)
ret_min, vol_min = portfolio_performance(w_minvar, mean_returns, cov_matrix)

print("\nMIN VAR PORTFOLIO")
print("Return:", ret_min)
print("Volatility:", vol_min)
print("Sum weights:", w_minvar.sum())

# Max Sharpe
w_sharpe = max_sharpe(mean_returns, cov_matrix)
ret_sh, vol_sh = portfolio_performance(w_sharpe, mean_returns, cov_matrix)

print("\nMAX SHARPE PORTFOLIO")
print("Return:", ret_sh)
print("Volatility:", vol_sh)
print("Sum weights:", w_sharpe.sum())

weights_minvar = pd.Series(w_minvar, index=mean_returns.index).sort_values(ascending=False)
weights_sharpe = pd.Series(w_sharpe, index=mean_returns.index).sort_values(ascending=False)

print("\nTOP 10 WEIGHTS - MIN VAR")
print(weights_minvar.head(10))
print("\nBOTTOM 10 WEIGHTS - MIN VAR")
print(weights_minvar.tail(10))

print("\nTOP 10 WEIGHTS - MAX SHARPE")
print(weights_sharpe.head(10))
print("\nBOTTOM 10 WEIGHTS - MAX SHARPE")
print(weights_sharpe.tail(10))
