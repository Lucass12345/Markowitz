import pandas as pd
from pathlib import Path

from src.markowitz_classic import max_sharpe
from src.utils_portfolio import port_stats_log

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "Data"
PRED_DIR = ROOT / "outputs" / "predictions"
OUT_DIR = ROOT / "outputs" / "portfolios"


def optimize_with_predictions():

    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0)

    preds = pd.read_csv(PRED_DIR / "predictions.csv")

    tickers = preds["ticker"].values

    returns = returns[tickers]

    mu_pred = preds["mu_pred"].values

    cov_matrix = returns.cov()

    weights = max_sharpe(mu_pred, cov_matrix)

    stats = port_stats_log(weights, mu_pred, cov_matrix)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "ticker": tickers,
        "weight": weights
    }).to_csv(OUT_DIR / "portfolio_ai_returns.csv", index=False)

    print("\nAI portfolio stats")
    print(stats)

    return weights