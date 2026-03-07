import sys
from pathlib import Path

# Add project root to python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.markowitz_classic import max_sharpe
from src.utils_portfolio import port_stats_log


DATA_DIR = ROOT / "Data"
PRED_DIR = ROOT / "outputs" / "predictions"
FIG_DIR = ROOT / "outputs" / "figures"


def compare_historical_vs_ai():

    # =========================
    # LOAD DATA
    # =========================

    returns = pd.read_csv(DATA_DIR / "returns.csv", index_col=0)

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # =========================
    # HISTORICAL MARKOWITZ
    # =========================

    w_hist = max_sharpe(mean_returns.values, cov_matrix)

    stats_hist = port_stats_log(
        w_hist,
        mean_returns.values,
        cov_matrix
    )

    ret_hist, vol_hist, sharpe_hist = stats_hist

    # =========================
    # AI MARKOWITZ
    # =========================

    preds = pd.read_csv(PRED_DIR / "predictions.csv")

    tickers = preds["ticker"].values

    returns_ai = returns[tickers]

    mu_pred = preds["mu_pred"].values

    cov_matrix_ai = returns_ai.cov()

    w_ai = max_sharpe(mu_pred, cov_matrix_ai)

    stats_ai = port_stats_log(
        w_ai,
        mu_pred,
        cov_matrix_ai
    )

    ret_ai, vol_ai, sharpe_ai = stats_ai

    # =========================
    # PRINT RESULTS
    # =========================

    print("\nHistorical portfolio")
    print(f"Return: {ret_hist:.6f}")
    print(f"Volatility: {vol_hist:.6f}")
    print(f"Sharpe: {sharpe_hist:.6f}")

    print("\nAI portfolio")
    print(f"Return: {ret_ai:.6f}")
    print(f"Volatility: {vol_ai:.6f}")
    print(f"Sharpe: {sharpe_ai:.6f}")

    # =========================
    # PLOT
    # =========================

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(
        vol_hist,
        ret_hist,
        label="Historical",
        s=150
    )

    ax.scatter(
        vol_ai,
        ret_ai,
        label="AI predicted",
        s=150
    )

    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.set_title("Historical vs AI Portfolio")

    ax.legend()
    ax.grid(True)

    plt.savefig(FIG_DIR / "compare_historical_vs_ai.png")
    plt.show()


if __name__ == "__main__":

    compare_historical_vs_ai()