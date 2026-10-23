from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]

files = {
    "src/optimization/inputs_builder.py": r'''
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "outputs" / "predictions"


def load_predictions() -> pd.DataFrame:
    path = PRED_DIR / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Primero corré:\n"
            f"1) run_train_returns.py\n"
            f"2) run_train_volatility.py\n"
            f"3) run_predict_all.py"
        )

    preds = pd.read_csv(path)
    required = {"ticker", "mu_pred", "var_pred", "vol_pred"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"predictions.csv no tiene columnas requeridas: {missing}")

    preds["ticker"] = preds["ticker"].astype(str).str.strip()
    return preds


def _build_ai_vol_cov(returns_subset: pd.DataFrame, pred_vol: pd.Series) -> pd.DataFrame:
    """
    Reescala la cov histórica usando las volatilidades predichas,
    manteniendo la estructura de correlaciones histórica.

    cov_ai = D * Corr_hist * D
    donde D = diag(vol_pred)
    """
    hist_cov = returns_subset.cov()
    hist_vol = returns_subset.std()

    hist_vol = hist_vol.replace(0, np.nan)

    corr = hist_cov.div(hist_vol, axis=0).div(hist_vol, axis=1)
    corr = corr.fillna(0.0)

    d = pd.Series(pred_vol, index=returns_subset.columns, dtype=float)
    cov_ai = corr.mul(d, axis=0).mul(d, axis=1)

    cov_ai = (cov_ai + cov_ai.T) / 2.0
    return cov_ai


def build_model_inputs(
    model_type: str,
    returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """
    Devuelve:
    - mu
    - cov
    - tickers alineados

    model_type:
    - classic
    - ai_returns
    - ai_volatility
    """
    model_type = str(model_type).strip().lower()

    returns = returns.copy()
    returns.columns = returns.columns.astype(str).str.strip()

    if model_type == "classic":
        mu = returns.mean()
        cov = returns.cov()
        tickers = list(mu.index)
        return mu, cov, tickers

    preds = load_predictions().set_index("ticker")

    common = [t for t in returns.columns if t in preds.index]
    if not common:
        raise ValueError("No hay tickers en común entre returns.csv y predictions.csv")

    returns_sub = returns[common].copy()

    if model_type == "ai_returns":
        mu = preds.loc[common, "mu_pred"].astype(float)
        cov = returns_sub.cov()
        return mu, cov, common

    if model_type == "ai_volatility":
        mu = returns_sub.mean()
        pred_vol = preds.loc[common, "vol_pred"].astype(float)
        cov = _build_ai_vol_cov(returns_sub, pred_vol)
        return mu, cov, common

    raise ValueError(
        f"model_type inválido: {model_type}. "
        f"Usá 'classic', 'ai_returns' o 'ai_volatility'."
    )
''',

    "src/optimization/frontier_engine.py": r'''
import pandas as pd

from src.markowitz_classic import (
    min_variance,
    max_sharpe,
    efficient_frontier,
)
from src.utils_portfolio import port_stats_log, annualize_log, annualized_sharpe


def _stats_dict(
    weights,
    mu,
    cov,
    rf_log_daily: float = 0.0,
    periods: int = 252,
) -> dict:
    mu_d, vol_d, sh_d = port_stats_log(
        weights,
        mu,
        cov,
        rf_log_daily=rf_log_daily
    )
    mu_a, vol_a = annualize_log(mu_d, vol_d, periods=periods)

    return {
        "return_daily": float(mu_d),
        "vol_daily": float(vol_d),
        "sharpe_daily": float(sh_d),
        "return_annual": float(mu_a),
        "vol_annual": float(vol_a),
        "sharpe_annual": annualized_sharpe(sh_d, periods),
    }


def run_frontier(
    mu,
    cov,
    rf_log_daily: float = 0.0,
    periods: int = 252,
    n_points: int = 80,
) -> dict:
    """
    Corre frontera, min var y max sharpe para el modo seleccionado.
    """
    tickers = list(mu.index) if hasattr(mu, "index") else list(range(len(mu)))

    w_min = min_variance(mu, cov)
    w_max = max_sharpe(mu, cov, risk_free_rate=rf_log_daily)

    stats_min = _stats_dict(w_min, mu, cov, rf_log_daily=rf_log_daily, periods=periods)
    stats_max = _stats_dict(w_max, mu, cov, rf_log_daily=rf_log_daily, periods=periods)

    frontier_pts = efficient_frontier(mu, cov, n_points=n_points)

    frontier_df = pd.DataFrame(
        frontier_pts,
        columns=["vol_daily", "return_daily"]
    )

    annualized = [
        annualize_log(r, v, periods=periods)
        for v, r in frontier_pts
    ]
    frontier_df["return_annual"] = [x[0] for x in annualized]
    frontier_df["vol_annual"] = [x[1] for x in annualized]

    w_min_df = pd.DataFrame({"ticker": tickers, "weight": w_min}).sort_values(
        "weight", ascending=False
    ).reset_index(drop=True)

    w_max_df = pd.DataFrame({"ticker": tickers, "weight": w_max}).sort_values(
        "weight", ascending=False
    ).reset_index(drop=True)

    return {
        "frontier": frontier_df,
        "minvar": {
            "weights_array": w_min,
            "weights_df": w_min_df,
            "stats": stats_min,
        },
        "maxsharpe": {
            "weights_array": w_max,
            "weights_df": w_max_df,
            "stats": stats_max,
        },
    }
''',

    "src/optimization/optimize_blocks.py": r'''
import numpy as np
import pandas as pd

from src.markowitz import max_sharpe_with_blocks
from src.utils_portfolio import port_stats_log, annualize_log, annualized_sharpe


def build_blocks_from_meta(tickers: list[str], meta: pd.DataFrame, targets: dict) -> dict:
    meta2 = meta.copy()
    meta2["ticker"] = meta2["ticker"].astype(str).str.strip()
    meta2 = meta2.set_index("ticker").loc[tickers]

    blocks = {}
    for block_name in targets.keys():
        idx = np.where(meta2["block"].values == block_name)[0]
        blocks[block_name] = list(idx)

    return blocks


def compute_block_exposures(weights: pd.Series, meta: pd.DataFrame, targets: dict) -> pd.DataFrame:
    meta2 = meta.copy()
    meta2["ticker"] = meta2["ticker"].astype(str).str.strip()
    meta2 = meta2.set_index("ticker").loc[weights.index]

    rows = []
    for block_name, target in targets.items():
        exposure = float(weights.loc[meta2.index[meta2["block"] == block_name]].sum())
        rows.append({
            "block": block_name,
            "target": float(target),
            "achieved": exposure,
            "diff": exposure - float(target),
        })

    return pd.DataFrame(rows)


def run_blocks_optimizer(
    mu,
    cov,
    tickers: list[str],
    meta: pd.DataFrame,
    targets: dict,
    max_weight: float = 0.15,
    rf_log_daily: float = 0.0,
    periods: int = 252,
) -> dict:
    blocks = build_blocks_from_meta(tickers, meta, targets)

    res = max_sharpe_with_blocks(
        mean_returns=mu,
        cov_matrix=cov,
        blocks=blocks,
        block_targets=targets,
        risk_free_rate=rf_log_daily,
        max_weight=max_weight,
    )

    if not res.success:
        raise RuntimeError(f"max_sharpe_with_blocks failed: {res.message}")

    w = pd.Series(res.x, index=tickers, name="weight")
    w_df = w.sort_values(ascending=False).reset_index()
    w_df.columns = ["ticker", "weight"]

    meta2 = meta.copy()
    meta2["ticker"] = meta2["ticker"].astype(str).str.strip()
    meta2 = meta2.set_index("ticker").loc[tickers].reset_index()

    w_df = w_df.merge(meta2, on="ticker", how="left")

    exposures_df = compute_block_exposures(w, meta, targets)

    mu_d, vol_d, sh_d = port_stats_log(
        w.values,
        mu,
        cov,
        rf_log_daily=rf_log_daily
    )
    mu_a, vol_a = annualize_log(mu_d, vol_d, periods=periods)

    stats = {
        "return_daily": float(mu_d),
        "vol_daily": float(vol_d),
        "sharpe_daily": float(sh_d),
        "return_annual": float(mu_a),
        "vol_annual": float(vol_a),
        "sharpe_annual": annualized_sharpe(sh_d, periods),
    }

    return {
        "weights_array": w.values,
        "weights_series": w,
        "weights_df": w_df,
        "stats": stats,
        "exposure_by_block": exposures_df,
        "solver_message": res.message,
    }
''',

    "src/optimization/optimizer_service.py": r'''
from pathlib import Path
import pandas as pd

from src.config import TRADING_DAYS
from src.markowitz import rf_simple_annual_to_log_daily
from src.data.load_data import load_returns
from src.optimization.inputs_builder import build_model_inputs
from src.optimization.frontier_engine import run_frontier
from src.optimization.optimize_blocks import run_blocks_optimizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"


def load_metadata() -> pd.DataFrame:
    path = DATA_DIR / "assets_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    meta = pd.read_csv(path)
    meta["ticker"] = meta["ticker"].astype(str).str.strip()
    return meta


def optimize_portfolio_mode(
    model_type: str,
    targets: dict,
    max_weight: float = 0.15,
    rf_annual_simple: float = 0.0,
    periods: int = TRADING_DAYS,
    n_frontier_points: int = 80,
) -> dict:
    """
    Orquestador principal para los 3 modos:
    - classic
    - ai_returns
    - ai_volatility
    """
    returns = load_returns().dropna(how="any")
    meta = load_metadata()

    rf_log_daily = rf_simple_annual_to_log_daily(rf_annual_simple)

    mu, cov, tickers = build_model_inputs(
        model_type=model_type,
        returns=returns,
    )

    if hasattr(mu, "index") and hasattr(cov, "loc"):
        cov = cov.loc[mu.index, mu.index]
        tickers = list(mu.index)

    frontier_result = run_frontier(
        mu=mu,
        cov=cov,
        rf_log_daily=rf_log_daily,
        periods=periods,
        n_points=n_frontier_points,
    )

    blocks_result = run_blocks_optimizer(
        mu=mu,
        cov=cov,
        tickers=tickers,
        meta=meta,
        targets=targets,
        max_weight=max_weight,
        rf_log_daily=rf_log_daily,
        periods=periods,
    )

    return {
        "model_type": model_type,
        "inputs": {
            "targets": targets,
            "max_weight": max_weight,
            "rf_annual_simple": rf_annual_simple,
            "rf_log_daily": rf_log_daily,
            "periods": periods,
        },
        "frontier": frontier_result["frontier"],
        "minvar": frontier_result["minvar"],
        "maxsharpe": frontier_result["maxsharpe"],
        "blocks": blocks_result,
    }
''',

    "scripts/run_mode_classic.py": r'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DEFAULT_CONFIG
from src.optimization.optimizer_service import optimize_portfolio_mode

if __name__ == "__main__":
    result = optimize_portfolio_mode(
        model_type="classic",
        targets=DEFAULT_CONFIG.targets,
        max_weight=DEFAULT_CONFIG.max_weight_blocks,
        rf_annual_simple=DEFAULT_CONFIG.risk_free_annual_simple,
    )

    print("\n=== CLASSIC ===")
    print("MinVar:", result["minvar"]["stats"])
    print("MaxSharpe:", result["maxsharpe"]["stats"])
    print("Blocks:", result["blocks"]["stats"])
    print("\nTop weights blocks:")
    print(result["blocks"]["weights_df"].head(10))
''',

    "scripts/run_mode_ai_returns.py": r'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DEFAULT_CONFIG
from src.optimization.optimizer_service import optimize_portfolio_mode

if __name__ == "__main__":
    result = optimize_portfolio_mode(
        model_type="ai_returns",
        targets=DEFAULT_CONFIG.targets,
        max_weight=DEFAULT_CONFIG.max_weight_blocks,
        rf_annual_simple=DEFAULT_CONFIG.risk_free_annual_simple,
    )

    print("\n=== AI RETURNS ===")
    print("MinVar:", result["minvar"]["stats"])
    print("MaxSharpe:", result["maxsharpe"]["stats"])
    print("Blocks:", result["blocks"]["stats"])
    print("\nTop weights blocks:")
    print(result["blocks"]["weights_df"].head(10))
''',

    "scripts/run_mode_ai_volatility.py": r'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DEFAULT_CONFIG
from src.optimization.optimizer_service import optimize_portfolio_mode

if __name__ == "__main__":
    result = optimize_portfolio_mode(
        model_type="ai_volatility",
        targets=DEFAULT_CONFIG.targets,
        max_weight=DEFAULT_CONFIG.max_weight_blocks,
        rf_annual_simple=DEFAULT_CONFIG.risk_free_annual_simple,
    )

    print("\n=== AI VOLATILITY ===")
    print("MinVar:", result["minvar"]["stats"])
    print("MaxSharpe:", result["maxsharpe"]["stats"])
    print("Blocks:", result["blocks"]["stats"])
    print("\nTop weights blocks:")
    print(result["blocks"]["weights_df"].head(10))
''',
}


def main():
    for relative_path, content in files.items():
        path = ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        print(f"created: {relative_path}")

    print("\nAll optimizer core files created successfully.")


if __name__ == "__main__":
    main()