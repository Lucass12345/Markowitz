from pathlib import Path

ROOT = Path.cwd()

def write_file(rel_path, content):
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"WROTE: {rel_path}")


# =========================
# 1. LSTM FEATURES
# =========================
write_file(
    "src/models/lstm_features.py",
    '''
import pandas as pd

AVAILABLE_FEATURES = [
    'return',
    'lag_1',
    'lag_2',
    'rolling_mean_5',
    'rolling_std_5',
]

def build_lstm_feature_frame(series: pd.Series) -> pd.DataFrame:
    s = series.dropna().astype(float)

    df = pd.DataFrame(index=s.index)
    df['return'] = s
    df['lag_1'] = s.shift(1)
    df['lag_2'] = s.shift(2)
    df['rolling_mean_5'] = s.rolling(5).mean()
    df['rolling_std_5'] = s.rolling(5).std()

    return df.dropna()
'''
)


# =========================
# 2. LSTM LOADER
# =========================
write_file(
    "src/optimization/lstm_lab_loader.py",
    '''
import pandas as pd
from sqlalchemy import create_engine
from app.db import DATABASE_URL

def load_active_lstm_predictions_df():
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT prt.ticker, prt.mu_pred
    FROM prediction_runs pr
    JOIN prediction_run_tickers prt ON pr.run_id = prt.run_id
    WHERE pr.is_active = 1
    """

    df = pd.read_sql(query, engine)
    return df
'''
)


# =========================
# 3. INPUTS BUILDER FIX
# =========================
write_file(
    "src/optimization/inputs_builder.py",
    '''
from src.optimization.lstm_lab_loader import load_active_lstm_predictions_df

def build_inputs(returns, model_type):

    if model_type == "ai_returns_lstm":
        preds = load_active_lstm_predictions_df()

        if preds.empty:
            raise ValueError("No active LSTM Lab run found.")

        preds = preds.dropna()
        preds = preds.drop_duplicates("ticker").set_index("ticker")

        common = [t for t in returns.columns if t in preds.index]

        if not common:
            raise ValueError("No matching tickers.")

        returns_sub = returns[common]
        mu = preds.loc[common, "mu_pred"]
        cov = returns_sub.cov()

        return mu, cov, common

    # fallback
    mu = returns.mean()
    cov = returns.cov()
    return mu, cov, list(returns.columns)
'''
)


# =========================
# 4. STORAGE (SIN JSONB)
# =========================
write_file(
    "app/lstm_lab_storage.py",
    '''
from sqlalchemy import create_engine, text
from app.db import DATABASE_URL

def ensure_lstm_lab_tables():
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prediction_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            created_at TEXT,
            is_active INTEGER
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prediction_run_tickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            ticker TEXT,
            mu_pred REAL
        );
        """))

        conn.commit()

    print("DB OK")
'''
)


print("\nDONE ✅")
print("Ahora corré:")
print("C:\\Users\\lucas\\anaconda3\\envs\\IIA_cuda_env\\python.exe guccimane.py")
print("y después:")
print("docker compose restart streamlit fastapi")