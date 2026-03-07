
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"

def load_returns():
    df = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    df.columns = df.columns.astype(str).str.strip()
    return df
