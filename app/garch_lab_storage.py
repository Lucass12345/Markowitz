
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "outputs" / "garch_lab" / "runs"


def save_garch_run(df_metrics: pd.DataFrame, metadata: dict):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_metrics.to_csv(RUNS_DIR / f"{run_id}_metrics.csv", index=False)

    with open(RUNS_DIR / f"{run_id}_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return run_id


def load_all_runs():
    if not RUNS_DIR.exists():
        return []

    runs = []
    for f in RUNS_DIR.glob("*_meta.json"):
        with open(f) as file:
            runs.append(json.load(file))

    return runs
