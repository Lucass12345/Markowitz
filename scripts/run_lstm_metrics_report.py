import sys
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

PRED_DIR = ROOT / "outputs" / "predictions"

if __name__ == "__main__":
    df = pd.read_csv(PRED_DIR / "return_predictions_lstm.csv")

    print("\n=== TOP 10 BEST BY RMSE ===")
    print(df.sort_values("rmse_test").head(10))

    print("\n=== TOP 10 WORST BY RMSE ===")
    print(df.sort_values("rmse_test", ascending=False).head(10))

    print("\n=== GLOBAL MEANS ===")
    print(df[["mse_test", "rmse_test", "mae_test", "best_val_loss"]].mean())

    summary_path = PRED_DIR / "lstm_metrics_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        print("\n=== SUMMARY JSON ===")
        print(summary)
