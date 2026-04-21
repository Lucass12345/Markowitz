import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "outputs" / "lstm_lab" / "runs"
LATEST_PTR = ROOT / "outputs" / "lstm_lab" / "latest_run.json"


def _compute_amp_row(csv_path: Path) -> dict | None:
    ticker = csv_path.name.replace("_predictions.csv", "")
    df = pd.read_csv(csv_path)
    if "split" not in df.columns or "y_true" not in df.columns or "y_pred" not in df.columns:
        return None
    test_df = df.loc[df["split"].astype(str).str.lower() == "test", ["y_true", "y_pred"]].copy()
    if test_df.empty:
        return None
    std_y_true = float(pd.to_numeric(test_df["y_true"], errors="coerce").std())
    std_y_pred = float(pd.to_numeric(test_df["y_pred"], errors="coerce").std())
    amp_ratio = (std_y_pred / std_y_true) if pd.notna(std_y_true) and std_y_true > 0 else float("nan")
    return {
        "ticker": ticker,
        "n_test": int(len(test_df)),
        "std_y_true_test": std_y_true,
        "std_y_pred_test": std_y_pred,
        "amp_ratio": amp_ratio,
        "mean_y_true_test": float(pd.to_numeric(test_df["y_true"], errors="coerce").mean()),
        "mean_y_pred_test": float(pd.to_numeric(test_df["y_pred"], errors="coerce").mean()),
    }


def _update_run(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics.csv"
    per_ticker_dir = run_dir / "per_ticker"
    summary_path = run_dir / "summary.json"
    if not metrics_path.exists() or not per_ticker_dir.exists():
        return

    metrics_df = pd.read_csv(metrics_path)
    amp_rows = []
    for csv_path in sorted(per_ticker_dir.glob("*_predictions.csv")):
        row = _compute_amp_row(csv_path)
        if row is not None:
            amp_rows.append(row)

    if not amp_rows:
        return

    amp_df = pd.DataFrame(amp_rows)
    merged = metrics_df.merge(amp_df, on="ticker", how="left")
    merged.to_csv(metrics_path, index=False)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if "amp_ratio" in merged.columns:
            s = pd.to_numeric(merged["amp_ratio"], errors="coerce")
            summary["mean_amp_ratio_test"] = float(s.mean()) if s.notna().any() else None
            summary["share_amp_ratio_gt_015"] = float((s > 0.15).mean()) if s.notna().any() else None
        summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[ok] {run_dir.name}: metrics.csv actualizado con amp_ratio.")


def _resolve_run_dirs(run_id: str | None, all_runs: bool) -> list[Path]:
    if all_runs:
        return [d for d in sorted(RUNS_DIR.glob("*")) if d.is_dir()]

    if run_id:
        only = RUNS_DIR / run_id
        if not only.exists():
            raise FileNotFoundError(f"No existe corrida {run_id} en {RUNS_DIR}")
        return [only]

    if LATEST_PTR.exists():
        latest = json.loads(LATEST_PTR.read_text())
        latest_id = str(latest.get("run_id", "")).strip()
        if latest_id:
            only = RUNS_DIR / latest_id
            if only.exists():
                return [only]

    raise FileNotFoundError(
        "No pude resolver la corrida objetivo. Usá --run-id <RUN_ID> o --all."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill de amp_ratio para corridas LSTM.")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID específico a actualizar.")
    parser.add_argument("--all", action="store_true", help="Actualizar todas las corridas en outputs/lstm_lab/runs.")
    args = parser.parse_args()

    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"No existe {RUNS_DIR}")

    for run_dir in _resolve_run_dirs(run_id=args.run_id, all_runs=args.all):
        _update_run(run_dir)
