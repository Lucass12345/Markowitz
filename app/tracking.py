from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from infrastructure.storage.minio_client import upload_text


def track_run_mlflow(run_db_id: int, run_label: str | None, config: dict, result: dict) -> tuple[str | None, str | None]:
    """
    Track one optimization run in MLflow.

    Returns:
        (mlflow_run_id, error_message)
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "markowitz-streamlit")

    if not tracking_uri:
        return None, "MLFLOW_TRACKING_URI not configured"

    try:
        import mlflow
    except Exception as exc:
        return None, f"mlflow import failed: {exc}"

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        run_name = run_label or f"{config['mode']}-run-{run_db_id}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "source": "fastapi",
                    "db_run_id": str(run_db_id),
                    "model_type": str(config["mode"]),
                }
            )

            # -------------------------
            # Params
            # -------------------------
            mlflow.log_params(
                {
                    "model_type": str(config["mode"]),
                    "max_weight": float(config["max_weight"]),
                    "rf_annual_simple": float(config["rf_annual_simple"]),
                    "n_frontier_points": int(config["n_frontier_points"]),
                }
            )

            for block, target in config["targets"].items():
                mlflow.log_param(f"target_{block}", float(target))

            # -------------------------
            # Metrics
            # -------------------------
            block_stats = result["blocks"]["stats"]
            minvar_stats = result["minvar"]["stats"]
            maxsharpe_stats = result["maxsharpe"]["stats"]

            mlflow.log_metrics(
                {
                    "blocks_return_annual": float(block_stats["return_annual"]),
                    "blocks_vol_annual": float(block_stats["vol_annual"]),
                    "blocks_sharpe_daily": float(block_stats["sharpe_daily"]),
                    "blocks_return_log_annual": float(block_stats["return_annual"]),
                    "blocks_sharpe_annual": float(block_stats["sharpe_annual"]),
                    "minvar_return_annual": float(minvar_stats["return_annual"]),
                    "minvar_vol_annual": float(minvar_stats["vol_annual"]),
                    "minvar_sharpe_annual": float(minvar_stats["sharpe_annual"]),
                    "maxsharpe_return_annual": float(maxsharpe_stats["return_annual"]),
                    "maxsharpe_vol_annual": float(maxsharpe_stats["vol_annual"]),
                    "maxsharpe_sharpe_annual": float(maxsharpe_stats["sharpe_annual"]),
                }
            )

            # -------------------------
            # Local temp artifacts
            # -------------------------
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                frontier_path = tmp_path / "frontier.csv"
                weights_path = tmp_path / "weights_blocks.csv"
                exposure_path = tmp_path / "block_exposure.csv"
                comparison_path = tmp_path / "comparison.csv"

                frontier_df = pd.DataFrame(result["frontier"])
                weights_df = pd.DataFrame(result["blocks"]["weights_df"])
                exposure_df = pd.DataFrame(result["blocks"]["exposure_by_block"])
                comparison_df = pd.DataFrame(
                    [
                        {
                            "portfolio": "Min Variance",
                            **result["minvar"]["stats"],
                        },
                        {
                            "portfolio": "Max Sharpe",
                            **result["maxsharpe"]["stats"],
                        },
                        {
                            "portfolio": "Max Sharpe + Blocks",
                            **result["blocks"]["stats"],
                        },
                    ]
                )

                frontier_df.to_csv(frontier_path, index=False)
                weights_df.to_csv(weights_path, index=False)
                exposure_df.to_csv(exposure_path, index=False)
                comparison_df.to_csv(comparison_path, index=False)

                # MLflow artifacts
                mlflow.log_artifact(str(frontier_path), artifact_path="tables")
                mlflow.log_artifact(str(weights_path), artifact_path="tables")
                mlflow.log_artifact(str(exposure_path), artifact_path="tables")
                mlflow.log_artifact(str(comparison_path), artifact_path="tables")

                # MinIO artifacts
                bucket = "optimization-runs"
                prefix = f"run_{run_db_id}"

                minio_frontier = upload_text(
                    bucket_name=bucket,
                    object_name=f"{prefix}/frontier.csv",
                    text=frontier_df.to_csv(index=False),
                    content_type="text/csv",
                )
                minio_weights = upload_text(
                    bucket_name=bucket,
                    object_name=f"{prefix}/weights_blocks.csv",
                    text=weights_df.to_csv(index=False),
                    content_type="text/csv",
                )
                minio_exposure = upload_text(
                    bucket_name=bucket,
                    object_name=f"{prefix}/block_exposure.csv",
                    text=exposure_df.to_csv(index=False),
                    content_type="text/csv",
                )
                minio_comparison = upload_text(
                    bucket_name=bucket,
                    object_name=f"{prefix}/comparison.csv",
                    text=comparison_df.to_csv(index=False),
                    content_type="text/csv",
                )

                mlflow.set_tags(
                    {
                        "minio_frontier_path": minio_frontier,
                        "minio_weights_path": minio_weights,
                        "minio_exposure_path": minio_exposure,
                        "minio_comparison_path": minio_comparison,
                    }
                )

            return run.info.run_id, None

    except Exception as exc:
        return None, f"mlflow tracking failed: {exc}"
