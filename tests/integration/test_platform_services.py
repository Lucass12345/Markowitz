"""Opt-in, unmocked end-to-end platform validation against running services.

The CI workflow also runs the script in two phases around actual Docker process
restarts. This pytest entry point is useful against a disposable host stack.
"""
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PLATFORM_INTEGRATION") != "1",
    reason="Requires an isolated live PostgreSQL/Redis/MinIO/MLflow/API/worker stack",
)


def test_real_platform_round_trip(tmp_path):
    from scripts.platform_integration import run
    report = run(state_path=tmp_path / "state.json", timeout=int(os.getenv("INTEGRATION_JOB_TIMEOUT", "1200")))
    assert report["before_restart_verified"]
    assert report["after_restart_verified"]
    assert report["duplicate_delivery_verified"]
    assert set(report["models"]) == {"lstm", "mlp", "garch"}
    assert len(report["experiments"]) == 6
