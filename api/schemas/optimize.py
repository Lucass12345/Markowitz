from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ModelType = Literal["classic", "ai_returns", "ai_returns_lstm", "ai_volatility"]


class OptimizeRequest(BaseModel):
    model_type: ModelType = Field(..., description="Optimization mode")
    targets: dict[str, float] = Field(..., description="Block targets")
    max_weight: float = Field(0.15, ge=0.0, le=1.0)
    rf_annual_simple: float = Field(0.0, ge=0.0, le=1.0)
    n_frontier_points: int = Field(80, ge=10, le=500)
    run_label: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_targets(self) -> "OptimizeRequest":
        expected_blocks = {"AR", "US", "FUNDS_EQUITY", "FUNDS_BONDS", "GOLD"}
        received_blocks = set(self.targets.keys())

        if received_blocks != expected_blocks:
            missing = expected_blocks - received_blocks
            extra = received_blocks - expected_blocks
            raise ValueError(
                f"Targets must contain exactly these blocks: {sorted(expected_blocks)}. "
                f"Missing={sorted(missing)} Extra={sorted(extra)}"
            )

        total = sum(float(v) for v in self.targets.values())
        if total <= 0:
            raise ValueError("Targets must sum to a positive value.")

        return self


class OptimizeResponse(BaseModel):
    run_id: int | None
    status: str
    message: str
    result: dict
    mlflow_run_id: str | None = None
    mlflow_error: str | None = None