"""Synthetic feasibility checks before numerical optimization."""
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from api.schemas.optimize import OptimizeRequest
from src import markowitz
from src.optimization.errors import OptimizationInputError, ConstraintFeasibilityError
from src.optimization.optimize_blocks import build_blocks_from_meta, validate_optimizer_inputs

TARGETS = {"AR": .2, "US": .5, "FUNDS_EQUITY": .1, "FUNDS_BONDS": .1, "GOLD": .1}


@pytest.mark.parametrize("targets", [
    {name: value * 2 for name, value in TARGETS.items()},
    {**TARGETS, "AR": -.1, "US": .8},
    {**TARGETS, "AR": np.nan}, {**TARGETS, "AR": np.inf},
])
def test_schema_rejects_invalid_targets(targets):
    with pytest.raises(ValidationError):
        OptimizeRequest(model_type="classic", targets=targets)


def test_api_keeps_exact_five_blocks_and_disallows_zero_cap():
    with pytest.raises(ValidationError):
        OptimizeRequest(model_type="classic", targets={"OTHER": 1.})
    with pytest.raises(ValidationError):
        OptimizeRequest(model_type="classic", targets=TARGETS, max_weight=0.)
    assert OptimizeRequest(model_type="classic", targets=TARGETS).targets == TARGETS


@pytest.mark.parametrize("targets", [{"A": np.nan}, {"A": np.inf}, {"A": -1., "B": 2.}, {"A": .5}])
def test_core_validates_targets_too(targets):
    with pytest.raises(ConstraintFeasibilityError):
        markowitz.min_variance_with_blocks(np.eye(2), {"A": [0], "B": [1]}, targets)


@pytest.mark.parametrize("blocks,targets,n,expected", [
    ({"A": [0], "B": [1]}, {"A": .5, "B": .5}, 2, [.5, .5]),
    ({"A": [0, 1, 2], "B": []}, {"A": 1., "B": 0.}, 3, [1/3] * 3),
    ({"A": [0]}, {"A": 1.}, 1, [1.]),
])
def test_feasible_partitions_and_zero_empty_blocks_solve(blocks, targets, n, expected):
    result = markowitz.min_variance_with_blocks(np.eye(n), blocks, targets)
    assert result.success, result.message
    np.testing.assert_allclose(result.x, expected, atol=1e-7)


def test_budget_kept_when_other_assets_are_unassigned():
    result = markowitz.min_variance_with_blocks(np.diag([1., .001]), {"A": [0]}, {"A": 1.})
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [1., 0.], atol=1e-7)


@pytest.mark.parametrize("blocks,targets,n,cap", [
    ({"A": []}, {"A": 1.}, 2, 1.),
    ({"A": [0], "B": [1, 2]}, {"A": .6, "B": .4}, 3, .5),
    ({"A": [0, 1]}, {"A": 1.}, 2, .4),
    ({"A": [0, 0]}, {"A": 1.}, 2, 1.),
    ({"A": [0], "B": [0]}, {"A": .5, "B": .5}, 2, 1.),
    ({"A": [-1]}, {"A": 1.}, 2, 1.),
    ({"A": [2]}, {"A": 1.}, 2, 1.),
    ({"A": [0.]}, {"A": 1.}, 2, 1.),
    ({"A": [0]}, {"A": 1.}, 1, np.nan),
])
def test_infeasible_or_invalid_blocks_never_reach_slsqp(monkeypatch, blocks, targets, n, cap):
    monkeypatch.setattr(markowitz, "minimize", lambda *a, **k: pytest.fail("Invalid problem reached SLSQP"))
    with pytest.raises(ConstraintFeasibilityError):
        markowitz.min_variance_with_blocks(np.eye(n), blocks, targets, cap)


@pytest.mark.parametrize("mu,cov", [
    (np.array([]), np.empty((0, 0))),
    (np.array([[1., 2.]]), np.eye(2)),
    (np.array([np.nan, 1.]), np.eye(2)),
    (np.array([1., 2.]), np.ones((2, 3))),
    (np.array([1., 2.]), np.array([[1., np.inf], [np.inf, 1.]])),
    (pd.Series([1., 2.], index=["A", "A"]), np.eye(2)),
    (pd.Series([1., 2.], index=["A", "B"]), pd.DataFrame(np.eye(2), index=["A", "C"], columns=["A", "C"])),
    (np.array([1., 2.]), pd.DataFrame(np.eye(2), index=["A", "A"], columns=["A", "A"])),
])
def test_runtime_dimensions_finiteness_and_labels_validated(mu, cov):
    with pytest.raises(OptimizationInputError):
        markowitz.max_sharpe_with_blocks(mu, cov, {"X": [0, 1]}, {"X": 1.})


def test_covariance_labels_are_aligned_without_changing_asset_meaning():
    mu = pd.Series([.01, .02], index=["A", "B"])
    cov = pd.DataFrame(np.diag([1., 4.]), index=["A", "B"], columns=["A", "B"])
    permuted = cov.loc[["B", "A"], ["A", "B"]]
    result = markowitz.portfolio_performance([1., 0.], mu, permuted)
    assert result == pytest.approx((.01, 1.))


@pytest.mark.parametrize("tickers,meta", [
    (["A", "A"], pd.DataFrame({"ticker": ["A"], "block": ["X"]})),
    (["A"], pd.DataFrame({"ticker": ["A", "A"], "block": ["X", "X"]})),
    (["A"], pd.DataFrame({"ticker": ["B"], "block": ["X"]})),
    (["A"], pd.DataFrame({"ticker": ["A"], "block": [None]})),
])
def test_duplicate_or_missing_metadata_rejected(tickers, meta):
    with pytest.raises(OptimizationInputError):
        build_blocks_from_meta(tickers, meta, {"X": 1.})


def test_capacity_checked_after_model_filtering_before_frontier(monkeypatch):
    from src.optimization import optimizer_service as service
    returns = pd.DataFrame({"A": [.01, .02], "B": [.02, .03]})
    meta = pd.DataFrame({"ticker": ["A", "B"], "block": ["X", "X"]})
    monkeypatch.setattr(service, "load_returns", lambda **kwargs: returns)
    monkeypatch.setattr(service, "load_metadata", lambda **kwargs: meta)
    monkeypatch.setattr(service, "build_model_inputs", lambda **kwargs: (pd.Series([.01], index=["A"]), pd.DataFrame([[.01]], index=["A"], columns=["A"]), ["A"]))
    monkeypatch.setattr(service, "run_frontier", lambda **kwargs: pytest.fail("Infeasible request reached frontier"))
    with pytest.raises(ConstraintFeasibilityError):
        service.optimize_portfolio_mode("classic", {"X": 1.}, max_weight=.6)
