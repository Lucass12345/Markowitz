import numpy as np
import pandas as pd
import pytest
from src.optimization.frontier_engine import run_constrained_frontier, run_frontier, _clean_frontier, check_frontier_weights


def test_every_constrained_point_respects_all_restrictions():
    mu = pd.Series([.0002,.0005,.001,.002],index=list('ABCD'))
    cov = np.diag([.0001,.00015,.0002,.0003])
    blocks, targets = {'LOCAL':[0,1],'OTHER':[2,3]}, {'LOCAL':.4,'OTHER':.6}
    result = run_constrained_frontier(mu,cov,blocks,targets,max_weight=.4,n_points=12)
    assert result['diagnostics']['accepted_points'] == 12
    for row in result['points'].itertuples():
        assert check_frontier_weights(row.weights,mu,blocks,targets,.4)
        assert row.return_log_annual == pytest.approx(row.return_daily*252)
    # Each constrained point is at least as risky as the unconstrained solution at its return.
    from src.markowitz_classic import min_variance_return_target
    for row in result['points'].itertuples():
        w = min_variance_return_target(mu,cov,row.return_daily)
        assert w @ cov @ w <= row.vol_daily**2 + 1e-9


@pytest.mark.parametrize('values', [[.001],[.001,.001]])
def test_degenerate_frontiers_return_valid_minimum(values):
    mu=pd.Series(values,index=list('AB')[:len(values)])
    cov=np.diag([.0001,.0004][:len(values)])
    result=run_frontier(mu,cov,n_points=10)
    assert len(result['frontier']) == 1
    if len(mu)==2:
        assert result['minvar']['weights_array'][0] == pytest.approx(.8,abs=1e-4)
    constrained=run_constrained_frontier(mu,cov,{'X':list(range(len(mu)))},{'X':1.})
    assert len(constrained['points'])==1


def test_clean_frontier_handles_invalid_and_empty_points():
    assert len(_clean_frontier([(np.nan,1),(.01,.001)])) == 1
    assert _clean_frontier([]).empty
