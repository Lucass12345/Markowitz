import numpy as np
import pandas as pd
import pytest
from src.optimization.covariance import RiskConfig, estimate_historical_risk, regularize_covariance


def test_pairwise_keeps_market_dates_and_repairs_indefinite_matrix():
    data = pd.DataFrame({'A':[-1,1,-1,1,np.nan,np.nan], 'B':[-1,1,np.nan,np.nan,-1,1],
                         'C':[np.nan,np.nan,-1,1,1,-1]}, index=pd.bdate_range('2024-01-01',periods=6))
    eligible, cov, diagnostic = estimate_historical_risk(data, RiskConfig(min_history=4,min_overlap=2))
    assert list(eligible) == ['A','B','C'] and len(eligible) == 6
    pd.testing.assert_frame_equal(eligible, data)
    assert diagnostic['global_complete_dates'] == 0
    assert diagnostic['pair_observations']['A']['B'] == 2
    assert diagnostic['min_eigenvalue_before'] < 0 and diagnostic['regularized']
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_exclusion_reasons_and_deterministic_overlap_resolution():
    data = pd.DataFrame({'A':[1,2,3,np.nan,np.nan,np.nan], 'B':[np.nan,np.nan,np.nan,2,3,4],
                         'SHORT':[1,np.nan,np.nan,np.nan,np.nan,np.nan]})
    eligible, _, info = estimate_historical_risk(data, RiskConfig(min_history=3,min_overlap=2))
    assert list(eligible) == ['B']
    assert 'history' in info['excluded_assets']['SHORT']
    assert 'overlap' in info['excluded_assets']['A']


def test_regularization_limit_and_duplicate_axes():
    with pytest.raises(ValueError, match='adjustment'):
        regularize_covariance(np.array([[1,2],[2,1.]]), RiskConfig(max_psd_adjustment=0))
    with pytest.raises(ValueError, match='Duplicate'):
        estimate_historical_risk(pd.DataFrame([[1,2],[2,3]],columns=['A','A']), RiskConfig(2,2))


def test_classic_input_builder_receives_valid_covariance_without_global_drop():
    from src.optimization.inputs_builder import build_model_inputs
    data = pd.DataFrame({'A':[-1,1,-1,1,np.nan,np.nan], 'B':[-1,1,np.nan,np.nan,-1,1],
                         'C':[np.nan,np.nan,-1,1,1,-1]}, index=pd.bdate_range('2024-01-01',periods=6))
    mu, cov, tickers = build_model_inputs('classic',data,RiskConfig(4,2))
    assert tickers == ['A','B','C']
    assert np.isfinite(mu).all() and np.linalg.eigvalsh(cov).min() > 0
    assert mu.attrs['input_diagnostics']['global_complete_dates'] == 0
