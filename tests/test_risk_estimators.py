import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf
from src.optimization.covariance import RiskConfig, ledoit_wolf_covariance
from src.optimization.inputs_builder import build_model_inputs


def test_shrinkage_matches_reference_and_records_conditioning():
    data=pd.DataFrame(np.random.default_rng(10).normal(size=(40,3))*.01,columns=list('ABC'))
    cov,diag=ledoit_wolf_covariance(data,RiskConfig(20,10))
    np.testing.assert_allclose(cov,LedoitWolf().fit(data).covariance_,atol=1e-12)
    assert 0<=diag['shrinkage']<=1 and diag['joint_observations_used']==40
    assert diag['min_eigenvalue_after']>0
    mu,cov,_=build_model_inputs('classic',data,RiskConfig(20,10),covariance_estimator='ledoit_wolf')
    assert mu.attrs['input_diagnostics']['estimator']=='ledoit_wolf'


def test_shrinkage_rejects_insufficient_common_history_without_imputation():
    data=pd.DataFrame({'A':[1,2,3,np.nan], 'B':[np.nan,2,3,4]})
    with pytest.raises(ValueError,match='jointly observed'):
        ledoit_wolf_covariance(data,RiskConfig(3,3))


def test_garch_can_be_selected_for_classic_returns(monkeypatch):
    from src.optimization import inputs_builder
    data=pd.DataFrame(np.random.default_rng(3).normal(0,.01,size=(20,2)),columns=['A','B'],index=pd.bdate_range('2024-01-01',periods=20))
    data.attrs['dataset_version']='v1'
    monkeypatch.setattr(inputs_builder,'_load_signals', lambda mode: pd.DataFrame([{'ticker':'A','vol_pred':.02,'as_of':str(data.index[-1]),'dataset_version':'v1','base_currency':'USD'}]) if mode=='ai_volatility' else pd.DataFrame())
    mu,cov,tickers=build_model_inputs('classic',data,RiskConfig(10,10),covariance_estimator='gjr_garch')
    pd.testing.assert_series_equal(mu,data.mean())
    assert np.sqrt(cov.loc['A','A'])==pytest.approx(.02)
    assert np.sqrt(cov.loc['B','B'])==pytest.approx(data.B.std())
    assert tickers==['A','B']
