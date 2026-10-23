import numpy as np
import pandas as pd
import pytest
from src.portfolio_risk import path_risk, estimated_risk


def test_var_es_and_drawdown_include_initial_equity():
    data=np.log1p([-.1,.05,-.2,.01])
    risk=path_risk(data,.75)
    assert risk['var_historical']==pytest.approx(.125)
    assert risk['expected_shortfall_historical']==pytest.approx(.2)
    assert risk['max_drawdown']==pytest.approx(-.244)
    assert risk['risk_horizon_sessions']==1


def test_risk_contributions_sum_to_total_and_concentration():
    w=pd.Series([.6,.4],index=['A','B'])
    cov=pd.DataFrame([[.0001,.00002],[.00002,.0004]],index=w.index,columns=w.index)
    result=estimated_risk(w,cov)
    assert result['hhi']==pytest.approx(.52)
    details=result['risk_contributions']
    assert sum(r['component_vol_daily'] for r in details)==pytest.approx(result['estimated_vol_daily'])
    assert sum(r['percent_of_variance_risk'] for r in details)==pytest.approx(1)


def test_scenarios_insufficient_joint_calendar_do_not_invent_returns():
    data=pd.DataFrame({'A':[.01,-.02,np.nan,np.nan],'B':[np.nan,np.nan,.02,-.01]})
    result=estimated_risk(pd.Series([.5,.5],index=['A','B']),np.eye(2)*.01,data,min_joint_observations=2)
    assert result['historical_scenario']['n_joint_observations']==0
    assert 'var_historical' not in result['historical_scenario']


def test_covariance_labels_are_aligned_for_risk_contributions():
    weights=pd.Series([.9,.1],index=['A','B'])
    cov=pd.DataFrame(np.diag([.0001,.04]),index=weights.index,columns=weights.index)
    original=estimated_risk(weights,cov)
    reordered=estimated_risk(weights,cov.loc[['B','A'],['B','A']])
    assert original==reordered
    with pytest.raises(ValueError,match='labels'):
        estimated_risk(weights,cov.rename(index={'B':'C'},columns={'B':'C'}))
