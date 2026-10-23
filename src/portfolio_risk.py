"""Ex-ante covariance risk and explicitly labeled historical/realized path risk."""
import numpy as np
import pandas as pd


def path_risk(log_returns, confidence=.95):
    if not 0 < confidence < 1:
        raise ValueError('confidence must be between 0 and 1')
    values=np.asarray(log_returns,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Path risk requires at least two finite daily log returns')
    losses=-np.expm1(values)
    var=float(np.quantile(losses,confidence))
    es=float(losses[losses>=var].mean())
    wealth=np.r_[1.,np.exp(np.cumsum(values))]
    drawdown=wealth/np.maximum.accumulate(wealth)-1
    return {'var_historical':var,'expected_shortfall_historical':es,'confidence':confidence,
            'risk_horizon_sessions':1,'risk_units':'fraction_of_equity_simple_loss',
            'max_drawdown':float(drawdown.min()),'drawdown_units':'fraction_of_peak_equity',
            'n_observations':len(values)}


def estimated_risk(weights, cov, returns=None, periods=252, min_joint_observations=126, confidence=.95):
    from src.markowitz import _align_mu_sigma
    weights, cov = _align_mu_sigma(weights, cov)
    w=np.asarray(weights,dtype=float)
    matrix=np.asarray(cov,dtype=float)
    sigma=float(np.sqrt(w@matrix@w))
    if not np.isfinite(sigma) or sigma<=0:
        raise ValueError('Risk contributions require positive finite portfolio volatility')
    marginal=matrix@w/sigma
    component=w*marginal
    labels=list(weights.index) if isinstance(weights,pd.Series) else list(range(len(w)))
    details=[{'ticker':ticker,'weight':float(w[i]),'marginal_vol_daily':float(marginal[i]),
              'component_vol_daily':float(component[i]),'component_vol_annual':float(component[i]*np.sqrt(periods)),
              'percent_of_variance_risk':float(component[i]/sigma)} for i,ticker in enumerate(labels)]
    result={'measurement':'ex_ante','currency':'USD','return_type':'daily_log','periods':periods,
            'hhi':float(w@w),'effective_number_assets':float(1/(w@w)),
            'estimated_vol_daily':sigma,'estimated_vol_annual':sigma*np.sqrt(periods),'risk_contributions':details}
    if returns is not None:
        selected=returns.loc[:,labels]
        # Joint sample is only for historical scenarios, never the optimizer's pairwise risk input.
        complete=selected.loc[:,np.abs(w)>1e-12].dropna(how='any')
        scenario={'measurement':'historical_fixed_weight_scenario','n_joint_observations':len(complete),
                  'required_observations':min_joint_observations,'calendar_policy':'joint_observed_held_assets_only'}
        if len(complete)>=max(2,min_joint_observations):
            simple=np.expm1(complete)@w[np.abs(w)>1e-12]
            scenario.update(path_risk(np.log1p(simple),confidence))
        else:
            scenario['unavailable_reason']='insufficient_joint_history; no returns imputed'
        result['historical_scenario']=scenario
        result['observed_asset_vol_annual']=(selected.std()*np.sqrt(periods)).to_dict()
        result['observed_correlations']=selected.corr().to_dict()
        result['pair_observations']=(selected.notna().astype(int).T@selected.notna().astype(int)).to_dict()
    return result
