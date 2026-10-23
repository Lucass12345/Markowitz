"""Past-only model callbacks with explicit fixed model specifications.

These deliberately do not load today's active predictions into a historical test.
Hyperparameters must be fixed before choosing the evaluation window.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from src.optimization.covariance import RiskConfig, estimate_historical_risk
from src.markowitz_classic import max_sharpe


@dataclass(frozen=True)
class ModelStrategy:
    mode: str = 'classic'
    min_history: int = 252
    min_overlap: int = 126
    seq_len: int = 40
    epochs: int = 5
    seed: int = 42
    lstm_config: dict = field(default_factory=lambda: {'hidden_sizes':[64,64],'activation':'relu','lr':.001})

    def __call__(self, history):
        observed,cov,_=estimate_historical_risk(history,RiskConfig(self.min_history,self.min_overlap))
        mu=observed.mean()
        if self.mode=='gjr_garch':
            from src.models.vol_model_garch import forecast_gjr_garch_vol
            vol=pd.Series({ticker:forecast_gjr_garch_vol(observed[ticker])[1] for ticker in observed})
            from src.optimization.inputs_builder import _build_ai_vol_cov
            cov=_build_ai_vol_cov(cov,vol)
        elif self.mode=='lstm':
            import torch
            from src.models.sequence_utils import build_lstm_sequences
            from src.models.lstm_search import refit_deployment
            torch.set_num_threads(1)
            for ticker in observed:
                series=observed[ticker].dropna()
                X,y=build_lstm_sequences(series,self.seq_len,5)
                if len(y)<20:
                    raise ValueError(f'Insufficient past LSTM sequences for {ticker}')
                model,xm,xs,ym,ys=refit_deployment(X,y,self.lstm_config,self.epochs,seed=self.seed)
                latest=(series.iloc[-self.seq_len:].to_numpy(dtype=np.float32).reshape(1,self.seq_len,1)-xm)/xs
                with torch.no_grad():
                    mu[ticker]=float(model(torch.tensor(latest,dtype=torch.float32)).item()*ys+ym)
        elif self.mode!='classic':
            raise ValueError(f'Unknown strategy {self.mode}')
        return pd.Series(max_sharpe(mu,cov),index=mu.index)
