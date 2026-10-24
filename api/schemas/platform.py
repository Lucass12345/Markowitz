from typing import Literal, Any
from pydantic import BaseModel,Field,ConfigDict,model_validator
from api.schemas.optimize import OptimizeRequest


class RequestModel(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    timeout_seconds:int=Field(7200,ge=1,le=86400)
    max_attempts:int=Field(3,ge=1,le=5)

class RefreshRequest(RequestModel):
    dataset_version:str|None=None
    universe_id:str|None=None
    tickers:list[str]|None=None
    start:str='2015-01-01'
    end:str|None=None
    refresh_mode:Literal['full','incremental']='incremental'
    revision_overlap_days:int=Field(30,ge=5,le=365)
    full_refresh_every_days:int=Field(30,ge=1,le=365)
    max_fx_age_days:int=Field(7,ge=0,le=30)
    block_overrides:dict[str,str]=Field(default_factory=dict)
    activate:bool=False

class TrainingRequest(RequestModel):
    family:Literal['lstm','mlp','garch']
    dataset_version:str|None=None
    tickers:list[str]|None=None
    parameters:dict[str,Any]=Field(default_factory=dict)

    @model_validator(mode='after')
    def bounds(self):
        for key,low,high in [('epochs',1,1000),('seq_len',2,1000),('batch_size',1,4096),('max_iter',1,5000)]:
            value=self.parameters.get(key)
            if value is not None and (not isinstance(value,(int,float)) or not low<=value<=high):
                raise ValueError(f'{key} must be between {low} and {high}')
        if self.parameters.get('forecast_horizon',5)!=5:
            raise ValueError('Production horizon is five trading sessions')
        if self.parameters.get('device','auto') not in {'auto','cpu','cuda'}:
            raise ValueError('device must be auto, cpu or cuda')
        return self

class OptimizationRequest(OptimizeRequest):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    model_type:Literal['classic','ai_returns','ai_returns_lstm','ai_volatility','combined']
    dataset_version:str|None=None
    universe_id:str|None=None
    as_of:str|None=None
    return_model_ids:list[str]|None=None
    volatility_model_ids:list[str]|None=None
    tickers:list[str]|None=None
    timeout_seconds:int=Field(1800,ge=1,le=86400)
    max_attempts:int=Field(3,ge=1,le=5)
    return_model:Literal['historical','lstm','mlp']|None=None
    volatility_model:Literal['historical','garch','gjr_garch']|None=None

class BacktestRequest(RequestModel):
    dataset_version:str|None=None
    universe_id:str|None=None
    optimization:dict[str,Any]=Field(default_factory=dict)
    strategies:list[str]=Field(default_factory=lambda:['classic','lstm','mlp','gjr_garch','combined','equal_weight','buy_and_hold'])
    train_window:int=Field(504,ge=30)
    min_train:int=Field(252,ge=2)
    rebalance_every:int=Field(5,ge=1)
    start:str|None=None
    end:str|None=None
    oos_sessions:int|None=Field(None,ge=2)
    oos_start:str|None=None
    oos_end:str|None=None
    commission_bps:float=Field(0,ge=0,lt=1000)
    transaction_cost_bps:float=Field(0,ge=0,lt=1000)
    slippage_bps:float=Field(0,ge=0,lt=1000)
    model_parameters:dict[str,Any]=Field(default_factory=dict)
    benchmark_constraints:bool=True

class UniverseRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str
    assets:list[dict]
    config:dict=Field(default_factory=dict)

class UniverseUpdate(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str|None=None
    assets:list[dict]|None=None
    config:dict|None=None

class InferenceRequest(RequestModel):
    dataset_version:str|None=None

class JobRequest(RequestModel):
    kind:Literal['dataset.refresh','dataset.import','model.train','model.infer','optimization','backtest']
    payload:dict=Field(default_factory=dict)
