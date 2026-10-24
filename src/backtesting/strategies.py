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


class FoldContext:
    """An isolated model-training context under a parent backtest job.

    Training artifacts are nested under the backtest bundle; progress and safe
    cancellation delegate to its parent. No active model/dataset lookup occurs.
    """

    def __init__(self, parent, directory, identity, progress=None):
        from pathlib import Path
        from uuid import NAMESPACE_URL, uuid5
        self.parent = parent
        self.temp_dir = Path(directory)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.job_id = str(uuid5(NAMESPACE_URL, f'{parent.job_id}:{identity}'))
        self.parent_job_id = parent.job_id
        self._progress = progress

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def progress(self, percent, stage, message=None):
        if self._progress is not None:
            self._progress(percent, stage, message)

    def check_cancelled(self):
        self.parent.check_cancelled()


class ConstrainedModelStrategy:
    """Past-only retraining followed by the exact production portfolio engine.

    The model service fits from the supplied history, never from registry models
    trained after the decision. Signal acceptance, units, fallback, covariance and
    block solves are delegated to ``optimize_frame`` instead of reimplemented.
    """

    FAMILIES = {
        'classic': (), 'lstm': ('lstm',), 'mlp': ('mlp',),
        'gjr_garch': ('garch',), 'combined': ('lstm', 'garch'),
    }
    MODES = {
        'classic': 'classic', 'lstm': 'ai_returns_lstm', 'mlp': 'ai_returns',
        'gjr_garch': 'ai_volatility', 'combined': 'combined',
    }

    def __init__(self, mode, metadata, config, dataset_version, context, *, model_service=None, optimizer=None, model_configs=None):
        from copy import deepcopy
        if mode not in self.FAMILIES:
            raise ValueError(f'Unknown constrained backtest strategy: {mode}')
        self.mode, self.metadata = mode, metadata.copy(deep=True)
        self.config = deepcopy(config)
        self.dataset_version, self.context = dataset_version, context
        self.model_service, self.optimizer = model_service, optimizer
        self.model_configs = deepcopy(model_configs or {})
        self.fold_number = 0
        self.fold_progress = None

    def __call__(self, history):
        from pathlib import Path
        from src.platform.optimization import optimize_frame, risk_configuration
        self.context.check_cancelled()
        history = history.copy(deep=True)
        history.attrs.update(dataset_version=self.dataset_version, base_currency='USD')
        cutoff = history.index[-1]
        mode_config = {**self.config, 'model_type': self.MODES[self.mode],
                       'dataset_version': self.dataset_version, 'as_of': str(cutoff),
                       'calculate_frontiers': False}
        # A shared optimizer configuration may describe a production return
        # model. Each comparison strategy must still select its own return model.
        mode_config.pop('return_model', None)
        if self.mode in {'gjr_garch', 'combined'}:
            mode_config['covariance_estimator'] = 'gjr_garch'
            mode_config['volatility_model'] = 'garch'
        # A non-volatility strategy can still explicitly select GARCH risk, as in
        # the optimizer. Fit that risk model on the same historical frame.
        families = list(self.FAMILIES[self.mode])
        from src.platform.optimization import _mode
        _, covariance_estimator = _mode(mode_config)
        if covariance_estimator == 'gjr_garch' and 'garch' not in families:
            families.append('garch')
        # Use precisely the optimizer's past-only eligibility before expensive
        # training, so a short newly listed asset cannot fail an otherwise valid
        # decision for assets that the portfolio engine would retain.
        training_history = estimate_historical_risk(history, risk_configuration(mode_config))[0] if families else history
        predictions, records = {}, []
        for position, family in enumerate(families):
            self.context.check_cancelled()
            if self.model_service is None:
                from src.platform.models import ModelService
                self.model_service = ModelService()
            directory = Path(self.context.temp_dir) / 'folds' / self.mode / f'{self.fold_number:05d}' / family
            def progress(percent, stage, message=None, position=position, family=family):
                if self.fold_progress is not None:
                    fraction = (position + float(percent) / 100.) / max(1, len(families))
                    self.fold_progress(fraction, f'{family}:{stage}', message)
            child = FoldContext(self.context, directory, f'{self.mode}:{self.fold_number}:{family}', progress)
            specification = self.model_configs.get(family, {})
            model_config = {**(specification.get('parameters') or {}), **specification, 'dataset_version': self.dataset_version}
            model_config.setdefault('start_date', 'full')
            trained = self.model_service.train_on_frame(family, training_history.copy(deep=True), self.dataset_version, model_config, child)
            frame = pd.DataFrame(trained.get('predictions', []))
            predictions[family] = frame
            artifact_dir = trained.get('artifacts_dir')
            if artifact_dir and not Path(artifact_dir).resolve().is_relative_to(Path(self.context.temp_dir).resolve()):
                raise ValueError('Fold model artifacts must remain in the isolated backtest workspace')
            records.append({'family': family, 'train_cutoff': str(cutoff), 'job_id': child.job_id,
                            'metrics': trained.get('metrics', {}), 'models': trained.get('models', []),
                            'artifacts': str(Path(artifact_dir).resolve().relative_to(Path(self.context.temp_dir).resolve())) if artifact_dir else None})
        optimizer = self.optimizer or optimize_frame
        result = optimizer(history, self.metadata, mode_config,
                           return_predictions=predictions.get('lstm', predictions.get('mlp', pd.DataFrame())),
                           volatility_predictions=predictions.get('garch', pd.DataFrame()))
        weights = result['blocks']['weights_series'].copy()
        weights.attrs['decision_diagnostics'] = {
            'model_type': self.MODES[self.mode], 'dataset_version': self.dataset_version,
            'train_cutoff': str(cutoff), 'configuration': result.get('inputs', mode_config),
            'selection': result.get('diagnostics', {}), 'training': records,
            'solver_message': result['blocks'].get('solver_message'),
        }
        self.fold_number += 1
        return weights


def matched_benchmark_weights(history, metadata, config):
    """Equal weights within each requested block, using training eligibility only."""
    from src.optimization.optimize_blocks import build_blocks_from_meta
    from src.markowitz import validate_block_feasibility
    from src.platform.optimization import risk_configuration
    observed, _, diagnostic = estimate_historical_risk(history, risk_configuration(config))
    targets = config['targets']
    tickers = list(observed.columns)
    blocks = build_blocks_from_meta(tickers, metadata, targets)
    cap = float(config.get('max_weight', .15))
    validate_block_feasibility(len(tickers), blocks, targets, cap)
    weights = pd.Series(0., index=tickers)
    for block, indices in blocks.items():
        if indices:
            weights.iloc[indices] = float(targets[block]) / len(indices)
    weights.attrs['decision_diagnostics'] = {
        'benchmark': 'equal_weight_within_blocks', 'targets': dict(targets), 'max_weight': cap,
        'dataset_version': history.attrs.get('dataset_version'),
        'eligibility': diagnostic, 'constraints_timing': 'initial/rebalance target; holdings drift afterward',
    }
    return weights
