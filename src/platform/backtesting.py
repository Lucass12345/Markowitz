"""Worker-domain service for reproducible constrained walk-forward research."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.walk_forward import BacktestConfig, walk_forward
from src.backtesting.strategies import ConstrainedModelStrategy, matched_benchmark_weights
from src.utils.serialization import to_builtin


STRATEGIES = ('classic', 'lstm', 'mlp', 'gjr_garch', 'combined', 'equal_weight', 'buy_and_hold')


def _write_json(path, value):
    path.write_text(json.dumps(to_builtin(value), indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')


def _universe_assets(config, database):
    frozen = config.get('universe_snapshot')
    if frozen is None and config.get('universe_id'):
        from src.platform.universes import UniverseService
        frozen = UniverseService(database=database).get(config['universe_id'])
    if frozen is None:
        return None
    return frozen.get('assets_json', frozen.get('assets', [])) if isinstance(frozen, dict) else frozen


def _select_universe(returns, metadata, config, database):
    metadata = metadata.copy(deep=True)
    assets = _universe_assets(config, database)
    requested_tickers = config.get('tickers')
    if requested_tickers is None:
        requested_tickers = (config.get('optimization') or {}).get('tickers')
    tickers = list(returns.columns if requested_tickers is None else requested_tickers)
    overrides = config.get('metadata_overrides') or []
    if assets is not None:
        from src.platform.optimization import _enabled
        tickers = [asset['ticker'] for asset in assets if _enabled(asset.get('enabled', asset.get('active', True)))]
        overrides = [*assets, *(overrides.values() if isinstance(overrides, dict) else overrides)]
    elif isinstance(overrides, dict):
        overrides = [dict(values, ticker=ticker) for ticker, values in overrides.items()]
    if not tickers or len(set(tickers)) != len(tickers) or not set(tickers).issubset(returns.columns):
        raise ValueError('Backtest universe must contain unique known enabled tickers')
    for item in overrides:
        if not isinstance(item, dict) or 'ticker' not in item:
            raise ValueError('Metadata overrides require a ticker')
        mask = metadata.ticker.eq(item['ticker'])
        for field, value in item.items():
            if field not in {'ticker', 'active', 'enabled'} and value is not None:
                metadata.loc[mask, field] = value
    missing = set(tickers) - set(metadata.ticker)
    if missing:
        raise ValueError(f'Missing frozen metadata for tickers: {sorted(missing)}')
    if metadata.ticker.duplicated().any():
        raise ValueError('Dataset metadata tickers must be unique')
    return returns.loc[:, tickers].copy(), metadata.loc[metadata.ticker.isin(tickers)].copy(), deepcopy(assets)


class BacktestService:
    """Prepare reports in a temporary job workspace; publication belongs to caller.

    This service never marks SQL jobs completed or activates models. If a strategy
    fails, it raises and leaves only unpublished temporary evidence for diagnosis.
    """

    def __init__(self, dataset_service=None, model_service=None, optimizer=None):
        self.dataset_service = dataset_service
        self.model_service = model_service
        self.optimizer = optimizer

    def run(self, config, context):
        from src.platform.optimization import _configuration
        config = deepcopy(_configuration(config))
        context.check_cancelled()
        context.progress(1, 'backtest.dataset', 'Materializing the pinned dataset')
        if self.dataset_service is None:
            from src.platform.datasets import DatasetService
            self.dataset_service = DatasetService(store=context.store, database=context.database)
        requested = getattr(context, 'dataset_version', None) or config.get('dataset_version')
        resolved = self.dataset_service.resolve(requested)
        dataset_version = resolved['version']
        snapshot = self.dataset_service.materialize(dataset_version, Path(context.temp_dir) / 'dataset')
        for explicit in (config.get('dataset_version'), getattr(context, 'dataset_version', None)):
            if explicit and explicit not in {snapshot.id, snapshot.version}:
                raise ValueError('Backtest job and request reference different dataset versions')
        returns, metadata, universe = _select_universe(snapshot.load_returns(), snapshot.load_metadata(), config, context.database)
        returns.attrs.update(dataset_version=snapshot.version, base_currency='USD')
        # A date when no selected asset traded is not an observed portfolio session.
        # Partial missing marks remain and cause an explicit valuation error if held.
        dates_without_any_mark = int(returns.isna().all(axis=1).sum())
        returns = returns.dropna(how='all')
        selected = list(config.get('strategies') if config.get('strategies') is not None else ['classic', 'equal_weight', 'buy_and_hold'])
        if len(selected) != len(set(selected)) or not selected or not set(selected).issubset(STRATEGIES):
            raise ValueError(f'Choose unique backtest strategies from {STRATEGIES}')
        optimization = deepcopy(config.get('optimization') or {})
        if not optimization.get('targets'):
            raise ValueError('Backtesting requires explicit optimizer block targets')
        # The public request calls this model_parameters; retain models for
        # existing frame-level callers and store the effective fixed settings.
        model_configs = deepcopy(config.get('models') or config.get('model_parameters') or {})
        model_requirements = self._model_history_requirements(selected, optimization, model_configs)
        simulation = self._simulation_config(config, returns, max(model_requirements.values(), default=0))
        if simulation.oos_end is not None:
            returns = returns.loc[:simulation.oos_end]
        optimization.update(periods=simulation.periods, rf_annual_simple=simulation.rf_annual_simple,
                            dataset_version=snapshot.version, calculate_frontiers=False, tickers=list(returns.columns))
        for field, default in (('min_history', simulation.min_train), ('min_overlap', min(126, simulation.min_train))):
            if field not in (optimization.get('risk_config') or {}):
                optimization.setdefault(field, default)
        matched = bool(config.get('benchmark_constraints', True))
        output = Path(context.temp_dir) / 'backtest'
        output.mkdir(parents=True, exist_ok=True)
        results = {}
        for index, name in enumerate(selected):
            context.check_cancelled()
            base = 5. + index * 75. / len(selected)
            width = 75. / len(selected)
            context.progress(base, f'backtest.{name}', 'Starting past-only walk-forward decisions')
            estimator = None
            if name not in {'equal_weight', 'buy_and_hold'}:
                estimator = ConstrainedModelStrategy(name, metadata, optimization, snapshot.version,
                    self._report_context(context, output), model_service=self.model_service,
                    optimizer=self.optimizer, model_configs=model_configs)
            elif matched:
                estimator = lambda history: matched_benchmark_weights(history, metadata, optimization)
            def on_decision(number, total, date, estimator=estimator, base=base, width=width, name=name):
                context.progress(base + width * number / total, f'backtest.{name}.decision', str(date))
                if isinstance(estimator, ConstrainedModelStrategy):
                    estimator.fold_progress = lambda fraction, stage, message=None: context.progress(
                        base + width * (number + .9 * fraction) / total, f'backtest.{name}.{stage}', message)
            result = walk_forward(returns, estimator=estimator, config=simulation,
                strategy=name if name in {'equal_weight', 'buy_and_hold'} else 'model',
                on_decision=on_decision, check_cancelled=context.check_cancelled)
            result['path'].to_csv(output / f'{name}_path.csv', index_label='date')
            _write_json(output / f'{name}_decisions.json', result['decisions'])
            results[name] = result
        dates = next(iter(results.values()))['path'].index
        if any(not value['path'].index.equals(dates) for value in results.values()):
            raise ValueError('Strategies produced different OOS evaluation calendars')
        curves = pd.DataFrame({name: value['path']['wealth'] for name, value in results.items()})
        curves.index.name = 'date'
        curves.to_csv(output / 'wealth_curves.csv')
        metrics = {name: value['metrics'] for name, value in results.items()}
        differences = self._differences(metrics)
        summary = {'dataset_id': snapshot.id, 'dataset_version': snapshot.version,
                   'job_id': context.job_id, 'actor': context.actor,
                   'configuration': {**config, 'optimization': optimization, 'models': model_configs, 'simulation': to_builtin(simulation.__dict__),
                                     'dataset_version': snapshot.version, 'universe_snapshot': universe},
                   'strategies': selected, 'metrics': metrics, 'differences_vs_equal_weight': differences,
                   'training_requirements': {'requested_min_train': int(config.get('min_train', 252)),
                                             'effective_min_train': simulation.min_train,
                                             'model_minimum_observations': model_requirements},
                   'benchmark_definition': 'equal within blocks; buy-and-hold from that initial allocation' if matched else 'standard unconstrained equal weight and buy-and-hold',
                   'calendar_policy': 'dates with any selected asset observed; reject missing held-asset marks',
                   'dates_without_any_selected_mark': dates_without_any_mark,
                   'constraints_timing': 'target weights at rebalances; holdings drift between decisions',
                   'limitations': ['Fixed chosen universe; no correction for survivorship or historical metadata changes.',
                                   'No performance superiority inferred from this run.'],
                   'oos_start': str(dates[0]), 'oos_end': str(dates[-1]),
                   'methodology': next(iter(results.values()))['methodology']}
        _write_json(output / 'report.json', summary)
        _write_json(output / 'configuration.json', summary['configuration'])
        rows = [{'strategy': name, **value} for name, value in metrics.items()]
        pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
        context.check_cancelled()
        context.progress(85, 'backtest.artifacts', 'Reports prepared; mandatory object-store publication remains')
        report = {**summary, 'summary': rows, 'wealth_curves': curves.reset_index().to_dict(orient='records')}
        return to_builtin({**report, 'result': report, 'artifacts_dir': str(output)})

    @staticmethod
    def _model_history_requirements(selected, optimization, model_configs):
        from src.platform.optimization import _mode
        _, estimator = _mode({**optimization, 'model_type': 'classic', 'return_model': 'historical'})
        uses_garch = bool(set(selected) & {'gjr_garch', 'combined'}) or (
            estimator == 'gjr_garch' and bool(set(selected) - {'equal_weight', 'buy_and_hold'}))
        if not uses_garch:
            return {}
        specification = model_configs.get('garch') or {}
        specification = {**(specification.get('parameters') or {}), **specification}
        minimum = max(252, int(specification.get('min_obs', 252)))
        validation, test = int(specification.get('validation_days', 20)), int(specification.get('test_days', 20))
        if validation < 1 or test < 1:
            raise ValueError('GARCH validation_days and test_days must be positive')
        return {'garch': minimum + validation + test}

    @staticmethod
    def _simulation_config(config, returns, required_model_history=0):
        start = config.get('oos_start') or config.get('start')
        end = config.get('oos_end') or config.get('end')
        if config.get('oos_sessions') is not None:
            sessions = int(config['oos_sessions'])
            if sessions < 2:
                raise ValueError('At least two OOS sessions are required')
            ending = returns.loc[:end] if end is not None else returns
            if len(ending) <= sessions:
                raise ValueError('OOS sessions leave no training history')
            derived = ending.index[-sessions]
            start = str(max(pd.Timestamp(start), derived)) if start is not None else str(derived)
        optimization = config.get('optimization') or {}
        train_window = int(config.get('train_window', 504))
        if train_window < required_model_history:
            raise ValueError(f'GARCH needs at least {required_model_history} training observations including validation/test; increase train_window')
        return BacktestConfig(train_window=train_window,
            min_train=max(int(config.get('min_train', 252)), required_model_history), rebalance_every=int(config.get('rebalance_every', 5)),
            periods=int(config.get('periods', optimization.get('periods', 252))),
            commission_bps=float(config.get('commission_bps', 0.)),
            transaction_cost_bps=float(config.get('transaction_cost_bps', 0.)),
            rf_annual_simple=float(optimization.get('rf_annual_simple', config.get('rf_annual_simple', 0.))),
            slippage_bps=float(config.get('slippage_bps', 0.)), oos_start=start, oos_end=end)

    @staticmethod
    def _report_context(context, output):
        # A proxy keeps all model/scaler checkpoints beneath the published report.
        from src.backtesting.strategies import FoldContext
        return FoldContext(context, output, 'backtest-report')

    @staticmethod
    def _differences(metrics):
        baseline = metrics.get('equal_weight')
        if baseline is None:
            return {}
        keys = ('cumulative_simple_return', 'annualized_simple_return', 'realized_vol_annual',
                'realized_sharpe_annual', 'max_drawdown', 'turnover_total')
        return {name: {key: float(values[key] - baseline[key]) if np.isfinite(values[key]) and np.isfinite(baseline[key]) else None
                       for key in keys} for name, values in metrics.items() if name != 'equal_weight'}
