"""Reproducible comparison; fixed specifications, no selection on the OOS window."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data.load_data import load_returns
from src.backtesting import BacktestConfig, compare_strategies
from src.backtesting.strategies import ModelStrategy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tickers',nargs='+',required=True)
    parser.add_argument('--train-window',type=int,default=504)
    parser.add_argument('--oos-days',type=int,default=20)
    parser.add_argument('--rebalance-every',type=int,default=5)
    parser.add_argument('--lstm-epochs',type=int,default=5)
    parser.add_argument('--commission-bps',type=float,default=0.)
    parser.add_argument('--transaction-cost-bps',type=float,default=0.)
    parser.add_argument('--output',default='outputs/backtesting/comparison')
    args=parser.parse_args()
    data=load_returns()[args.tickers].dropna(how="all").tail(args.train_window+args.oos_days)
    if data.isna().any().any():
        raise ValueError('Choose an explicitly shared observed evaluation calendar. Missing marks cannot be priced by this backtest.')
    config=BacktestConfig(args.train_window,args.train_window,args.rebalance_every,commission_bps=args.commission_bps,transaction_cost_bps=args.transaction_cost_bps)
    models={mode:ModelStrategy(mode=mode,epochs=args.lstm_epochs) for mode in ['classic','lstm','gjr_garch']}
    results=compare_strategies(data,models,config)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    summary={'dataset_version':data.attrs.get('dataset_version'),'data_cutoff':str(data.index[-1]),
             'calendar_policy':'dates with at least one selected asset observed; reject partial missing marks',
             'tickers':args.tickers,'config':asdict(config),'models':{k:asdict(v) for k,v in models.items()},
             'metrics':{name:value['metrics'] for name,value in results.items()},
             'limitations':'Fixed universe; no survivorship correction. Short windows are execution demonstrations, not evidence of performance improvement.'}
    for name,value in results.items():
        value['path'].to_csv(output/f'{name}_path.csv')
        (output/f'{name}_decisions.json').write_text(json.dumps(value['decisions'],indent=2))
    (output/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(pd.DataFrame(summary['metrics']).T[['cumulative_simple_return','realized_sharpe_annual','max_drawdown','turnover_total']].to_string())
    print(f'Report: {output}')

if __name__=='__main__':
    main()
