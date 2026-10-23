"""Build an isolated candidate; activate data and models together only after quality gates."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.yahoo_dataset import refresh_dataset


def build_candidate(args):
    from src.data.load_data import DATA_DIR, load_returns
    from src.models.train_returns_lstm_lab import run_lstm_lab
    from src.models.lstm_features import AVAILABLE_FEATURES
    from src.models.train_volatility import forecast_all_volatility
    from src.optimization.lstm_lab_loader import _compute_amp_ratio_from_run_dir
    from src.publication.releases import seal_candidate,activate_release
    with tempfile.TemporaryDirectory() as tmp:
        staging=Path(tmp)
        manifest=refresh_dataset(args.tickers,start=args.start,ccl_file=args.ccl_file,
            max_fx_age_days=args.max_fx_age_days,data_dir=staging/'download',activate=False)
        version=manifest['version']
        dataset=staging/'download'/'dynamic'/version
        returns=load_returns(data_dir=dataset)
        trained=run_lstm_lab(selected_features=AVAILABLE_FEATURES,epochs=args.epochs,start_date='full',
            feature_search=True,returns_df=returns,dataset_version=version,publish_latest=False,persist_db=False)
        candidate=staging/'candidate';(candidate/'data').mkdir(parents=True)
        shutil.copytree(dataset,candidate/'data'/version)
        (candidate/'predictions').mkdir()
        import pandas as pd
        preds=trained['predictions'].merge(trained['metrics'].drop(columns=['mu_pred'],errors='ignore'),on='ticker',validate='one_to_one')
        amp=_compute_amp_ratio_from_run_dir(str(Path(trained['predictions_path']).parent))
        preds=preds.merge(amp,on='ticker',validate='one_to_one')
        preds['model_run_id']=trained['run_id'];preds['prediction_units']='daily_log_return'
        preds.to_csv(candidate/'predictions'/'lstm.csv',index=False)
        vol=forecast_all_volatility(start_date=None,returns_df=returns,dataset_version=version,output_dir=candidate/'predictions')
        vol.to_csv(candidate/'predictions'/'garch.csv',index=False)
        shutil.copytree(Path(trained['predictions_path']).parent,candidate/'lstm_run')
        shutil.copytree(Path(trained['model_dir']),candidate/'lstm_models')
        from src.optimization.provenance import code_version
        (candidate/'candidate.json').write_text(json.dumps({'dataset_version':version,'lstm_run_id':trained['run_id'],
            'base_currency':'USD','code_commit':code_version()['commit'],'forecast_horizon':5,'publication_mode':'candidate'}))
        sealed=seal_candidate(candidate,min_signal_coverage=args.min_signal_coverage)
        if not args.candidate_only:
            activate_release(sealed.name)
        print(json.dumps({'release_id':sealed.name,'path':str(sealed),'activated':not args.candidate_only}))
        return sealed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tickers',nargs='+')
    parser.add_argument('--start',default='2015-01-01')
    parser.add_argument('--ccl-file',help='CSV histórico CCL ARS/USD: fecha,cierre o date,rate')
    parser.add_argument('--max-fx-age-days',type=int,default=7)
    parser.add_argument('--retrain',action='store_true')
    parser.add_argument('--candidate-only',action='store_true')
    parser.add_argument('--min-signal-coverage',type=float,default=.25)
    parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args()
    if not 0<args.min_signal_coverage<=1:
        parser.error('min-signal-coverage must be in (0,1]')
    if not args.tickers:
        from src.data.load_data import load_metadata
        args.tickers=load_metadata().ticker.tolist()
    if args.retrain:
        build_candidate(args)
    else:
        if args.candidate_only:
            parser.error('--candidate-only requires --retrain')
        print(refresh_dataset(args.tickers,start=args.start,ccl_file=args.ccl_file,max_fx_age_days=args.max_fx_age_days))

if __name__=='__main__':
    main()
