import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import uuid
import numpy as np
import pandas as pd
from src.optimization.provenance import json_bytes


def _data_root(data_dir=None):
    if data_dir is not None:
        return Path(data_dir)
    from src.data.load_data import DATA_DIR
    return DATA_DIR


def active_release(data_dir=None):
    root=_data_root(data_dir)
    pointer=root/'active_release.json'
    if not pointer.exists():
        return None
    info=json.loads(pointer.read_text())
    release_id=info['release_id']
    if len(release_id)!=64 or any(c not in '0123456789abcdef' for c in release_id):
        raise ValueError('Invalid release id')
    directory=root/'releases'/release_id
    manifest=json.loads((directory/'release.json').read_text())
    return directory,manifest


def validate_candidate(directory,min_history=252,min_signal_coverage=.25):
    """Validation gate uses validation metrics only. Test columns are never read."""
    from src.optimization.covariance import RiskConfig,estimate_historical_risk
    from src.optimization.predictions import checked_predictions, row_reason, PredictionPolicy
    if not np.isfinite(min_signal_coverage) or not 0 < min_signal_coverage <= 1:
        raise ValueError("Signal coverage must be in (0,1]")
    directory=Path(directory)
    release=json.loads((directory/'candidate.json').read_text())
    version=release['dataset_version']
    if Path(version).name!=version:
        raise ValueError('Invalid candidate dataset version')
    dataset=directory/'data'/version
    manifest=json.loads((dataset/'manifest.json').read_text())
    if manifest.get('base_currency')!='USD' or manifest.get('return_type')!='log':
        raise ValueError('Candidate requires USD daily log returns')
    returns=pd.read_csv(dataset/'returns.csv',index_col=0,parse_dates=True)
    returns.attrs['dataset_version']=version
    eligible,_,_=estimate_historical_risk(returns,RiskConfig(min_history,min(126,min_history)))
    if len(eligible.columns)!=len(returns.columns):
        raise ValueError('Candidate has insufficient asset history/overlap')
    meta=pd.read_csv(dataset/'assets_metadata.csv')
    if meta.ticker.duplicated().any() or set(meta.ticker)!=set(returns.columns) or meta.block.isna().any():
        raise ValueError('Candidate metadata does not cover its universe')
    gates={}
    for name,value in [('lstm','mu_pred'),('garch','vol_pred')]:
        signals=checked_predictions(pd.read_csv(directory/'predictions'/f'{name}.csv'),returns,value)
        accepted=[]
        for ticker in returns:
            row=signals.loc[ticker] if ticker in signals.index else None
            if row_reason(row,returns[ticker],value,PredictionPolicy()) is not None:
                continue
            if name=='lstm':
                metrics=pd.to_numeric(row.reindex(['dir_acc_val','amp_ratio','rmse_val','baseline_rmse_val']),errors='coerce')
                if not np.isfinite(metrics).all() or not (metrics.dir_acc_val>.52 and metrics.amp_ratio>.15 and metrics.rmse_val<metrics.baseline_rmse_val):
                    continue
            accepted.append(ticker)
        coverage=len(accepted)/len(returns.columns)
        if coverage<min_signal_coverage or not accepted:
            raise ValueError(f'{name} validation coverage {coverage:.1%} below required {min_signal_coverage:.1%}')
        gates[name]={'accepted_tickers':accepted,'coverage':coverage,'selection_split':'validation' if name=='lstm' else 'finite_daily_volatility'}
    return {'passed':True,'min_history':min_history,'min_signal_coverage':min_signal_coverage,'models':gates}


def seal_candidate(candidate,data_dir=None,**gate_config):
    candidate=Path(candidate)
    gates=validate_candidate(candidate,**gate_config)
    files={str(p.relative_to(candidate)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(candidate.rglob('*')) if p.is_file() and p.name!='release.json'}
    release_id=hashlib.sha256(json_bytes(files)).hexdigest()
    metadata=json.loads((candidate/'candidate.json').read_text())
    manifest={**metadata,'release_id':release_id,'sha256':files,'gates':gates}
    root=_data_root(data_dir)/'releases'; root.mkdir(parents=True,exist_ok=True)
    destination=root/release_id
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            staging=Path(tmp)/'bundle'
            shutil.copytree(candidate,staging)
            (staging/'release.json').write_bytes(json_bytes(manifest))
            staging.rename(destination)
    verify_release(destination)
    return destination


def verify_release(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'release.json').read_text())
    if hashlib.sha256(json_bytes(manifest['sha256'])).hexdigest()!=manifest['release_id']:
        raise ValueError('Release manifest integrity failed')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name!='release.json'}
    if actual!=set(manifest['sha256']) or any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('Release inventory or symbolic links invalid')
    for name,digest in manifest['sha256'].items():
        path=directory/name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Release checksum/path invalid: {name}')
    # Re-run quality checks, never trust a claimed passed flag in a transported bundle.
    validate_candidate(directory,min_history=manifest['gates']['min_history'],min_signal_coverage=manifest['gates']['min_signal_coverage'])
    return manifest


def activate_release(release_id,data_dir=None):
    root=_data_root(data_dir)
    if Path(release_id).name!=release_id:
        raise ValueError('Invalid release id')
    directory=root/'releases'/release_id
    verify_release(directory)
    old=active_release(root)
    pointer={'release_id':release_id,'previous_release_id':old[1]['release_id'] if old else None}
    temporary=root/f'.release_{uuid.uuid4().hex}.json'
    temporary.write_bytes(json_bytes(pointer))
    temporary.replace(root/'active_release.json')
    return pointer


def rollback(data_dir=None):
    root=_data_root(data_dir)
    previous=json.loads((root/'active_release.json').read_text()).get('previous_release_id')
    if previous is None:
        raise ValueError('No previous release to roll back to')
    return activate_release(previous,root)


def import_release(source,data_dir=None):
    """Transport an unpacked bundle from an authenticated download/shared volume."""
    manifest=verify_release(source)
    root=_data_root(data_dir)/'releases';root.mkdir(parents=True,exist_ok=True)
    destination=root/manifest['release_id']
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            staging=Path(tmp)/'bundle';shutil.copytree(source,staging)
            verify_release(staging);staging.rename(destination)
    verify_release(destination)
    return destination
