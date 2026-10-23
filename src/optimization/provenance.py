"""Content-addressed optimization evidence, independent of mutable active pointers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from src.utils.serialization import to_builtin

ROOT=Path(__file__).resolve().parents[2]


def code_version():
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    try:
        return {'commit':git('rev-parse','HEAD'),'dirty':bool(git('status','--porcelain','--untracked-files=no')),
                'working_diff_sha256':hashlib.sha256(git('diff','HEAD','--','src','api','app').encode()).hexdigest()}
    except (OSError,subprocess.CalledProcessError):
        return {'commit':os.getenv('CODE_VERSION','unknown'),'dirty':None}


def json_bytes(value):
    return json.dumps(to_builtin(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def seal_snapshot(returns, metadata, mu, cov, result, root=None):
    root=Path(root or os.getenv('OPTIMIZATION_SNAPSHOT_DIR',ROOT/'storage'/'optimization_snapshots'))
    root.mkdir(parents=True,exist_ok=True)
    provenance={'dataset_version':returns.attrs.get('dataset_version',result['inputs']['dataset_version']),
                'data_cutoff':returns.index.max().isoformat(),'base_currency':'USD','code':code_version(),
                'models':[signal for signal in (result['diagnostics']['risk'].get('signals',[])+result['diagnostics']['risk'].get('risk_signals',[])) if signal['source']=='ai'],
                'effective_configuration':result['inputs']}
    files={'returns.csv':returns.to_csv().encode(),'assets_metadata.csv':metadata.to_csv(index=False).encode(),
           'mu.csv':mu.to_csv().encode(),'covariance.csv':cov.to_csv().encode(),
           'result.json':json_bytes(result),'provenance.json':json_bytes(provenance)}
    hashes={name:hashlib.sha256(content).hexdigest() for name,content in files.items()}
    digest=hashlib.sha256(json_bytes(hashes)).hexdigest()
    directory=root/digest
    if not directory.exists():
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            staging=Path(tmp)/'snapshot'; staging.mkdir()
            for name,content in files.items():
                (staging/name).write_bytes(content)
            (staging/'manifest.json').write_bytes(json_bytes({'snapshot_id':digest,'sha256':hashes}))
            try:
                staging.rename(directory)
            except FileExistsError:
                pass
    load_snapshot(directory) # Refuse corrupted or overwritten evidence.
    return {**provenance,'snapshot_id':digest,'snapshot_path':str(directory)}


def load_snapshot(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    hashes=manifest['sha256']
    if hashlib.sha256(json_bytes(hashes)).hexdigest()!=manifest['snapshot_id']:
        raise ValueError('Snapshot manifest integrity failed')
    for name,digest in hashes.items():
        if Path(name).name!=name or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Snapshot integrity failed: {name}')
    return json.loads((directory/'result.json').read_text())
