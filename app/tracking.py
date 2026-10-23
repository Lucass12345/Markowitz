"""Track each remote independently; preserve successful IDs when another service fails."""
import json
import os
from pathlib import Path
from infrastructure.storage.minio_client import upload_bytes
from src.utils.serialization import to_builtin


def track_run_details(run_db_id, run_label, config, result):
    provenance=result.get('provenance',{})
    directory=Path(provenance['snapshot_path']) if provenance.get('snapshot_path') else None
    status={'db_run_id':run_db_id,'snapshot_id':provenance.get('snapshot_id'),
            'mlflow':{'status':'not_configured','run_id':None},'minio':{'status':'not_configured','objects':[]}}
    if os.getenv('MLFLOW_TRACKING_URI'):
        try:
            import mlflow
            mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
            mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME','markowitz-streamlit'))
            with mlflow.start_run(run_name=run_label or f'{config["mode"]}-run-{run_db_id}') as run:
                status['mlflow']['run_id']=run.info.run_id
                mlflow.set_tags({'source':'fastapi','db_run_id':str(run_db_id),'snapshot_id':provenance.get('snapshot_id','legacy'),
                                 'dataset_version':provenance.get('dataset_version','unknown'),
                                 'code_commit':provenance.get('code',{}).get('commit','unknown')})
                mlflow.log_dict(to_builtin(config),'configuration.json')
                for name in ('minvar','maxsharpe','blocks'):
                    metrics={f'{name}_{key}':float(value) for key,value in to_builtin(result[name]['stats']).items()
                             if isinstance(value,(float,int)) and not isinstance(value,bool)}
                    mlflow.log_metrics(metrics)
                if directory:
                    mlflow.log_artifacts(str(directory),artifact_path='snapshot')
            status['mlflow']['status']='completed'
        except Exception as exc:
            status['mlflow'].update(status='failed',error=str(exc))
    if os.getenv('MINIO_ENDPOINT') and directory:
        try:
            prefix=f'run_{run_db_id}/{provenance["snapshot_id"]}'
            for path in sorted(directory.iterdir()):
                uri=upload_bytes('optimization-runs',f'{prefix}/{path.name}',path.read_bytes())
                status['minio']['objects'].append(uri)
            status['minio']['status']='completed'
        except Exception as exc:
            status['minio'].update(status='failed',error=str(exc))
    return status


def track_run_mlflow(run_db_id, run_label, config, result):
    """Backward-compatible tuple interface for existing callers."""
    status=track_run_details(run_db_id,run_label,config,result)
    errors=[f'{name}: {info.get("error",info["status"])}' for name,info in status.items()
            if name in {'mlflow','minio'} and info['status']!='completed']
    return status['mlflow']['run_id'], '; '.join(errors) or None
