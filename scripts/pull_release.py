"""Runtime pull from GitHub Actions using gh's configured authentication.

Transport authorization belongs to GitHub; local content/quality checks precede activation.
Run on the serving host/container with the SAME persistent Data volume as FastAPI.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.publication.releases import import_release,activate_release,verify_release


def safe_unpack(archive,destination):
    destination=Path(destination)
    with zipfile.ZipFile(archive) as bundle:
        if sum(item.file_size for item in bundle.infolist())>2*1024**3:
            raise ValueError('Candidate exceeds 2 GiB extraction limit')
        for item in bundle.infolist():
            path=destination/item.filename
            if not path.resolve().is_relative_to(destination.resolve()) or (item.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe candidate archive path or symlink')
        bundle.extractall(destination)



def validate_source_run(run, repo):
    workflow=str(run.get('path','')).split('@')[0]
    owner=str((run.get('head_repository') or {}).get('full_name',''))
    if (run.get('conclusion')!='success' or run.get('head_branch')!='main'
            or run.get('event') not in {'schedule','workflow_dispatch'}
            or owner.casefold()!=repo.casefold()
            or workflow!='.github/workflows/yahoo-models.yml'
            or not run.get('head_sha')):
        raise ValueError('Require successful main schedule/dispatch from the repository Yahoo workflow')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',type=int,required=True)
    parser.add_argument('--repo',default='Lucass12345/Markowitz')
    parser.add_argument('--data-dir',default='Data')
    parser.add_argument('--activate',action='store_true')
    args=parser.parse_args()
    run=json.loads(subprocess.check_output(['gh','api',f'repos/{args.repo}/actions/runs/{args.run_id}']))
    validate_source_run(run,args.repo)
    artifacts=json.loads(subprocess.check_output(['gh','api',f'repos/{args.repo}/actions/runs/{args.run_id}/artifacts']))['artifacts']
    matches=[a for a in artifacts if a['name']==f'yahoo-candidate-{args.run_id}' and not a['expired']]
    if len(matches)!=1:
        raise ValueError('Expected exactly one unexpired candidate artifact')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);archive=root/'artifact.zip'
        with archive.open('wb') as stream:
            subprocess.run(['gh','api',f'repos/{args.repo}/actions/artifacts/{matches[0]["id"]}/zip'],stdout=stream,check=True)
        safe_unpack(archive,root/'unpacked')
        bundles=list((root/'unpacked').rglob('release.json'))
        if len(bundles)!=1:
            raise ValueError('Expected exactly one candidate bundle')
        manifest=verify_release(bundles[0].parent)
        if manifest.get('code_commit') != run['head_sha']:
            raise ValueError('Candidate code commit differs from the trusted workflow run')
        installed=import_release(bundles[0].parent,args.data_dir)
        if args.activate:
            activate_release(installed.name,args.data_dir)
        print(json.dumps({'release_id':installed.name,'activated':args.activate}))

if __name__=='__main__':
    main()
