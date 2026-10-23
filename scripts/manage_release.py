"""Authenticated transport is performed by the operator; validate locally before activation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.publication.releases import import_release,activate_release,rollback,verify_release

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['import','verify','activate','rollback'])
    parser.add_argument('value',nargs='?')
    parser.add_argument('--data-dir',default='Data')
    args=parser.parse_args()
    if args.action=='rollback':
        print(rollback(args.data_dir))
    elif args.value is None:
        parser.error('This action requires a value')
    elif args.action=='import':
        print(import_release(args.value,args.data_dir))
    elif args.action=='verify':
        print(verify_release(args.value)['release_id'])
    else:
        print(activate_release(args.value,args.data_dir))
