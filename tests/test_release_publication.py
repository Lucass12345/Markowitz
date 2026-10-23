import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.publication.releases import seal_candidate,activate_release,active_release,rollback,import_release


def candidate(root,version='v1',quality=.7):
    folder=root/version;data=folder/'data'/version;data.mkdir(parents=True)
    values=pd.DataFrame(np.random.default_rng(1).normal(0,.01,(20,2)),columns=['A','B'],index=pd.bdate_range('2024-01-01',periods=20))
    values.to_csv(data/'returns.csv')
    pd.DataFrame({'ticker':['A','B'],'block':['OTHER','OTHER']}).to_csv(data/'assets_metadata.csv',index=False)
    (data/'manifest.json').write_text(json.dumps({'version':version,'base_currency':'USD','return_type':'log'}))
    (folder/'candidate.json').write_text(json.dumps({'dataset_version':version}))
    (folder/'predictions').mkdir()
    common={'dataset_version':version,'base_currency':'USD','as_of':str(values.index[-1]),'mu_pred':.001,'vol_pred':.02,
            'dir_acc_val':quality,'dir_acc_test':1.,'amp_ratio':.3,'rmse_val':.01,'baseline_rmse_val':.02}
    preds=pd.DataFrame([{'ticker':ticker,**common} for ticker in ['A','B']])
    for name in ['lstm','garch']:
        preds.to_csv(folder/'predictions'/f'{name}.csv',index=False)
    return folder


def test_atomic_activation_and_rollback_share_data_and_signals(tmp_path,monkeypatch):
    root=tmp_path/'runtime'
    first=seal_candidate(candidate(tmp_path/'build'),root,min_history=10)
    activate_release(first.name,root)
    from src.data import load_data
    monkeypatch.setattr(load_data,'DATA_DIR',root)
    loaded=load_data.load_returns()
    assert loaded.attrs['dataset_version']=='v1'
    assert loaded.attrs['release_dir']==str(first)
    from src.optimization.inputs_builder import _pinned_signals
    assert _pinned_signals('ai_returns_lstm',loaded).dataset_version.eq('v1').all()
    second=seal_candidate(candidate(tmp_path/'build','v2'),root,min_history=10)
    activate_release(second.name,root)
    # A previously pinned data object still reads its own model snapshot.
    assert _pinned_signals('ai_returns_lstm',loaded).dataset_version.eq('v1').all()
    rollback(root)
    assert active_release(root)[1]['release_id']==first.name


def test_failed_candidate_or_tampered_transport_preserves_active_release(tmp_path):
    root=tmp_path/'runtime'
    valid=seal_candidate(candidate(tmp_path/'build'),root,min_history=10)
    activate_release(valid.name,root)
    with pytest.raises(ValueError,match='validation coverage'):
        seal_candidate(candidate(tmp_path/'bad','bad',quality=.1),root,min_history=10)
    assert active_release(root)[0]==valid
    (valid/'predictions'/'lstm.csv').write_text('corrupt')
    with pytest.raises(ValueError,match='checksum'):
        import_release(valid,tmp_path/'other')
    assert json.loads((root/'active_release.json').read_text())['release_id']==valid.name


def test_archive_rejects_path_traversal(tmp_path):
    import zipfile
    from scripts.pull_release import safe_unpack
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as bundle:
        bundle.writestr('../escape','bad')
    with pytest.raises(ValueError,match='Unsafe'):
        safe_unpack(archive,tmp_path/'destination')
