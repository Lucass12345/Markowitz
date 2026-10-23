import json
import sys
from types import SimpleNamespace
import pandas as pd
import pytest
from src.optimization import lstm_lab_loader as loader


def test_local_loader_binds_identity_and_uses_validation_amplitude_once(tmp_path,monkeypatch):
    monkeypatch.setattr(loader,'ROOT',tmp_path)
    monkeypatch.setitem(sys.modules,'app.lstm_lab_storage',SimpleNamespace())
    output=tmp_path/'outputs'/'predictions';output.mkdir(parents=True)
    run=tmp_path/'run';(run/'per_ticker').mkdir(parents=True)
    pd.DataFrame({'split':['val','val'],'y_true':[0.,2.],'y_pred':[0.,1.]}).to_csv(run/'per_ticker'/'A_predictions.csv',index=False)
    pd.DataFrame({'ticker':['A'],'mu_pred':[.001],'model_run_id':['run'],'amp_ratio':[.9]}).to_csv(output/'return_predictions_lstm.csv',index=False)
    pointer=output/'lstm_production.json';pointer.write_text(json.dumps({'run_id':'run','run_output_dir':str(run)}))
    result=loader.load_active_lstm_predictions_df()
    assert result.amp_ratio.iloc[0]==pytest.approx(.5)
    assert not any(name.endswith(('_x','_y')) for name in result)
    pointer.write_text(json.dumps({'run_id':'other','run_output_dir':str(run)}))
    with pytest.raises(ValueError,match='identity'):
        loader.load_active_lstm_predictions_df()
