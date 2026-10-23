import numpy as np
import pandas as pd
from src.config import block_configuration, DEFAULT_CONFIG
from src.optimization.optimize_blocks import run_blocks_optimizer


def test_legacy_targets_retained():
    metadata = pd.DataFrame({'ticker':list('ABCDE'),'block':DEFAULT_CONFIG.blocks_order})
    assert block_configuration(metadata)['default_targets'] == DEFAULT_CONFIG.targets


def test_custom_market_can_optimize_without_python_changes():
    metadata = pd.DataFrame({'ticker':['UK','JP','JP2','X'],'block':['UK_EQUITY','JAPAN','JAPAN','OTHER']})
    config = block_configuration(metadata)
    assert config['blocks'] == {'JAPAN':2,'OTHER':1,'UK_EQUITY':1}
    mu = pd.Series([.001,.002,.0015,.0003], index=metadata.ticker)
    cov = pd.DataFrame(np.eye(4)*.0001,index=mu.index,columns=mu.index)
    result = run_blocks_optimizer(mu,cov,list(mu.index),metadata,config['default_targets'],max_weight=.6)
    assert np.abs(result['exposure_by_block']['diff']).max() < 1e-7
    assert result['coverage']['available_blocks']['JAPAN'] == 2
