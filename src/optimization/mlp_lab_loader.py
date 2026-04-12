from __future__ import annotations

import pandas as pd
from app.mlp_lab_storage import load_active_mlp_predictions_df


def load_active_mlp_predictions() -> pd.DataFrame:
    return load_active_mlp_predictions_df()
