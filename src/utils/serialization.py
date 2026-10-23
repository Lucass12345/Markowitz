"""Strict JSON boundary shared by API, SQL JSON and immutable artifacts."""
import math
from datetime import date, datetime
from pathlib import Path
import numpy as np
import pandas as pd


def to_builtin(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value,dict):
        return {str(k):to_builtin(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [to_builtin(v) for v in value]
    if isinstance(value,pd.DataFrame):
        return to_builtin(value.to_dict(orient='records'))
    if isinstance(value,pd.Series):
        return to_builtin(value.to_dict())
    if isinstance(value,(datetime,date,pd.Timestamp)):
        return value.isoformat()
    if isinstance(value,Path):
        return str(value)
    if isinstance(value,(np.bool_,bool)):
        return bool(value)
    if isinstance(value,(np.integer,int)):
        return int(value)
    if isinstance(value,(np.floating,float)):
        return float(value) if math.isfinite(value) else None
    return value
