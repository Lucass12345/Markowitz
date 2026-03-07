
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "predictions"


def build_prediction_table():

    ret = pd.read_csv(OUTPUT_DIR/"return_predictions.csv")

    vol = pd.read_csv(OUTPUT_DIR/"volatility_predictions.csv")

    df = ret.merge(vol,on="ticker")

    df.to_csv(OUTPUT_DIR/"predictions.csv",index=False)

    print(df.head())
