from pathlib import Path
import pandas as pd

from src.data.load_data import assert_current_dataset

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "predictions"


def build_prediction_table():
    ret = pd.read_csv(OUTPUT_DIR / "return_predictions.csv")
    vol = pd.read_csv(OUTPUT_DIR / "volatility_predictions.csv")
    for name, predictions in (("retornos", ret), ("volatilidad", vol)):
        if "base_currency" not in predictions or not predictions["base_currency"].eq("USD").all():
            raise ValueError(f"Las predicciones de {name} no tienen moneda base USD verificable. Regeneralas.")
        if ("dataset_version" not in predictions or predictions["dataset_version"].isna().any()
                or predictions["dataset_version"].nunique() != 1):
            raise ValueError(f"Las predicciones de {name} requieren una única versión de dataset verificable.")
        assert_current_dataset(predictions)
    if ret["dataset_version"].iloc[0] != vol["dataset_version"].iloc[0]:
        raise ValueError("Las predicciones de retornos y volatilidad pertenecen a datasets diferentes.")
    # Keep one common provenance; prevent ambiguous dataset_version_x/y fields.
    volatility = vol.drop(columns=["dataset_version", "base_currency"])
    collisions = (set(ret.columns) & set(volatility.columns)) - {"ticker"}
    volatility = volatility.rename(columns={name: f"volatility_{name}" for name in collisions})
    df = ret.merge(volatility, on="ticker", validate="one_to_one")
    assert_current_dataset(df)
    df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(df.head())
    return df
