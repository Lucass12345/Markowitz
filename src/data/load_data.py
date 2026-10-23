from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"


def active_data_dir():
    pointer = DATA_DIR / "active_dataset.json"
    if not pointer.exists():
        return DATA_DIR
    version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
    if Path(version).name != version:
        raise ValueError("Invalid dataset version")
    return DATA_DIR / "dynamic" / version


def load_returns(data_dir=None):
    directory = Path(data_dir) if data_dir is not None else active_data_dir()
    assert_usd_dataset(directory)
    df = pd.read_csv(directory / "returns.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_metadata():
    return pd.read_csv(active_data_dir() / "assets_metadata.csv")


def load_prices():
    directory = active_data_dir()
    assert_usd_dataset(directory)
    return pd.read_csv(directory / "prices.csv", index_col=0, parse_dates=True).sort_index()


def assert_current_dataset(predictions):
    """Reject predictions from another data vintage after a dynamic refresh."""
    current = active_data_dir()
    if current == DATA_DIR:
        if "dataset_version" in predictions and not predictions["dataset_version"].eq(DATA_DIR.name).all():
            raise ValueError("Las predicciones no corresponden al dataset activo. Actualizá el dataset y reentrená el modelo.")
        return
    assert_usd_dataset(current)
    if "dataset_version" not in predictions or not predictions["dataset_version"].eq(current.name).all():
        raise ValueError("Las predicciones no corresponden al dataset activo. Reentrená el modelo antes de optimizar.")


def assert_usd_dataset(directory=None):
    directory = Path(directory) if directory is not None else active_data_dir()
    if directory == DATA_DIR:
        return  # Legacy CSV was already converted with CCL by notebooks/Base.ipynb.
    path = directory / "manifest.json"
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("base_currency") != "USD":
        raise ValueError("Este dataset no tiene retornos unificados en USD. Regeneralo desde Dataset Dinámico antes de entrenar u optimizar.")
