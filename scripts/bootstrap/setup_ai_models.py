from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

files = {

"src/data/load_data.py": """
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"

def load_returns():
    df = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    df.columns = df.columns.astype(str).str.strip()
    return df
""",

"src/data/features.py": """
import pandas as pd

def build_features_for_ticker(series, lags=20):

    s = series.dropna()

    df = pd.DataFrame(index=s.index)
    df["ret"] = s

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = s.shift(lag)

    df["roll_mean_5"] = s.shift(1).rolling(5).mean()
    df["roll_mean_20"] = s.shift(1).rolling(20).mean()

    df["roll_std_5"] = s.shift(1).rolling(5).std()
    df["roll_std_20"] = s.shift(1).rolling(20).std()

    df["target"] = s.shift(-1)

    return df.dropna()
""",

"src/models/return_model_mlp.py": """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
from pathlib import Path


class ReturnMLPModel:

    def __init__(self):

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64,32),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True
            ))
        ])


    def fit(self,X,y):
        self.model.fit(X,y)


    def predict(self,X):
        return self.model.predict(X)


    def evaluate(self,X,y):

        pred = self.predict(X)

        mse = mean_squared_error(y,pred)

        return {
            "mse":float(mse),
            "rmse":float(np.sqrt(mse))
        }


    def save(self,path):

        Path(path).parent.mkdir(parents=True,exist_ok=True)

        joblib.dump(self.model,path)
""",

"src/models/vol_model_ewma.py": """
import pandas as pd


def ewma_variance(series, lam=0.94):

    s = series.dropna()

    var = pd.Series(index=s.index)

    var.iloc[0] = s.var()

    for i in range(1,len(s)):
        var.iloc[i] = lam*var.iloc[i-1] + (1-lam)*(s.iloc[i-1]**2)

    return var


def next_day_variance_forecast(series):

    v = ewma_variance(series)

    return float(v.iloc[-1])
""",

"src/models/train_returns.py": """
from pathlib import Path
import pandas as pd

from src.data.load_data import load_returns
from src.data.features import build_features_for_ticker
from src.models.return_model_mlp import ReturnMLPModel

ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = ROOT / "models_artifacts" / "returns"
OUTPUT_DIR = ROOT / "outputs" / "predictions"


def train_all_return_models():

    returns = load_returns()

    rows = []

    for ticker in returns.columns:

        df = build_features_for_ticker(returns[ticker])

        if len(df) < 200:
            continue

        X = df.drop(columns=["ret","target"])
        y = df["target"]

        split = int(len(df)*0.8)

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        X_test = X.iloc[split:]
        y_test = y.iloc[split:]


        model = ReturnMLPModel()

        model.fit(X_train,y_train)

        metrics = model.evaluate(X_test,y_test)

        pred = float(model.predict(X.tail(1))[0])

        model_path = ARTIFACTS_DIR / f"{ticker}_mlp.joblib"

        model.save(model_path)


        rows.append({
            "ticker":ticker,
            "mse":metrics["mse"],
            "rmse":metrics["rmse"],
            "mu_pred":pred
        })


    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_DIR/"return_predictions.csv",index=False)

    print(df.head())
""",

"src/models/train_volatility.py": """
from pathlib import Path
import pandas as pd

from src.data.load_data import load_returns
from src.models.vol_model_ewma import next_day_variance_forecast

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "predictions"


def forecast_all_volatility():

    returns = load_returns()

    rows = []

    for ticker in returns.columns:

        s = returns[ticker].dropna()

        if len(s) < 60:
            continue

        var = next_day_variance_forecast(s)

        rows.append({
            "ticker":ticker,
            "var_pred":var,
            "vol_pred":var**0.5
        })


    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_DIR/"volatility_predictions.csv",index=False)

    print(df.head())
""",

"src/models/predict.py": """
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
"""
}

for path,content in files.items():

    file_path = ROOT / path

    file_path.parent.mkdir(parents=True,exist_ok=True)

    file_path.write_text(content)

    print("created",path)