
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
