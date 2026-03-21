import pandas as pd
from sqlalchemy import create_engine
from app.db import DATABASE_URL

def load_active_lstm_predictions_df():
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT prt.ticker, prt.mu_pred
    FROM prediction_runs pr
    JOIN prediction_run_tickers prt ON pr.run_id = prt.run_id
    WHERE pr.is_active = 1
    """

    df = pd.read_sql(query, engine)
    return df
