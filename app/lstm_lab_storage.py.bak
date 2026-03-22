from sqlalchemy import create_engine, text
from app.db import DATABASE_URL

def ensure_lstm_lab_tables():
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prediction_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            created_at TEXT,
            is_active INTEGER
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prediction_run_tickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            ticker TEXT,
            mu_pred REAL
        );
        """))

        conn.commit()

    print("DB OK")
