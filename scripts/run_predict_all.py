import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.predict import build_prediction_table


if __name__ == "__main__":

    build_prediction_table()

    print("\nPrediction table generated")