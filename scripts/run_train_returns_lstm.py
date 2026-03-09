import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.train_returns_lstm import train_all_return_models_lstm

if __name__ == "__main__":
    train_all_return_models_lstm()
    print("\nLSTM return models trained.")
