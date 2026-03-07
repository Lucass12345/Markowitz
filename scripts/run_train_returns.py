import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.train_returns import train_all_return_models


if __name__ == "__main__":

    train_all_return_models()

    print("\nReturn models trained")