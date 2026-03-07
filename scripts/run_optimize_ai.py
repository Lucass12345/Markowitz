import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.optimization.optimize_predicted_returns import optimize_with_predictions

if __name__ == "__main__":

    optimize_with_predictions()