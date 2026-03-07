from pathlib import Path

# raíz del repo
ROOT = Path(__file__).resolve().parents[1]

folders = [

    # DATA
    ROOT / "Data" / "raw",
    ROOT / "Data" / "processed",

    # SRC SUBMODULES
    ROOT / "src" / "data",
    ROOT / "src" / "models",
    ROOT / "src" / "optimization",
    ROOT / "src" / "evaluation",

    # MODELS ARTIFACTS
    ROOT / "models_artifacts",
    ROOT / "models_artifacts" / "returns",
    ROOT / "models_artifacts" / "volatility",

    # OUTPUTS
    ROOT / "outputs",
    ROOT / "outputs" / "predictions",
    ROOT / "outputs" / "portfolios",
    ROOT / "outputs" / "figures",

    # APP (para streamlit después)
    ROOT / "app"
]

print("\nCreating project folders...\n")

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"✔ {folder.relative_to(ROOT)}")

print("\nStructure ready.\n")