from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

moves = [
    # Data -> outputs/legacy
    ("Data/classic_points_annual.csv", "outputs/legacy/classic_points_annual.csv"),
    ("Data/cloud_classic_annual.csv", "outputs/legacy/cloud_classic_annual.csv"),
    ("Data/compare_classic_vs_blocks.png", "outputs/legacy/compare_classic_vs_blocks.png"),
    ("Data/frontier_classic_annual.csv", "outputs/legacy/frontier_classic_annual.csv"),
    ("Data/weights_maxsharpe_blocks_detailed.csv", "outputs/legacy/weights_maxsharpe_blocks_detailed.csv"),
    ("Data/weights_minvar_blocks_detailed.csv", "outputs/legacy/weights_minvar_blocks_detailed.csv"),

    # outputs old figure -> outputs/legacy
    ("outputs/figures/compare_historical_vs_ai.png", "outputs/legacy/compare_historical_vs_ai.png"),

    # old scripts -> legacy
    ("scripts/run_blocks.py", "scripts/legacy/run_blocks.py"),
    ("scripts/run_classic.py", "scripts/legacy/run_classic.py"),
    ("scripts/run_compare.py", "scripts/legacy/run_compare.py"),

    # notebook
    ("Base.ipynb", "notebooks/Base.ipynb"),

    # bootstrap
    ("scripts/setup_lstm_full.py", "scripts/bootstrap/setup_lstm_full.py"),
]

delete_files = [
    "scripts/setup_lstm_returns.py",
]

create_dirs = [
    "outputs/legacy",
    "scripts/legacy",
    "scripts/bootstrap",
    "notebooks",
]


def ensure_dirs():
    for d in create_dirs:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"dir ok: {d}")


def move_file(src_rel, dst_rel):
    src = ROOT / src_rel
    dst = ROOT / dst_rel

    if not src.exists():
        print(f"skip (not found): {src_rel}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"skip (destination exists): {dst_rel}")
        return

    shutil.move(str(src), str(dst))
    print(f"moved: {src_rel} -> {dst_rel}")


def delete_file(path_rel):
    path = ROOT / path_rel
    if path.exists():
        path.unlink()
        print(f"deleted: {path_rel}")
    else:
        print(f"skip delete (not found): {path_rel}")


def main():
    print("\n=== Creating directories ===")
    ensure_dirs()

    print("\n=== Moving files ===")
    for src, dst in moves:
        move_file(src, dst)

    print("\n=== Deleting redundant files ===")
    for f in delete_files:
        delete_file(f)

    print("\nCleanup finished.\n")


if __name__ == "__main__":
    main()