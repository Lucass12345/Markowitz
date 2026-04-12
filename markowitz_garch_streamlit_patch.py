from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import sys


def locate_repo_root() -> Path:
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    ]

    for candidate in candidates:
        if (
            (candidate / "app").exists()
            and (candidate / "src").exists()
            and (candidate / "Data").exists()
        ):
            return candidate

    raise RuntimeError(
        "No pude encontrar la raíz del repo. "
        "Ejecutá este script desde la carpeta raíz de Markowitz."
    )


def backup_file(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_garch_fix_{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


INPUTS_BUILDER = r'''from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.optimization.lstm_lab_loader import load_active_lstm_predictions_df

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "outputs" / "predictions"


def _read_prediction_csv(filename: str) -> pd.DataFrame:
    path = PRED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. "
            f"Generalo primero desde el pipeline correspondiente."
        )

    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{filename} debe contener la columna 'ticker'.")

    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


def load_predictions(prediction_file: str = "predictions.csv") -> pd.DataFrame:
    return _read_prediction_csv(prediction_file)


def load_volatility_predictions(prediction_file: str = "volatility_predictions.csv") -> pd.DataFrame:
    preds = _read_prediction_csv(prediction_file)

    required_cols = {"ticker", "vol_pred"}
    missing = required_cols - set(preds.columns)
    if missing:
        raise ValueError(
            f"{prediction_file} no tiene columnas requeridas: {sorted(missing)}"
        )

    preds["vol_pred"] = preds["vol_pred"].astype(float)
    preds = preds.dropna(subset=["vol_pred"]).copy()
    preds = preds.drop_duplicates(subset=["ticker"], keep="last").copy()
    return preds


def _build_ai_vol_cov(returns_subset: pd.DataFrame, pred_vol: pd.Series) -> pd.DataFrame:
    hist_cov = returns_subset.cov()
    hist_vol = returns_subset.std()

    hist_vol = hist_vol.replace(0, np.nan)

    corr = hist_cov.div(hist_vol, axis=0).div(hist_vol, axis=1)
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    d = pd.Series(pred_vol, index=returns_subset.columns, dtype=float)
    cov_ai = corr.mul(d, axis=0).mul(d, axis=1)
    cov_ai = (cov_ai + cov_ai.T) / 2.0

    return cov_ai


def build_model_inputs(
    model_type: str,
    returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """
    Devuelve:
    - mu
    - cov
    - tickers alineados

    model_type:
    - classic
    - ai_returns
    - ai_returns_lstm
    - ai_volatility
    """
    model_type = str(model_type).strip().lower()

    returns = returns.copy()
    returns.columns = returns.columns.astype(str).str.strip()

    if model_type == "classic":
        mu = returns.mean()
        cov = returns.cov()
        tickers = list(mu.index)
        return mu, cov, tickers

    if model_type == "ai_returns":
        preds = load_predictions("predictions.csv").set_index("ticker")

        if "mu_pred" not in preds.columns:
            raise ValueError("predictions.csv debe contener columna 'mu_pred'")

        common = [t for t in returns.columns if t in preds.index]
        if not common:
            raise ValueError("No hay tickers en común entre returns.csv y predictions.csv")

        returns_sub = returns[common].copy()
        mu = preds.loc[common, "mu_pred"].astype(float)
        cov = returns_sub.cov()
        return mu, cov, common

    if model_type == "ai_returns_lstm":
        preds = load_active_lstm_predictions_df().copy()
        preds["ticker"] = preds["ticker"].astype(str).str.strip()
        preds = preds.set_index("ticker")

        if "mu_pred" not in preds.columns:
            raise ValueError("Active LSTM Lab predictions must contain column 'mu_pred'")

        if "dir_acc_test" not in preds.columns:
            raise ValueError("Active LSTM Lab predictions must contain column 'dir_acc_test'")

        common = [t for t in returns.columns if t in preds.index]
        if not common:
            raise ValueError("No active LSTM Lab predictions available for optimizer universe.")

        returns_sub = returns[common].copy()
        preds = preds.loc[common].copy()

        hist_mu = returns_sub.mean()
        mu = hist_mu.copy()

        good_mask = preds["dir_acc_test"].astype(float) > 0.52
        mu.loc[good_mask.index[good_mask]] = preds.loc[good_mask, "mu_pred"].astype(float)

        cov = returns_sub.cov()

        mu = mu.loc[common]
        cov = cov.loc[common, common]

        return mu, cov, common

    if model_type == "ai_volatility":
        preds = load_volatility_predictions("volatility_predictions.csv").set_index("ticker")

        common = [t for t in returns.columns if t in preds.index]
        if not common:
            raise ValueError(
                "No hay tickers en común entre returns.csv y volatility_predictions.csv"
            )

        returns_sub = returns[common].copy()
        mu = returns_sub.mean()
        pred_vol = preds.loc[common, "vol_pred"].astype(float)
        cov = _build_ai_vol_cov(returns_sub, pred_vol)

        mu = mu.loc[common]
        cov = cov.loc[common, common]

        return mu, cov, common

    raise ValueError(
        f"model_type inválido: {model_type}. "
        f"Usá 'classic', 'ai_returns', 'ai_returns_lstm' o 'ai_volatility'."
    )
'''

GARCH_PAGE = r'''from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_data import load_returns
from src.models.train_volatility import forecast_all_volatility
from app.garch_lab_storage import save_garch_run

PRED_PATH = ROOT / "outputs" / "predictions" / "volatility_predictions.csv"


def _get_file_mtime(path: Path) -> str:
    if not path.exists():
        return "No generado todavía"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def _load_current_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRED_PATH)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


st.title("GARCH Lab")
st.caption(
    "Esta página usa directamente Data/returns.csv. "
    "Cuando actualizás las predicciones, se escribe outputs/predictions/volatility_predictions.csv "
    "y el modo ai_volatility de Markowitz consume ese archivo en forma directa."
)

returns = load_returns().copy()
returns.index = pd.to_datetime(returns.index)

with st.sidebar:
    st.header("Configuración")
    start_date = st.selectbox(
        "Training start date",
        options=["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
        index=2,
    )
    min_obs = st.number_input(
        "Minimum observations per ticker",
        min_value=60,
        max_value=5000,
        value=252,
        step=10,
    )
    save_run = st.checkbox("Guardar run en outputs/garch_lab/runs", value=True)

st.subheader("Dataset fuente")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(returns):,}")
c2.metric("Tickers", f"{returns.shape[1]:,}")
c3.metric("Date range", f"{returns.index.min().date()} → {returns.index.max().date()}")

with st.expander("Preview Data/returns.csv", expanded=False):
    st.dataframe(returns.tail(20), use_container_width=True)

st.subheader("Archivo que usa Markowitz")
st.code(
    f"Fuente optimizer ai_volatility:\n{PRED_PATH}\n"
    f"Última actualización: {_get_file_mtime(PRED_PATH)}"
)

col_run, col_preview = st.columns([1, 1])

with col_run:
    if st.button("Actualizar volatilidades desde Data/returns.csv", use_container_width=True):
        with st.spinner("Corriendo GJR-GARCH / EWMA fallback por ticker..."):
            df_pred = forecast_all_volatility(
                start_date=str(start_date),
                min_obs=int(min_obs),
            )

        if df_pred.empty:
            st.warning("No se generaron predicciones. Revisá start_date / min_obs.")
        else:
            run_id = None
            if save_run:
                metadata = {
                    "model": "GJR-GARCH(1,1,1) Student-t",
                    "source_dataset": "Data/returns.csv",
                    "prediction_output": str(PRED_PATH),
                    "start_date": str(start_date),
                    "min_obs": int(min_obs),
                    "rows": int(len(df_pred)),
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                }
                run_id = save_garch_run(df_pred, metadata)

            st.success("Archivo de volatilidades actualizado correctamente.")
            if run_id:
                st.info(f"Run guardado: {run_id}")

            st.write("Resumen por fuente de modelo")
            if "model_source" in df_pred.columns:
                summary = (
                    df_pred["model_source"]
                    .value_counts(dropna=False)
                    .rename_axis("model_source")
                    .reset_index(name="count")
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

            st.write("Preview de volatility_predictions.csv")
            st.dataframe(df_pred, use_container_width=True, hide_index=True)

with col_preview:
    current = _load_current_predictions()
    if current.empty:
        st.info("Todavía no existe outputs/predictions/volatility_predictions.csv")
    else:
        st.write("Preview archivo actual")
        st.dataframe(current, use_container_width=True, hide_index=True)

        if {"vol_pred"}.issubset(current.columns):
            st.write("Top 15 volatilidades predichas")
            top_df = current.sort_values("vol_pred", ascending=False).head(15).copy()
            st.bar_chart(top_df.set_index("ticker")["vol_pred"])

st.divider()
st.subheader("Notas operativas")
st.markdown(
    "- Ya no hace falta subir ningún CSV a Streamlit.\n"
    "- Esta pantalla lee directamente `Data/returns.csv`.\n"
    "- El optimizer `ai_volatility` ya no depende de `predictions.csv`; toma `volatility_predictions.csv` en forma directa.\n"
    "- El modo `ai_returns` sigue usando `predictions.csv` para `mu_pred`, sin tocar tu flujo MLP actual."
)
'''

STREAMLIT_APP_PATCH = r'''MODES = ["classic", "ai_returns", "ai_returns_lstm", "ai_volatility"]
MODE_LABELS = {
    "classic": "Classic Markowitz",
    "ai_returns": "AI Returns (MLP)",
    "ai_returns_lstm": "AI Returns (LSTM)",
    "ai_volatility": "AI Volatility (GJR-GARCH)",
}
MODE_HELP = {
    "classic": "Historical mean returns + historical covariance.",
    "ai_returns": "Predicted expected returns from MLP + historical covariance.",
    "ai_returns_lstm": "Predicted expected returns from LSTM + historical covariance.",
    "ai_volatility": "Historical expected returns + covariance rescaled using predicted volatility from outputs/predictions/volatility_predictions.csv generated from Data/returns.csv.",
}
'''


def patch_streamlit_app(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    if 'AI Volatility (GJR-GJR-GARCH)' not in original and 'GJR-GARCH' in original and 'volatility_predictions.csv' in original:
        print(f"[SKIP] {path} ya parece estar parcheado.")
        return

    start = original.find('MODES = ["classic", "ai_returns", "ai_returns_lstm", "ai_volatility"]')
    end = original.find('API_BASE_URL =')
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("No pude ubicar el bloque MODES / MODE_HELP en app/streamlit_app.py")

    updated = original[:start] + STREAMLIT_APP_PATCH + "\n\n" + original[end:]
    backup = backup_file(path)
    write_text(path, updated)
    print(f"[OK] app/streamlit_app.py parcheado. Backup: {backup.name}")


def patch_full_file(repo: Path, path: Path, new_content: str) -> None:
    backup = backup_file(path)
    write_text(path, new_content)
    print(f"[OK] {path.relative_to(repo)} reescrito. Backup: {backup.name}")


def main() -> None:
    repo = locate_repo_root()
    print(f"[INFO] Repo detectado: {repo}")

    targets = [
        repo / "src" / "optimization" / "inputs_builder.py",
        repo / "app" / "pages" / "4_GARCH_Lab.py",
        repo / "app" / "streamlit_app.py",
    ]

    missing = [str(p) for p in targets if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan archivos esperados:\n- " + "\n- ".join(missing))

    patch_full_file(repo, repo / "src" / "optimization" / "inputs_builder.py", INPUTS_BUILDER)
    patch_full_file(repo, repo / "app" / "pages" / "4_GARCH_Lab.py", GARCH_PAGE)
    patch_streamlit_app(repo / "app" / "streamlit_app.py")

    print("\n[DONE] Integración GARCH ↔ Streamlit ↔ Markowitz aplicada.")
    print("\nQué cambia:")
    print("1) GARCH Lab deja de pedir upload manual y usa Data/returns.csv.")
    print("2) GARCH Lab actualiza outputs/predictions/volatility_predictions.csv.")
    print("3) ai_volatility consume volatility_predictions.csv directamente.")
    print("4) ai_returns sigue intacto con predictions.csv.")

    print("\nDespués reiniciá servicios:")
    print('docker compose restart streamlit fastapi')
    print("\nO si corrés Streamlit/FastAPI fuera de Docker, reiniciá ambos procesos.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
