from __future__ import annotations

from pathlib import Path
import pandas as pd


def convert_folder_xlsx_to_pkl(
    raw_dir: Path,
    processed_dir: Path,
    pattern: str = "*.xlsx",
    engine: str = "openpyxl",
) -> list[Path]:
    """
    Convierte todos los XLSX en raw_dir a PKL en processed_dir.
    Regresa la lista de archivos PKL generados.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(raw_dir.glob(pattern))
    if not xlsx_files:
        raise FileNotFoundError(f"No XLSX files found in: {raw_dir}")

    generated = []
    for xlsx_path in xlsx_files:
        df = pd.read_excel(xlsx_path, engine=engine)
        pkl_path = processed_dir / f"{xlsx_path.stem}.pkl"
        df.to_pickle(pkl_path)
        generated.append(pkl_path)

    return generated
