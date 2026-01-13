from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

# -------- INPUT (raw) --------
xlsx_path = (
    BASE_DIR
    / "data"
    / "raw"
    / "FakeNewsCorpusSpanish"
    / "train.xlsx"
)

df = pd.read_excel(xlsx_path, engine="openpyxl")

# -------- OUTPUT (processed) --------
pkl_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "FakeNewsCorpusSpanish"
    / "train.pkl"
)

pkl_path.parent.mkdir(parents=True, exist_ok=True)

df.to_pickle(pkl_path)

# -------- VALIDATION --------
print(type(df))
print(df.shape)
print(df.head().to_string())
print(df.iloc[19]) # Imprime la fila 45 (índice 44)