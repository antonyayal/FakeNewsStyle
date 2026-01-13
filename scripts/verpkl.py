from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

pkl_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "FakeNewsCorpusSpanish"
    / "test.pkl"
)

df = pd.read_pickle(pkl_path)
print(df.iloc[17]) # Imprime la fila 45 (índice 44)