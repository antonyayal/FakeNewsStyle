from pathlib import Path
import os
import pandas as pd

# ---------------------------------
# Base directory del proyecto
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------
# Input PKL
# ---------------------------------
env_path = os.environ.get("DATA_PATH")

pkl_path = (
    Path(env_path)
    if env_path
    else BASE_DIR / "data" / "processed" / "FakeNewsCorpusSpanish" / "test.pkl"
)

df = pd.read_pickle(pkl_path, compression="infer")

print(type(df))
print(df.shape)
print(df.head().to_string())

# ---------------------------------
# Export PKL → CSV
# ---------------------------------
csv_path = (
    BASE_DIR / "data" / "exports" / "FakeNewsCorpusSpanish" / "test_dataset.csv"
)
csv_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(csv_path, index=False)

# ---------------------------------
# CSV → PKL
# ---------------------------------
pkl_from_csv_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "FakeNewsCorpusSpanish"
    / "test_dataset_from_csv.pkl"
)

df_csv = pd.read_csv(csv_path)
df_csv.to_pickle(pkl_from_csv_path, compression="infer")

print("Conversion from CSV to PKL completed.")
print(type(df_csv))
