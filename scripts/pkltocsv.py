from pathlib import Path
import os
import pandas as pd

# ---------------------------------
# Base directory del proyecto
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------
# Input PKL (from env or default)
# ---------------------------------
env_path = os.environ.get("DATA_PATH")

pkl_path = (
    Path(env_path)
    if env_path
    else BASE_DIR / "data" / "processed_by_model" / "FakeNewsCorpusSpanish" / "test.pkl"
)

if not pkl_path.exists():
    raise FileNotFoundError(f"PKL not found: {pkl_path}")

# ---------------------------------
# Load PKL
# ---------------------------------
df = pd.read_pickle(pkl_path, compression="infer")

print("Loaded PKL:")
print(" - Path :", pkl_path)
print(" - Shape:", df.shape)
print(" - Columns:", list(df.columns))

# ---------------------------------
# Output CSV
# ---------------------------------
csv_path = (
    BASE_DIR
    / "data"
    / "exports"
    / "FakeNewsCorpusSpanish"
    / f"{pkl_path.stem}.csv"
)

csv_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Save CSV
# ---------------------------------
df.to_csv(csv_path, index=False, encoding="utf-8")

print("CSV created:")
print(" - Path :", csv_path)
