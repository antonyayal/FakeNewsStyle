# scripts/peek_pkl_row.py
# =====================================================
# Quick peek at a single row of a PKL (columns, text preview,
# embedding column stats). For deeper inspection across all rows,
# use scripts/inspect_pkl.py instead.
# =====================================================

import argparse
import pickle
import pandas as pd
import numpy as np

# ============================================
# CLI
# ============================================

parser = argparse.ArgumentParser(description="Peek at a single row of a PKL file")
parser.add_argument("--pkl", required=True, help="Path to the PKL file")
parser.add_argument("--row", type=int, default=0, help="Row index to inspect (default: 0)")

args = parser.parse_args()

pkl_path = args.pkl
row = args.row

# ============================================
# LOAD
# ============================================

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"\nLoaded type: {type(data)}")

# ============================================
# DATAFRAME
# ============================================

if isinstance(data, pd.DataFrame):

    print("\nColumns:")
    print(data.columns.tolist())

    print(f"\nRow {row}:")
    print(data.iloc[row])

    text_cols = [
        "Text",
        "text",
        "text_raw",
        "text_xlmr",
        "Headline",
    ]

    for col in text_cols:
        if col in data.columns:
            print(f"\n===== {col} =====\n")
            print(data.iloc[row][col])
            break

    # ============================================
    # Detectar embeddings
    # ============================================

    emb_cols = []

    for c in data.columns:
        if str(c).lower().endswith("_emb"):
            emb_cols.append(c)

    if emb_cols:

        print("\n==============================")
        print("EMBEDDING ANALYSIS")
        print("==============================")

        for c in emb_cols:

            arr = np.array(data[c].tolist(), dtype=np.float32)

            print(f"\nEmbedding column: {c}")
            print(f"Shape: {arr.shape}")

            print(f"Global min:  {arr.min():.6f}")
            print(f"Global max:  {arr.max():.6f}")
            print(f"Global mean: {arr.mean():.6f}")
            print(f"Global std:  {arr.std():.6f}")

            print("\nFirst 20 dimensions:")
            print(arr[row][:20])

            print("\nRange per dimension (first 20):")

            max_show = min(20, arr.shape[1])

            for i in range(max_show):

                col_vals = arr[:, i]

                print(
                    f"Dim {i:4d} | "
                    f"min={col_vals.min():.6f} | "
                    f"max={col_vals.max():.6f} | "
                    f"mean={col_vals.mean():.6f} | "
                    f"std={col_vals.std():.6f}"
                )

# ============================================
# DICT
# ============================================

elif isinstance(data, dict):

    print("\nKeys:")
    print(data.keys())

    if "X" in data:

        X = np.asarray(data["X"], dtype=np.float32)

        print("\nShape:")
        print(X.shape)

        print(f"\nGlobal min:  {X.min():.6f}")
        print(f"Global max:  {X.max():.6f}")
        print(f"Global mean: {X.mean():.6f}")
        print(f"Global std:  {X.std():.6f}")

        print("\nFirst vector (first 20 dims):")
        print(X[0][:20])

        print("\nRange per dimension (first 20):")

        max_show = min(20, X.shape[1])

        for i in range(max_show):

            col_vals = X[:, i]

            print(
                f"Dim {i:4d} | "
                f"min={col_vals.min():.6f} | "
                f"max={col_vals.max():.6f} | "
                f"mean={col_vals.mean():.6f} | "
                f"std={col_vals.std():.6f}"
            )

else:
    print("Tipo no soportado")