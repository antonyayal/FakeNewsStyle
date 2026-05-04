# src/features/merge_raw_features_for_kan.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMNS = ["label", "labels", "y", "Category"]

IGNORE_KEYS = [
    "label",
    "labels",
    "y",
    "Category",
    "id",
    "ids",
    "metadata",
    "feature_names",
    "columns",
]


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df.astype(np.float32)


def expand_vector_column(series: pd.Series, col_name: str) -> pd.DataFrame:
    values = series.tolist()

    if len(values) == 0:
        return pd.DataFrame()

    first_valid = None
    for v in values:
        if v is not None:
            first_valid = v
            break

    if first_valid is None:
        return pd.DataFrame({col_name: [0.0] * len(values)})

    # Case 1: scalar numeric or scalar categorical
    if not isinstance(first_valid, (list, tuple, np.ndarray)):
        numeric_series = pd.to_numeric(series, errors="coerce")

        # Numeric scalar
        if numeric_series.notna().sum() > 0:
            return pd.DataFrame({
                col_name: numeric_series.fillna(0.0).astype(float)
            })

        # Categorical scalar -> one-hot
        clean = series.astype(str).str.strip().str.lower()
        clean = clean.replace({"": "unknown", "none": "unknown", "nan": "unknown"})

        return pd.get_dummies(
            clean,
            prefix=col_name,
            dtype=float
        ).reset_index(drop=True)

    # Case 2: vector/list numeric
    clean_values = []
    lengths = []
    numeric_possible = True

    for v in values:
        if v is None:
            arr = []
        else:
            try:
                arr = np.asarray(v, dtype=float).flatten().tolist()
            except Exception:
                numeric_possible = False
                break

        clean_values.append(arr)
        lengths.append(len(arr))

    if numeric_possible:
        max_len = max(lengths) if lengths else 0

        padded = [
            arr + [0.0] * (max_len - len(arr))
            for arr in clean_values
        ]

        return pd.DataFrame(
            padded,
            columns=[f"{col_name}_{i}" for i in range(max_len)]
        )

    # Case 3: vector/list categorical -> multi-hot
    categorical_rows = []
    all_categories = set()

    for v in values:
        if v is None:
            items = []
        elif isinstance(v, (list, tuple, np.ndarray)):
            items = [str(x).strip().lower() for x in v]
        else:
            items = [str(v).strip().lower()]

        items = [x for x in items if x not in ["", "none", "nan"]]

        categorical_rows.append(items)
        all_categories.update(items)

    all_categories = sorted(all_categories)

    if not all_categories:
        return pd.DataFrame({f"{col_name}_unknown": [1.0] * len(values)})

    data = []

    for items in categorical_rows:
        row = {
            f"{col_name}_{cat}": 1.0 if cat in items else 0.0
            for cat in all_categories
        }
        data.append(row)

    return pd.DataFrame(data).reset_index(drop=True)


def object_to_numeric_df(obj, prefix: str):
    """
    Converts feature PKLs into numeric DataFrames for KAN.

    Supports:
    - DataFrame with scalar numeric columns
    - DataFrame with vector/list numeric columns
    - DataFrame with categorical scalar columns
    - DataFrame with categorical vector/list columns
    - dict with arrays

    Returns:
    - numeric feature DataFrame
    - optional label Series
    """

    # -------------------------------------------------
    # Case 1: pandas DataFrame
    # -------------------------------------------------
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

        label = None

        for label_col in LABEL_COLUMNS:
            if label_col in df.columns:
                label = df[label_col].reset_index(drop=True)
                df = df.drop(columns=[label_col])
                break

        expanded_parts = []

        for col in df.columns:
            if col in IGNORE_KEYS:
                continue

            series = df[col].reset_index(drop=True)

            # Numeric scalar column
            if pd.api.types.is_numeric_dtype(series):
                expanded_parts.append(
                    pd.DataFrame({
                        f"{prefix}_{col}": series.astype(float).fillna(0.0)
                    })
                )
                continue

            # Object column: may be vector/list/categorical
            expanded = expand_vector_column(series, f"{prefix}_{col}")

            if expanded is not None and not expanded.empty:
                expanded_parts.append(expanded)

        if not expanded_parts:
            raise ValueError(
                f"No numeric/vector/categorical features found for prefix: {prefix}"
            )

        out_df = pd.concat(expanded_parts, axis=1).reset_index(drop=True)
        out_df = clean_numeric_df(out_df)

        return out_df, label

    # -------------------------------------------------
    # Case 2: dict
    # -------------------------------------------------
    if isinstance(obj, dict):
        label = None

        for label_key in LABEL_COLUMNS:
            if label_key in obj:
                label = pd.Series(obj[label_key]).reset_index(drop=True)
                break

        expanded_parts = []

        for k, v in obj.items():
            if k in IGNORE_KEYS:
                continue

            arr = np.asarray(v, dtype=object)

            # Shape: (N,)
            if arr.ndim == 1:
                series = pd.Series(arr)
                expanded = expand_vector_column(series, f"{prefix}_{k}")

                if expanded is not None and not expanded.empty:
                    expanded_parts.append(expanded)

            # Shape: (N, D)
            elif arr.ndim == 2:
                try:
                    arr_float = arr.astype(float)

                    expanded_parts.append(
                        pd.DataFrame(
                            arr_float,
                            columns=[
                                f"{prefix}_{k}_{i}"
                                for i in range(arr_float.shape[1])
                            ],
                        )
                    )

                except Exception:
                    series = pd.Series(list(arr))
                    expanded = expand_vector_column(series, f"{prefix}_{k}")

                    if expanded is not None and not expanded.empty:
                        expanded_parts.append(expanded)

            # Shape: higher dimensions
            else:
                try:
                    flat = np.asarray([
                        np.asarray(x, dtype=float).flatten()
                        for x in arr
                    ])

                    expanded_parts.append(
                        pd.DataFrame(
                            flat,
                            columns=[
                                f"{prefix}_{k}_{i}"
                                for i in range(flat.shape[1])
                            ],
                        )
                    )

                except Exception:
                    continue

        if not expanded_parts:
            raise ValueError(
                f"No numeric/vector/categorical features found for prefix: {prefix}"
            )

        out_df = pd.concat(expanded_parts, axis=1).reset_index(drop=True)
        out_df = clean_numeric_df(out_df)

        return out_df, label

    raise ValueError(f"Unsupported feature PKL format: {type(obj)}")


def normalize_label_series(label: pd.Series) -> pd.Series:
    """
    Standard:
    1 = Fake
    0 = True / Real
    """

    def convert(v):
        if isinstance(v, (bool, np.bool_)):
            return 0 if bool(v) else 1

        s = str(v).strip().lower()

        if s in ["fake", "false", "falsa", "0"]:
            return 1

        if s in ["true", "real", "verdadera", "verdadero", "1"]:
            return 0

        return v

    return label.map(convert)


def find_feature_file(feature_dir: Path, split: str, feature_name: str) -> Path:
    candidates = [
        feature_dir / f"{split}_{feature_name}.pkl",
        feature_dir / f"{split}.pkl",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No PKL found for feature '{feature_name}' and split '{split}'. "
        f"Tried: {candidates}"
    )


def align_columns(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Ensures one-hot columns are aligned across splits.
    """

    all_columns = sorted(
        set(train_df.columns)
        | set(val_df.columns)
        | set(test_df.columns)
    )

    train_df = train_df.reindex(columns=all_columns, fill_value=0.0)
    val_df = val_df.reindex(columns=all_columns, fill_value=0.0)
    test_df = test_df.reindex(columns=all_columns, fill_value=0.0)

    train_df = clean_numeric_df(train_df)
    val_df = clean_numeric_df(val_df)
    test_df = clean_numeric_df(test_df)

    return train_df, val_df, test_df


def merge_split(
    split: str,
    feature_dirs: dict[str, Path],
    output_dir: Path,
) -> pd.DataFrame:
    dfs = []
    labels = None

    print("=" * 80)
    print(f"Merging split: {split}")
    print("=" * 80)

    for feature_name, feature_dir in feature_dirs.items():
        pkl_path = find_feature_file(feature_dir, split, feature_name)

        obj = load_pkl(pkl_path)
        df, current_labels = object_to_numeric_df(obj, feature_name)

        if current_labels is not None:
            current_labels = normalize_label_series(
                current_labels.reset_index(drop=True)
            )

            if labels is None:
                labels = current_labels
            else:
                if len(labels) != len(current_labels):
                    raise ValueError(
                        f"Label length mismatch in split '{split}' "
                        f"for feature '{feature_name}'. "
                        f"Expected {len(labels)}, got {len(current_labels)}"
                    )

        dfs.append(df)

        print(f"{split} | {feature_name}: {df.shape} | source={pkl_path}")

    merged_df = pd.concat(dfs, axis=1)

    if labels is None:
        raise ValueError(f"No labels found for split '{split}'")

    if len(merged_df) != len(labels):
        raise ValueError(
            f"Feature/label length mismatch for split '{split}'. "
            f"Features={len(merged_df)}, labels={len(labels)}"
        )

    merged_df = clean_numeric_df(merged_df)
    merged_df["label"] = labels.values

    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{split}.pkl"
    merged_df.to_pickle(out_path)

    print(
        f"Saved raw merged features {split}: {out_path} | "
        f"samples={len(merged_df)} | dims={merged_df.shape[1]}"
    )

    return merged_df


def merge_all_splits(
    feature_dirs: dict[str, Path],
    output_dir: Path,
):
    """
    Merges train/val/test and aligns one-hot columns across splits.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    split_dfs = {}

    for split in ["train", "val", "test"]:
        split_dfs[split] = merge_split(split, feature_dirs, output_dir)

    train_y = split_dfs["train"]["label"]
    val_y = split_dfs["val"]["label"]
    test_y = split_dfs["test"]["label"]

    train_x = split_dfs["train"].drop(columns=["label"])
    val_x = split_dfs["val"].drop(columns=["label"])
    test_x = split_dfs["test"].drop(columns=["label"])

    train_x, val_x, test_x = align_columns(train_x, val_x, test_x)

    split_dfs["train"] = train_x
    split_dfs["train"]["label"] = train_y.values

    split_dfs["val"] = val_x
    split_dfs["val"]["label"] = val_y.values

    split_dfs["test"] = test_x
    split_dfs["test"]["label"] = test_y.values

    for split, df in split_dfs.items():
        out_path = output_dir / f"{split}.pkl"
        df.to_pickle(out_path)

        X = df.drop(columns=["label"])
        nan_count = int(np.isnan(X.values).sum())
        inf_count = int(np.isinf(X.values).sum())

        print(
            f"Aligned and saved {split}: {out_path} | "
            f"shape={df.shape} | NaN={nan_count} | Inf={inf_count}"
        )

    print("Raw feature merge completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge raw semantic/emotion/style/context features for KAN baseline"
    )

    parser.add_argument("--semantic_dir", default="data/features/semantic")
    parser.add_argument("--emotion_dir", default="data/features/emotion")
    parser.add_argument("--style_dir", default="data/features/style")
    parser.add_argument("--context_dir", default="data/features/context")
    parser.add_argument("--output_dir", default="data/features_merged_for_kan")

    args = parser.parse_args()

    feature_dirs = {
        "semantic": Path(args.semantic_dir),
        "emotion": Path(args.emotion_dir),
        "style": Path(args.style_dir),
        "context": Path(args.context_dir),
    }

    output_dir = Path(args.output_dir)

    merge_all_splits(feature_dirs, output_dir)


if __name__ == "__main__":
    main()