# scripts/inspect_pkl.py
# =====================================================
# Generic PRO Feature Inspector for FakeNewsStyle
# Supports:
#   1) dict PKL: {"X": np.ndarray, "feature_names": [...], ...}
#   2) pandas.DataFrame with embedding column(s): sem_emb / *_emb
#   3) pandas.DataFrame with expanded numeric feature columns
# =====================================================

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================================================
# Utils
# =====================================================

def load_pkl(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_repr(x: Any, max_len: int = 300) -> str:
    s = repr(x)
    return s if len(s) <= max_len else s[:max_len] + "..."


def infer_embedding_cols(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []

    # Explicit priority
    priority = ["sem_emb", "emo_emb", "style_emb", "ctx_emb"]
    for c in priority:
        if c in df.columns:
            candidates.append(c)

    # Any *_emb
    for c in df.columns:
        if c not in candidates and str(c).lower().endswith("_emb"):
            candidates.append(c)

    # Verification: list/vector column
    final_cols: List[str] = []
    for c in candidates:
        series = df[c].dropna()
        if len(series) == 0:
            continue
        v = series.iloc[0]
        if isinstance(v, (list, tuple, np.ndarray)):
            final_cols.append(c)

    return final_cols


def infer_numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"Id", "id", "label", "Label", "Category", "category", "pooling", "model_name", "max_len", "l2_normalize"}
    num_cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


def build_matrix_from_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Returns:
      X: np.ndarray [N, D]
      feature_names: list[str]
      aux_info: dict
    """
    aux_info: Dict[str, Any] = {
        "format": "dataframe",
        "embedding_cols": [],
        "numeric_feature_cols": [],
    }

    # Case 1: DataFrame with one or more embedding columns (lists)
    emb_cols = infer_embedding_cols(df)
    if emb_cols:
        aux_info["embedding_cols"] = emb_cols

        blocks = []
        names = []

        for c in emb_cols:
            arr = np.array(df[c].tolist(), dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Embedding column '{c}' is not 2D after conversion. Got shape={arr.shape}")
            blocks.append(arr)
            names.extend([f"{c}_{i}" for i in range(arr.shape[1])])

        X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        return X, names, aux_info

    # Case 2: DataFrame with already-expanded numeric columns
    num_cols = infer_numeric_feature_cols(df)
    if num_cols:
        aux_info["numeric_feature_cols"] = num_cols
        X = df[num_cols].to_numpy(dtype=np.float32)
        return X, list(num_cols), aux_info

    raise ValueError(
        "Could not infer features from DataFrame. "
        "No embedding columns (*_emb) or numeric feature columns found."
    )


def build_matrix_from_dict(data: Dict[str, Any]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Soporta:
      - {"X": ..., "feature_names": ...}
      - {"X": ...}
    """
    aux_info: Dict[str, Any] = {
        "format": "dict",
        "keys": list(data.keys()),
    }

    if "X" not in data:
        raise ValueError(f"Dict PKL does not contain key 'X'. Keys found: {list(data.keys())}")

    X = np.asarray(data["X"], dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    feature_names = data.get("feature_names")
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(X.shape[1])]

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != X.shape[1] ({X.shape[1]})."
        )

    return X, list(feature_names), aux_info


def build_feature_matrix(data: Any) -> Tuple[np.ndarray, List[str], Dict[str, Any], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Retorna:
      X, feature_names, aux_info, df_if_any, dict_if_any
    """
    if isinstance(data, dict):
        X, feature_names, aux_info = build_matrix_from_dict(data)
        return X, feature_names, aux_info, None, data

    if isinstance(data, pd.DataFrame):
        X, feature_names, aux_info = build_matrix_from_dataframe(data)
        return X, feature_names, aux_info, data, None

    raise ValueError(f"Unsupported PKL object type: {type(data)}")


def summarize_labels(df: Optional[pd.DataFrame], dct: Optional[Dict[str, Any]]) -> None:
    print_header("LABEL DISTRIBUTION")

    if df is not None:
        label_col = None
        for c in ["label", "Label", "Category", "category", "y", "target"]:
            if c in df.columns:
                label_col = c
                break

        if label_col is None:
            print("No label column found in DataFrame.")
            return

        print(f"Label column: {label_col}")
        vc = df[label_col].value_counts(dropna=False)
        print(vc.to_string())
        return

    if dct is not None:
        if "labels" in dct:
            vals = dct["labels"]
        elif "y" in dct:
            vals = dct["y"]
        elif "label" in dct:
            vals = dct["label"]
        else:
            print("No labels found in dict.")
            return

        vals = list(vals)
        s = pd.Series(vals)
        print(s.value_counts(dropna=False).to_string())
        return

    print("No label information found.")


def summarize_metadata(df: Optional[pd.DataFrame], dct: Optional[Dict[str, Any]], aux_info: Dict[str, Any]) -> None:
    print_header("METADATA / STRUCTURE")

    print(f"Detected format: {aux_info.get('format')}")

    if df is not None:
        print(f"Columns ({len(df.columns)}):")
        for c in df.columns:
            print(f"  - {c} (dtype={df[c].dtype})")

        emb_cols = aux_info.get("embedding_cols", [])
        num_cols = aux_info.get("numeric_feature_cols", [])

        if emb_cols:
            print(f"\nEmbedding columns: {emb_cols}")
            for c in emb_cols:
                sample = df[c].dropna().iloc[0] if len(df[c].dropna()) > 0 else None
                dim = len(sample) if sample is not None else "unknown"
                print(f"  - {c}: dim={dim}")

        if num_cols:
            print(f"\nNumeric feature columns ({len(num_cols)}):")
            preview = num_cols[:30]
            for c in preview:
                print(f"  - {c}")
            if len(num_cols) > len(preview):
                print(f"  ... +{len(num_cols) - len(preview)} more")

        for c in ["pooling", "model_name", "max_len", "l2_normalize"]:
            if c in df.columns:
                uniq = df[c].dropna().unique().tolist()
                print(f"\n{c}: {uniq[:10]}")

        return

    if dct is not None:
        print(f"Keys ({len(dct.keys())}): {list(dct.keys())}")

        if "meta" in dct:
            print("\nmeta:")
            meta = dct["meta"]
            if isinstance(meta, dict):
                for k, v in meta.items():
                    print(f"  - {k}: {safe_repr(v)}")
            else:
                print(safe_repr(meta))

        if "feature_names" in dct:
            fns = dct["feature_names"]
            print(f"\nfeature_names count: {len(fns)}")
            preview = fns[:20]
            for f in preview:
                print(f"  - {f}")
            if len(fns) > len(preview):
                print(f"  ... +{len(fns) - len(preview)} more")


def summarize_basic_stats(X: np.ndarray) -> None:
    print_header("GLOBAL STATS")

    print(f"Shape: {X.shape}")
    print(f"Num samples: {X.shape[0]}")
    print(f"Feature dim: {X.shape[1]}")
    print(f"Min:         {X.min():.6f}")
    print(f"Max:         {X.max():.6f}")
    print(f"Mean:        {X.mean():.6f}")
    print(f"Std:         {X.std():.6f}")
    print(f"Mean |x|:    {np.mean(np.abs(X)):.6f}")

    zero_ratio = float(np.mean(X == 0.0))
    print(f"Zero ratio:  {zero_ratio:.6f}")

    l2_norms = np.linalg.norm(X, axis=1)
    print(f"Row L2 min:  {l2_norms.min():.6f}")
    print(f"Row L2 max:  {l2_norms.max():.6f}")
    print(f"Row L2 mean: {l2_norms.mean():.6f}")


def summarize_first_sample(X: np.ndarray, feature_names: List[str], show_full_vector: int = 0) -> None:
    print_header("FIRST SAMPLE")

    if X.shape[0] == 0:
        print("No samples.")
        return

    row = X[0]

    if show_full_vector:
        print("Full vector:")
        print(row)

        print("\nFeature-value pairs:")
        for i, v in enumerate(row):
            print(f"{i:4d} | {feature_names[i]:40s} | {v:.6f}")
    else:
        top = min(20, len(row))
        print(f"First {top} dims:")
        print(row[:top])

        print("\nFirst feature-value pairs:")
        for i in range(top):
            print(f"{i:4d} | {feature_names[i]:40s} | {row[i]:.6f}")


def summarize_top_variance(X: np.ndarray, feature_names: List[str], k: int = 15) -> None:
    print_header("TOP VARIANCE FEATURES")

    variances = X.var(axis=0)
    top_idx = np.argsort(-variances)[:k]

    for i in top_idx:
        print(
            f"{feature_names[i]:40s} | "
            f"var={variances[i]:.6f} | "
            f"min={X[:, i].min():.6f} | "
            f"max={X[:, i].max():.6f} | "
            f"mean={X[:, i].mean():.6f}"
        )


def summarize_outliers(X: np.ndarray, feature_names: List[str], threshold: float = 1.5) -> None:
    print_header("OUTLIERS CHECK")

    outliers = np.where(np.abs(X) > threshold)
    n_out = len(outliers[0])

    print(f"Threshold: |x| > {threshold}")
    print(f"Num outliers: {n_out}")

    if n_out > 0:
        print("\nSample outliers:")
        for i in range(min(20, n_out)):
            r = outliers[0][i]
            c = outliers[1][i]
            print(f"  sample={r}, feature={feature_names[c]}, value={X[r, c]:.6f}")


def summarize_per_dim_ranges(X: np.ndarray, feature_names: List[str], max_show: int = 20) -> None:
    print_header("PER-DIMENSION RANGES")

    show = min(max_show, X.shape[1])
    for i in range(show):
        col = X[:, i]
        print(
            f"{i:4d} | {feature_names[i]:40s} | "
            f"min={col.min():.6f} | max={col.max():.6f} | "
            f"mean={col.mean():.6f} | std={col.std():.6f}"
        )

    if X.shape[1] > show:
        print(f"... +{X.shape[1] - show} more dimensions")


def summarize_feature_groups(X: np.ndarray, feature_names: List[str]) -> None:
    print_header("FEATURE GROUPS (HEURISTIC)")

    groups = {
        "source": [i for i, f in enumerate(feature_names) if "source_emb" in f],
        "domain": [i for i, f in enumerate(feature_names) if "domain_emb" in f],
        "topic": [i for i, f in enumerate(feature_names) if "topic_emb" in f],
        "author": [i for i, f in enumerate(feature_names) if "author_emb" in f],
        "age": [i for i, f in enumerate(feature_names) if "age" in f],
        "flags": [i for i, f in enumerate(feature_names) if "has_" in f],
        "semantic": [i for i, f in enumerate(feature_names) if f.startswith("sem_emb_")],
        "emotion": [i for i, f in enumerate(feature_names) if f.startswith("emo_emb_") or "emotion" in f.lower()],
        "style": [i for i, f in enumerate(feature_names) if f.startswith("style_emb_") or "sty_" in f.lower() or "style_" in f.lower()],
        "context": [i for i, f in enumerate(feature_names) if f.startswith("ctx_")],
    }

    found_any = False
    for g, idxs in groups.items():
        if not idxs:
            continue
        found_any = True
        sub = X[:, idxs]
        print(f"\n[{g.upper()}]")
        print(f"  dims: {len(idxs)}")
        print(f"  min:  {sub.min():.6f}")
        print(f"  max:  {sub.max():.6f}")
        print(f"  mean: {sub.mean():.6f}")
        print(f"  std:  {sub.std():.6f}")

    if not found_any:
        print("No heuristic groups matched these feature names.")


def summarize_flags_if_any(X: np.ndarray, feature_names: List[str]) -> None:
    flag_idxs = [i for i, f in enumerate(feature_names) if "has_" in f]
    if not flag_idxs:
        return

    print_header("FLAGS DISTRIBUTION")

    for i in flag_idxs:
        vals = X[:, i]
        unique, counts = np.unique(vals, return_counts=True)
        dist = {float(u): int(c) for u, c in zip(unique, counts)}
        print(f"{feature_names[i]} -> {dist}")


def summarize_age_if_any(X: np.ndarray, feature_names: List[str]) -> None:
    age_idxs = [i for i, f in enumerate(feature_names) if "age" in f]
    if not age_idxs:
        return

    print_header("AGE ANALYSIS")

    for i in age_idxs:
        vals = X[:, i]
        print(f"{feature_names[i]}:")
        print(f"  min:  {vals.min():.6f}")
        print(f"  max:  {vals.max():.6f}")
        print(f"  mean: {vals.mean():.6f}")
        print(f"  std:  {vals.std():.6f}")
        print(f"  first 10: {vals[:10]}")


# =====================================================
# Main inspector
# =====================================================

def inspect_features(pkl_path: Path, show_full_vector: int = 0, outlier_threshold: float = 1.5) -> None:
    np.set_printoptions(
        precision=6,
        suppress=True,
        threshold=np.inf,
        linewidth=200
    )

    data = load_pkl(pkl_path)

    print_header("FILE INFO")
    print(f"Path: {pkl_path.resolve()}")
    print(f"Raw object type: {type(data)}")

    X, feature_names, aux_info, df, dct = build_feature_matrix(data)

    summarize_metadata(df, dct, aux_info)
    summarize_basic_stats(X)
    summarize_feature_groups(X, feature_names)
    summarize_first_sample(X, feature_names, show_full_vector=show_full_vector)
    summarize_per_dim_ranges(X, feature_names, max_show=20)
    summarize_top_variance(X, feature_names, k=15)
    summarize_flags_if_any(X, feature_names)
    summarize_age_if_any(X, feature_names)
    summarize_outliers(X, feature_names, threshold=outlier_threshold)
    summarize_labels(df, dct)

    print_header("DONE")


# =====================================================
# CLI
# =====================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect feature PKL (generic PRO)")
    parser.add_argument("--pkl", required=True, type=str, help="Path to feature PKL")
    parser.add_argument("--full", type=int, default=0, help="Print full first vector and all feature-value pairs (0/1)")
    parser.add_argument("--outlier_threshold", type=float, default=1.5, help="Threshold for outlier report using |x| > threshold")

    args = parser.parse_args()

    inspect_features(
        pkl_path=Path(args.pkl),
        show_full_vector=int(args.full),
        outlier_threshold=float(args.outlier_threshold),
    )


if __name__ == "__main__":
    main()