# scripts/pca_latent_dim_suggester.py
# =====================================================
# Suggest latent dimensions for feature extractors using PCA
# Supports:
#   1) dict PKL with "X"
#   2) DataFrame with embedding/list columns: sem_emb, emo_probs, sent_probs, signals, *_emb
#   3) DataFrame with numeric columns already expanded
# =====================================================

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =====================================================
# IO
# =====================================================

def load_pkl(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# =====================================================
# Feature loading
# =====================================================

def infer_embedding_cols(df: pd.DataFrame) -> List[str]:
    priority = ["sem_emb", "emo_probs", "sent_probs", "signals", "style_emb", "ctx_emb"]
    cols: List[str] = []

    for c in priority:
        if c in df.columns:
            cols.append(c)

    for c in df.columns:
        if c not in cols and str(c).lower().endswith("_emb"):
            cols.append(c)

    valid: List[str] = []
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        v = s.iloc[0]
        if isinstance(v, (list, tuple, np.ndarray)):
            valid.append(c)
    return valid


def infer_numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {
        "Id", "id", "label", "Label", "Category", "category",
        "pooling", "model_name", "max_len", "l2_normalize",
        "emo_labels", "sent_labels", "signal_names",
        "device", "batch_size", "normalize_signals_by", "use_preprocess_tweet"
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def dataframe_to_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    info: Dict[str, Any] = {"source_type": "dataframe"}

    emb_cols = infer_embedding_cols(df)
    if emb_cols:
        blocks = []
        names = []
        for c in emb_cols:
            arr = np.array(df[c].tolist(), dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Column '{c}' is not 2D after conversion. Shape={arr.shape}")
            blocks.append(arr)
            names.extend([f"{c}_{i}" for i in range(arr.shape[1])])

        X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        info["mode"] = "embedding_columns"
        info["embedding_columns"] = emb_cols
        return X, names, info

    num_cols = infer_numeric_feature_cols(df)
    if num_cols:
        X = df[num_cols].to_numpy(dtype=np.float32)
        info["mode"] = "numeric_columns"
        info["numeric_columns"] = num_cols
        return X, num_cols, info

    raise ValueError("Could not infer feature columns from DataFrame.")


def dict_to_matrix(data: Dict[str, Any]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    if "X" not in data:
        raise ValueError(f"Dict PKL missing key 'X'. Keys: {list(data.keys())}")

    X = np.asarray(data["X"], dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    feature_names = data.get("feature_names")
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(X.shape[1])]

    info = {
        "source_type": "dict",
        "mode": "X_matrix",
        "keys": list(data.keys()),
    }
    return X, list(feature_names), info


def load_feature_matrix(path: Path) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    data = load_pkl(path)
    if isinstance(data, dict):
        return dict_to_matrix(data)
    if isinstance(data, pd.DataFrame):
        return dataframe_to_matrix(data)
    raise ValueError(f"Unsupported PKL object type: {type(data)}")


# =====================================================
# PCA analysis
# =====================================================

def dims_for_thresholds(cum_var: np.ndarray, thresholds: List[float]) -> Dict[float, int]:
    out: Dict[float, int] = {}
    for t in thresholds:
        idx = int(np.searchsorted(cum_var, t) + 1)
        out[t] = idx
    return out


def heuristic_latent_suggestion(input_dim: int, dims_map: Dict[float, int]) -> int:
    """
    Practical rule:
    - starts from the dimension for 90%
    - adjusts it to a "nice" value for networks
    - doesn't let it grow too much if the original space is small
    """
    d90 = dims_map[0.90]

    nice_values = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    best = min(nice_values, key=lambda x: abs(x - d90))

    if input_dim <= 24:
        best = min(best, max(4, input_dim // 2))
        best = max(best, 4)
    elif input_dim <= 64:
        best = min(best, 24)
        best = max(best, 8)
    elif input_dim <= 160:
        best = min(best, 32)
        best = max(best, 12)
    else:
        best = min(best, 128)
        best = max(best, 16)

    return int(best)


def run_pca_analysis(
    X: np.ndarray,
    standardize: bool = True,
) -> Dict[str, Any]:
    if X.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape={X.shape}")

    if len(X) < 2:
        raise ValueError("Need at least 2 samples for PCA.")

    X_proc = X.copy()

    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    max_comp = min(X_proc.shape[0], X_proc.shape[1])
    pca = PCA(n_components=max_comp, svd_solver="full")
    pca.fit(X_proc)

    evr = pca.explained_variance_ratio_
    cum_var = np.cumsum(evr)

    thresholds = [0.80, 0.90, 0.95, 0.99]
    dims_map = dims_for_thresholds(cum_var, thresholds)

    suggestion = heuristic_latent_suggestion(X.shape[1], dims_map)

    return {
        "input_dim": int(X.shape[1]),
        "num_samples": int(X.shape[0]),
        "explained_variance_ratio": evr,
        "cumulative_variance": cum_var,
        "dims_map": dims_map,
        "suggested_latent_dim": suggestion,
    }


# =====================================================
# Plot
# =====================================================

def save_variance_plot(
    cum_var: np.ndarray,
    dims_map: Dict[float, int],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(1, len(cum_var) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(x, cum_var, linewidth=2)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    for t, d in dims_map.items():
        plt.axhline(y=t, linestyle="--", alpha=0.5)
        plt.axvline(x=d, linestyle="--", alpha=0.5)
        plt.text(d, min(t + 0.02, 0.99), f"{int(t*100)}% -> {d}", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =====================================================
# Report
# =====================================================

def print_report(path: Path, info: Dict[str, Any], result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"PKL: {path}")
    print("=" * 80)

    print(f"Detected format: {info.get('source_type')} / {info.get('mode')}")
    if "embedding_columns" in info:
        print(f"Embedding columns: {info['embedding_columns']}")
    if "numeric_columns" in info:
        preview = info["numeric_columns"][:12]
        print(f"Numeric columns: {preview}{' ...' if len(info['numeric_columns']) > 12 else ''}")

    print(f"\nSamples: {result['num_samples']}")
    print(f"Original dimension: {result['input_dim']}")

    print("\nDimensions required by cumulative variance:")
    for t in [0.80, 0.90, 0.95, 0.99]:
        print(f"  {int(t*100):>2}% -> {result['dims_map'][t]}")

    print(f"\nBase suggestion for VAE: {result['suggested_latent_dim']}")

    d_in = result["input_dim"]
    d_lat = result["suggested_latent_dim"]
    compression = d_in / max(d_lat, 1)
    print(f"Approximate compression: {d_in} -> {d_lat} ({compression:.2f}x)")

    print("\nInterpretation:")
    print("  - 80%: aggressive compression")
    print("  - 90%: good starting point")
    print("  - 95%: conservative compression")
    print("  - 99%: near-lossless, but not very compact")


# =====================================================
# Main
# =====================================================

def analyze_one_file(path: Path, output_dir: Path, standardize: bool = True) -> None:
    X, feature_names, info = load_feature_matrix(path)
    result = run_pca_analysis(X, standardize=standardize)
    print_report(path, info, result)

    stem = path.stem
    plot_path = output_dir / f"{stem}_pca_variance.png"
    save_variance_plot(
        cum_var=result["cumulative_variance"],
        dims_map=result["dims_map"],
        out_path=plot_path,
        title=f"PCA variance - {stem}",
    )
    print(f"\nPlot saved to: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest latent dimensions for feature PKLs using PCA")
    parser.add_argument(
        "--pkl",
        type=str,
        nargs="+",
        required=True,
        help="One or more PKL paths to analyze",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/pca_latent_dims",
        help="Directory to save variance plots",
    )
    parser.add_argument(
        "--no_standardize",
        type=int,
        default=0,
        help="Disable standardization before PCA (0=no, 1=yes)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    standardize = (args.no_standardize == 0)

    for p in args.pkl:
        analyze_one_file(Path(p), output_dir=output_dir, standardize=standardize)


if __name__ == "__main__":
    main()