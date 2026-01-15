# scripts/inspect_pkl.py
# Uso:
#   python scripts/inspect_pkl.py --pkl data/processed/FakeNewsCorpusSpanish/train.pkl --n 3
#
# Qué hace:
# - Carga el .pkl
# - Detecta si es pandas.DataFrame o lista/dict
# - Imprime: tipo, tamaño, columnas/keys, muestra de registros
# - Intenta adivinar columnas de texto y de label

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


TEXT_CANDIDATES = ("text", "content", "body", "article", "news", "title", "headline", "message")
LABEL_CANDIDATES = ("label", "labels", "y", "target", "class", "category")


def load_pkl(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"PKL not found: {path}")
    return pd.read_pickle(path)


def summarize_dataframe(df: pd.DataFrame, n: int) -> None:
    print("== PKL is a pandas.DataFrame ==")
    print(f"shape: {df.shape}")
    print("columns:")
    for c in df.columns:
        print(f"  - {c} (dtype={df[c].dtype})")

    # Try to guess text/label columns
    text_cols = [c for c in df.columns if str(c).lower() in TEXT_CANDIDATES]
    label_cols = [c for c in df.columns if str(c).lower() in LABEL_CANDIDATES]

    # Also try heuristic: object columns with long strings
    if not text_cols:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        scored: List[Tuple[str, float]] = []
        for c in obj_cols:
            s = df[c].dropna().astype(str)
            if len(s) == 0:
                continue
            avg_len = s.str.len().head(5000).mean()
            scored.append((c, float(avg_len)))
        scored.sort(key=lambda x: x[1], reverse=True)
        text_cols = [c for c, _ in scored[:3]]

    if not label_cols:
        # Heuristic: low-cardinality non-float columns
        cand: List[str] = []
        for c in df.columns:
            if c in text_cols:
                continue
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= 50:
                cand.append(c)
        label_cols = cand[:3]

    print("\npossible text columns:", text_cols if text_cols else "none")
    print("possible label columns:", label_cols if label_cols else "none")

    print("\nhead sample:")
    # Show only a few columns to avoid dumping huge text
    cols_to_show = list(dict.fromkeys((text_cols + label_cols + list(df.columns[:5]))))[:8]
    print(df[cols_to_show].head(n).to_string(index=False))

    # If we have a likely label column, show distribution
    if label_cols:
        col = label_cols[0]
        print(f"\nlabel distribution (top 20) for '{col}':")
        print(df[col].value_counts(dropna=False).head(20).to_string())


def summarize_list_of_dicts(items: List[Dict[str, Any]], n: int) -> None:
    print("== PKL is a list[dict] ==")
    print(f"num_items: {len(items)}")

    # keys union from first N items
    keys = set()
    for it in items[: max(n, 50)]:
        keys.update(it.keys())
    keys = sorted(keys)
    print("keys (from first items):")
    for k in keys:
        print(f"  - {k}")

    # Guess text/label keys
    low_keys = {k.lower(): k for k in keys}
    text_keys = [low_keys[k] for k in low_keys if k in TEXT_CANDIDATES]
    label_keys = [low_keys[k] for k in low_keys if k in LABEL_CANDIDATES]

    if not text_keys:
        # heuristic: string fields with long average length
        scored: List[Tuple[str, float]] = []
        for k in keys:
            vals = [str(it.get(k, "")) for it in items[: min(len(items), 2000)] if it.get(k) is not None]
            if not vals:
                continue
            avg_len = sum(len(v) for v in vals) / max(len(vals), 1)
            scored.append((k, float(avg_len)))
        scored.sort(key=lambda x: x[1], reverse=True)
        text_keys = [k for k, _ in scored[:3]]

    if not label_keys:
        # heuristic: low-cardinality keys
        cand: List[str] = []
        for k in keys:
            if k in text_keys:
                continue
            vals = [it.get(k) for it in items[: min(len(items), 5000)]]
            uniq = {v for v in vals if v is not None}
            if 1 < len(uniq) <= 50:
                cand.append(k)
        label_keys = cand[:3]

    print("\npossible text keys:", text_keys if text_keys else "none")
    print("possible label keys:", label_keys if label_keys else "none")

    print(f"\nshowing {min(n, len(items))} sample items (trimmed):")
    for i, it in enumerate(items[:n]):
        out = {}
        for k in list(dict.fromkeys(text_keys + label_keys))[:6]:
            v = it.get(k)
            if isinstance(v, str) and len(v) > 240:
                v = v[:240] + "…"
            out[k] = v
        if not out:
            # fallback: show first 6 keys
            for k in keys[:6]:
                v = it.get(k)
                if isinstance(v, str) and len(v) > 240:
                    v = v[:240] + "…"
                out[k] = v
        print(f"\n--- item {i} ---")
        for k, v in out.items():
            print(f"{k}: {v!r}")

    # Label distribution
    if label_keys:
        lk = label_keys[0]
        counts: Dict[Any, int] = {}
        for it in items:
            v = it.get(lk)
            counts[v] = counts.get(v, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
        print(f"\nlabel distribution (top 20) for '{lk}':")
        for v, c in top:
            print(f"  {v!r}: {c}")


def summarize_dict(obj: Dict[str, Any], n: int) -> None:
    print("== PKL is a dict ==")
    print(f"keys: {list(obj.keys())[:50]}")
    # If dict contains a dataframe-like structure
    for k in list(obj.keys())[:10]:
        v = obj[k]
        print(f"\n--- key '{k}' type={type(v)} ---")
        if isinstance(v, pd.DataFrame):
            summarize_dataframe(v, n)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            summarize_list_of_dicts(v, n)
        else:
            # Print a short preview
            s = repr(v)
            if len(s) > 500:
                s = s[:500] + "…"
            print(s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a .pkl dataset structure")
    parser.add_argument("--pkl", required=True, type=str, help="Path to .pkl file")
    parser.add_argument("--n", default=3, type=int, help="Number of samples to print")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    obj = load_pkl(pkl_path)

    print(f"pkl_path: {pkl_path.resolve()}")
    print(f"type: {type(obj)}")

    if isinstance(obj, pd.DataFrame):
        summarize_dataframe(obj, args.n)
    elif isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        summarize_list_of_dicts(obj, args.n)
    elif isinstance(obj, dict):
        summarize_dict(obj, args.n)
    else:
        # Unknown structure
        print("Unknown PKL structure. repr (trimmed):")
        s = repr(obj)
        if len(s) > 2000:
            s = s[:2000] + "…"
        print(s)


if __name__ == "__main__":
    main()
