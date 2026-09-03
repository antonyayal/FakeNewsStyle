# src/data/source_split_corpus.py
# -*- coding: utf-8 -*-
"""
Source-disjoint corpus packaging: pools the fixed train/development/test
corpus PKLs (data/01_corpus_pkl/*.pkl, produced by --prepare_corpus) and
repartitions them into N folds such that no news outlet (the `Source`
column) appears in more than one of a fold's train/val/test splits. Used by
main.py's --corpus_mode source_disjoint (see main.py's module docstring) and
by scripts/orchestrator_phase4.py.

Goal
----
src/data/kfold_corpus.py's stratified k-fold is stratified by label only --
the same outlet can (and per README.md's "Known Limitations & Caveats", does
43% of the time) appear in both train and test, so a classifier can partly
learn "this outlet always publishes Fake" instead of genuine style/semantic
signal. This module removes that specific axis of leakage: every `Source`
value is treated as an atomic group that lands entirely in train, val, or
test, never split across them.

This is a *stricter*, not a *better-balanced*, split. Outlet group sizes in
this corpus are highly uneven (a handful of outlets account for ~9% of all
rows each), so the resulting label balance and split sizes will not match
the clean ~70/30 of the original split or the plain stratified k-fold --
that's the accepted cost of removing the leakage at its source, not a bug.

Algorithm
---------
1. Pool train.pkl + development.pkl + test.pkl.
2. StratifiedGroupKFold(groups=Source) carves out each fold's test split --
   grouped so no test-fold Source leaks into that fold's train+val pool, and
   still label-stratified on top of the grouping (best effort; exact
   stratification isn't always achievable when group sizes are this
   uneven -- see the module docstring above).
3. GroupShuffleSplit(groups=Source) carves train/val out of what's left --
   grouped for the same reason: the plain (row-level) train_test_split
   kfold_corpus.py uses for this step would reintroduce leakage between
   train and val even with a source-disjoint test set.
4. Every fold is verified to have zero Source overlap between its three
   splits before being cached; violating that raises, it never fails
   silently.

Determinism
-----------
Same caching contract as kfold_corpus.py: all N folds for a given
(n_splits, split_seed) are generated together and cached on disk under
    {base_dir}/seed{split_seed}_n{n_splits}/fold{k}/{train,val,test}.pkl
so re-running with the same (n_splits, split_seed) reuses the cached folds,
and different (n_splits, split_seed) pairs never collide.

Usage (example)
---------------
from pathlib import Path
from src.data.source_split_corpus import ensure_source_disjoint_corpus, fold_dir

ensure_source_disjoint_corpus(
    source_dir=Path("data/01_corpus_pkl"),
    output_base_dir=Path("data/01_corpus_pkl_source_cv"),
    n_splits=5,
    split_seed=20260821,
)
train_pkl = fold_dir(Path("data/01_corpus_pkl_source_cv"), 5, 20260821, fold_index=0) / "train.pkl"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def fold_set_dir(output_base_dir: Path, n_splits: int, split_seed: int) -> Path:
    return Path(output_base_dir) / f"seed{split_seed}_n{n_splits}"


def fold_dir(output_base_dir: Path, n_splits: int, split_seed: int, fold_index: int) -> Path:
    return fold_set_dir(output_base_dir, n_splits, split_seed) / f"fold{fold_index}"


def _fold_set_is_complete(set_dir: Path, n_splits: int) -> bool:
    manifest = set_dir / "manifest.json"
    if not manifest.exists():
        return False
    for k in range(n_splits):
        fdir = set_dir / f"fold{k}"
        if not all((fdir / f"{s}.pkl").exists() for s in ["train", "val", "test"]):
            return False
    return True


_MISSING_SOURCE_PLACEHOLDER = "__missing_source__"


def _group_keys(df: pd.DataFrame, source_column: str) -> pd.Series:
    """Source values as grouping keys, with missing values collapsed onto a
    single placeholder so a handful of NaN rows don't each become their own
    singleton group (or, worse, crash sklearn: pandas' nullable/Arrow 'string'
    dtype -- what this corpus's Source column uses -- leaves NA as a bare
    float NaN even after .astype(str), and np.unique can't sort a mix of str
    and float). This is purely a grouping-key view; the underlying
    DataFrame's Source column (and whatever it saves to disk) is untouched."""
    return df[source_column].fillna(_MISSING_SOURCE_PLACEHOLDER).astype(str)


def _largest_source_share(df: pd.DataFrame, source_column: str) -> Dict[str, Any] | None:
    counts = _group_keys(df, source_column).value_counts()
    if len(counts) == 0:
        return None
    return {
        "source": counts.index[0],
        "count": int(counts.iloc[0]),
        "frac": float(counts.iloc[0] / len(df)),
    }


def ensure_source_disjoint_corpus(
    source_dir: Path,
    output_base_dir: Path,
    n_splits: int,
    split_seed: int,
    val_size: float = 0.2,
    source_column: str = "Source",
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Builds (or reuses, if already built and force=False) all n_splits
    folds for (n_splits, split_seed). Pools source_dir's train.pkl +
    development.pkl + test.pkl (data/01_corpus_pkl's schema: Id, Category,
    Topic, Source, Headline, Text, Link), runs a grouped (StratifiedGroupKFold
    by source_column) split to carve out each fold's test split, then a
    grouped (GroupShuffleSplit by source_column) split on each fold's
    remaining rows for train/val -- so no Source value ever appears in more
    than one of a fold's train/val/test.

    Returns the per-fold manifest (also cached at
    {fold_set_dir}/manifest.json): fold index, split sizes, source counts,
    label balance, and the largest single source's share of each split.

    Raises RuntimeError if a fold's train/val/test end up sharing any
    source_column value -- this should never happen by construction, but the
    whole point of this module is the disjointness guarantee, so it's
    checked rather than assumed.
    """
    set_dir = fold_set_dir(output_base_dir, n_splits, split_seed)

    if _fold_set_is_complete(set_dir, n_splits) and not force:
        with open(set_dir / "manifest.json", "r", encoding="utf-8") as f:
            return json.load(f)["folds"]

    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    source_dir = Path(source_dir)
    pool = pd.concat(
        [
            pd.read_pickle(source_dir / "train.pkl"),
            pd.read_pickle(source_dir / "development.pkl"),
            pd.read_pickle(source_dir / "test.pkl"),
        ],
        ignore_index=True,
    )
    y = pool["Category"].astype(str)
    groups = _group_keys(pool, source_column)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)

    folds_meta: List[Dict[str, Any]] = []
    for k, (rest_idx, test_idx) in enumerate(sgkf.split(pool, y, groups)):
        rest = pool.iloc[rest_idx]
        rest_groups = groups.iloc[rest_idx]
        test_df = pool.iloc[test_idx].reset_index(drop=True)

        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=split_seed + k)
        train_sub_idx, val_sub_idx = next(gss.split(rest, groups=rest_groups))
        train_df = rest.iloc[train_sub_idx].reset_index(drop=True)
        val_df = rest.iloc[val_sub_idx].reset_index(drop=True)

        train_sources = set(_group_keys(train_df, source_column))
        val_sources = set(_group_keys(val_df, source_column))
        test_sources = set(_group_keys(test_df, source_column))
        overlap = (train_sources & val_sources) | (train_sources & test_sources) | (val_sources & test_sources)
        if overlap:
            raise RuntimeError(
                f"Fold {k}: source-disjoint split invariant violated -- "
                f"{len(overlap)} '{source_column}' value(s) span more than one split: "
                f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
            )

        fdir = set_dir / f"fold{k}"
        fdir.mkdir(parents=True, exist_ok=True)
        train_df.to_pickle(fdir / "train.pkl")
        val_df.to_pickle(fdir / "val.pkl")
        test_df.to_pickle(fdir / "test.pkl")

        folds_meta.append({
            "fold": k,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_sources_train": len(train_sources),
            "n_sources_val": len(val_sources),
            "n_sources_test": len(test_sources),
            "label_balance_train": train_df["Category"].astype(str).value_counts().to_dict(),
            "label_balance_val": val_df["Category"].astype(str).value_counts().to_dict(),
            "label_balance_test": test_df["Category"].astype(str).value_counts().to_dict(),
            "largest_source_train": _largest_source_share(train_df, source_column),
            "largest_source_val": _largest_source_share(val_df, source_column),
            "largest_source_test": _largest_source_share(test_df, source_column),
        })

    manifest = {
        "n_splits": n_splits,
        "split_seed": split_seed,
        "val_size": val_size,
        "source_column": source_column,
        "folds": folds_meta,
    }
    with open(set_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return folds_meta
