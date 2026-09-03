# src/data/kfold_corpus.py
# -*- coding: utf-8 -*-
"""
K-fold corpus packaging: pools the fixed train/development/test corpus PKLs
(data/01_corpus_pkl/*.pkl, produced by --prepare_corpus) and repartitions
them into N stratified folds (by Category), each with its own train/val/test
split. Used by main.py's --corpus_mode kfold (see main.py's module
docstring) and by scripts/run_cv_packages.py.

Goal
----
Let the rest of the pipeline (preprocess/extract/VAE/KAN) run unmodified
against a fold's train/val/test PKLs exactly as it would against the
original fixed split -- the only difference is which rows end up in which
split. This is a *standard* stratified k-fold (stratified by label only),
not source-disjoint -- see README.md's "Known Limitations & Caveats" section
for why it doesn't correct the Source/Domain leakage documented there.

Determinism
-----------
All N folds for a given (n_splits, split_seed) are generated together in one
call and cached on disk under:
    {base_dir}/seed{split_seed}_n{n_splits}/fold{k}/{train,val,test}.pkl
so every fold index reuses the exact same partition across repeated runs
(and across main.py invocations run one fold at a time), and different
(n_splits, split_seed) pairs never collide.

Usage (example)
---------------
from pathlib import Path
from src.data.kfold_corpus import ensure_kfold_corpus, fold_dir

ensure_kfold_corpus(
    source_dir=Path("data/01_corpus_pkl"),
    output_base_dir=Path("data/01_corpus_pkl_cv"),
    n_splits=5,
    split_seed=20260820,
)
train_pkl = fold_dir(Path("data/01_corpus_pkl_cv"), 5, 20260820, fold_index=0) / "train.pkl"
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


def ensure_kfold_corpus(
    source_dir: Path,
    output_base_dir: Path,
    n_splits: int,
    split_seed: int,
    val_size: float = 0.2,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Builds (or reuses, if already built and force=False) all n_splits
    folds for (n_splits, split_seed). Pools source_dir's train.pkl +
    development.pkl + test.pkl (data/01_corpus_pkl's schema: Id, Category,
    Topic, Source, Headline, Text, Link), runs a label-stratified
    StratifiedKFold to carve out each fold's test split (every row is used
    as test exactly once across the n_splits folds), then a stratified
    train_test_split on each fold's remaining rows for train/val.

    Returns the per-fold manifest (also cached at
    {fold_set_dir}/manifest.json): fold index, split sizes, label balance.
    """
    set_dir = fold_set_dir(output_base_dir, n_splits, split_seed)

    if _fold_set_is_complete(set_dir, n_splits) and not force:
        with open(set_dir / "manifest.json", "r", encoding="utf-8") as f:
            return json.load(f)["folds"]

    from sklearn.model_selection import StratifiedKFold, train_test_split

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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)

    folds_meta: List[Dict[str, Any]] = []
    for k, (rest_idx, test_idx) in enumerate(skf.split(pool, y)):
        rest = pool.iloc[rest_idx]
        test_df = pool.iloc[test_idx].reset_index(drop=True)

        train_idx, val_idx = train_test_split(
            rest.index,
            test_size=val_size,
            stratify=rest["Category"].astype(str),
            random_state=split_seed + k,
        )
        train_df = pool.loc[train_idx].reset_index(drop=True)
        val_df = pool.loc[val_idx].reset_index(drop=True)

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
            "label_balance_train": train_df["Category"].astype(str).value_counts().to_dict(),
            "label_balance_val": val_df["Category"].astype(str).value_counts().to_dict(),
            "label_balance_test": test_df["Category"].astype(str).value_counts().to_dict(),
        })

    manifest = {"n_splits": n_splits, "split_seed": split_seed, "val_size": val_size, "folds": folds_meta}
    with open(set_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return folds_meta
