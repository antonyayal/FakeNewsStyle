# scripts/aggregate_results.py
# -*- coding: utf-8 -*-
"""
Aggregates the results of an orchestration JSON-lines file (Phase 1 or
Phase 2): groups by configuration, computes mean +/- standard deviation of
each metric across runs (one per seed), sorts by the mean of the ranking
metric (F1 by default), and applies a Wilcoxon signed-rank test paired by
seed between the configurations closest to the top to check statistical
significance.

Reusable as a module (orchestrator_phase2.py imports it to decide the
winner of each hyperparameter group) and as a CLI:

    python scripts/aggregate_results.py --input results/orchestrator_phase1.jsonl --group-by extractors
    python scripts/aggregate_results.py --input results/orchestrator_phase2.jsonl --group-by candidate
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import PHASE1_WINNERS_JSON, RANKING_METRIC, SEEDS  # noqa: E402

METRIC_COLUMNS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
    "roc_auc", "pr_auc", "mcc", "log_loss", "brier_score", "ece",
    "entropy_mean", "entropy_std", "n_params", "train_time_sec",
]


def load_runs(jsonl_path: Path) -> pd.DataFrame:
    """Reads the JSONL, keeps only status=='ok' rows, flattens metrics.* to
    columns. Returns an empty DataFrame (not an error) if the file is
    missing or has no successful runs -- callers decide how to react."""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return pd.DataFrame()

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") != "ok" or not record.get("metrics"):
                continue

            row = {k: v for k, v in record.items() if k != "metrics"}
            row.update(record["metrics"])
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _group_key_column(df: pd.DataFrame, group_by: str) -> pd.Series:
    if group_by == "extractors":
        return df["active_extractors"].apply(lambda combo: "+".join(combo))
    elif group_by == "candidate":
        return df["group"].astype(str) + "::" + df["candidate_label"].astype(str)
    else:
        raise ValueError(f"Unknown group_by: {group_by!r} (use 'extractors' or 'candidate')")


def aggregate_by_config(df: pd.DataFrame, group_by: str, metric: str = RANKING_METRIC) -> pd.DataFrame:
    """One row per configuration: n_runs, mean/std of every metric in
    METRIC_COLUMNS, sorted descending by {metric}_mean. Warns (does not
    raise) if a config has fewer than len(SEEDS) successful runs."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["_config_key"] = _group_key_column(work, group_by)

    available_cols = [c for c in METRIC_COLUMNS if c in work.columns]

    agg = work.groupby("_config_key")[available_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index().rename(columns={"_config_key": "config"})

    n_expected = len(SEEDS)
    count_col = f"{metric}_count"
    if count_col in agg.columns:
        for _, row in agg.iterrows():
            if row[count_col] < n_expected:
                warnings.warn(
                    f"Config '{row['config']}' has {int(row[count_col])}/{n_expected} "
                    f"successful runs -- mean/std computed over fewer seeds."
                )

    sort_col = f"{metric}_mean"
    if sort_col in agg.columns:
        agg = agg.sort_values(sort_col, ascending=False).reset_index(drop=True)

    return agg


def pairwise_wilcoxon(
    df: pd.DataFrame,
    group_by: str,
    ranking: pd.DataFrame,
    top_n: int = 4,
    metric: str = RANKING_METRIC,
) -> pd.DataFrame:
    """Wilcoxon signed-rank test, paired by seed, between consecutive pairs
    in the top_n ranking. Skips (with a warning) any pair that doesn't share
    the exact same set of seeds with a successful run on both sides."""
    if ranking.empty:
        return pd.DataFrame()

    work = df.copy()
    work["_config_key"] = _group_key_column(work, group_by)

    top_configs = ranking["config"].head(top_n).tolist()
    rows = []

    for a, b in zip(top_configs[:-1], top_configs[1:]):
        series_a = work[work["_config_key"] == a].set_index("seed")[metric]
        series_b = work[work["_config_key"] == b].set_index("seed")[metric]

        common_seeds = sorted(set(series_a.index) & set(series_b.index))
        if len(common_seeds) < 2:
            warnings.warn(f"'{a}' vs '{b}': fewer than 2 common seeds, skipping Wilcoxon.")
            continue

        if set(series_a.index) != set(series_b.index):
            warnings.warn(
                f"'{a}' vs '{b}': different seed sets, using the intersection "
                f"({len(common_seeds)} seeds)."
            )

        x = series_a.loc[common_seeds].values
        y = series_b.loc[common_seeds].values

        if np.allclose(x, y):
            rows.append({"config_a": a, "config_b": b, "n_seeds": len(common_seeds),
                         "statistic": 0.0, "p_value": 1.0, "note": "identical values"})
            continue

        try:
            stat, p_value = wilcoxon(x, y)
            rows.append({"config_a": a, "config_b": b, "n_seeds": len(common_seeds),
                         "statistic": float(stat), "p_value": float(p_value), "note": None})
        except ValueError as exc:
            rows.append({"config_a": a, "config_b": b, "n_seeds": len(common_seeds),
                         "statistic": None, "p_value": None, "note": str(exc)})

    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame) -> None:
    try:
        from tabulate import tabulate  # optional, nicer output if installed
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    except ImportError:
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(df.to_string(index=False))


def report(ranking: pd.DataFrame, wilcoxon_df: pd.DataFrame, metric: str = RANKING_METRIC, top_k: int = 3) -> None:
    if ranking.empty:
        print("No successful runs -- nothing to report.")
        return

    display_cols = ["config"] + [
        c for c in ranking.columns
        if c.startswith((f"{metric}_", "accuracy_", "roc_auc_", "mcc_")) and c != "config"
    ]
    print("\n=== Ranking by configuration (sorted by mean %s) ===" % metric)
    _print_table(ranking[display_cols] if display_cols else ranking)

    print(f"\n=== Top {top_k} ===")
    for i, row in ranking.head(top_k).iterrows():
        print(f"  {i + 1}. {row['config']}  ({metric}_mean={row[f'{metric}_mean']:.4f} "
              f"+/- {row[f'{metric}_std']:.4f}, n={int(row[f'{metric}_count'])})")

    if not wilcoxon_df.empty:
        print("\n=== Wilcoxon signed-rank (paired by seed), consecutive pairs from the top ===")
        _print_table(wilcoxon_df)
        for _, row in wilcoxon_df.iterrows():
            if row["p_value"] is not None:
                sig = "significant (p<0.05)" if row["p_value"] < 0.05 else "not significant"
                print(f"  {row['config_a']} vs {row['config_b']}: p={row['p_value']:.4f} ({sig})")


def _extract_active_extractors(ranking: pd.DataFrame, df: pd.DataFrame, top_k: int) -> List[List[str]]:
    winners = []
    for config in ranking["config"].head(top_k):
        match = df[df["active_extractors"].apply(lambda c: "+".join(c)) == config]
        winners.append(match.iloc[0]["active_extractors"])
    return winners


def main():
    parser = argparse.ArgumentParser(description="Aggregates and ranks results from an orchestration JSON-lines file")
    parser.add_argument("--input", required=True, help="Path to the JSON-lines (Phase 1 or Phase 2)")
    parser.add_argument("--group-by", choices=["extractors", "candidate"], required=True)
    parser.add_argument("--metric", default=RANKING_METRIC, help=f"Ranking metric (default: {RANKING_METRIC})")
    parser.add_argument("--top-n-wilcoxon", type=int, default=4, help="How many top configs to compare pairwise")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-winners", nargs="?", const=str(PHASE1_WINNERS_JSON), default=None,
                         help="If passed (with --group-by extractors), writes the top-k combos to this JSON "
                              f"(with no value, uses the default: {PHASE1_WINNERS_JSON}; format consumed by "
                              "orchestrator_phase2.py --winners)")
    args = parser.parse_args()

    df = load_runs(Path(args.input))
    if df.empty:
        print(f"No successful runs in {args.input}.")
        return

    ranking = aggregate_by_config(df, args.group_by, metric=args.metric)
    wilcoxon_df = pairwise_wilcoxon(df, args.group_by, ranking, top_n=args.top_n_wilcoxon, metric=args.metric)
    report(ranking, wilcoxon_df, metric=args.metric, top_k=args.top_k)

    if args.output_winners:
        if args.group_by != "extractors":
            parser.error("--output-winners only makes sense with --group-by extractors")
        winners = _extract_active_extractors(ranking, df, args.top_k)
        out_path = Path(args.output_winners)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"winners": [list(w) for w in winners], "metric": args.metric}, f, indent=2, ensure_ascii=False)
        print(f"\nTop {args.top_k} combos saved to: {out_path}")


if __name__ == "__main__":
    main()
