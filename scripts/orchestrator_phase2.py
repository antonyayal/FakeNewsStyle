# scripts/orchestrator_phase2.py
# -*- coding: utf-8 -*-
"""
Phase 2: extractor combinations -- the 15 non-empty subsets of {semantic,
emotion, style, context} x the 5 fixed seeds from experiment_config.SEEDS =
75 runs. Each active branch uses its rank-1 latent dimension from Phase 1
(results/phase1_top.json) -- Phase 1's top-2 per branch are NOT crossed
here, only the single best dimension per branch is used for every combo.

The VAE for each active branch (at its Phase-1-winning dim) trains ONCE via
experiment_runner.ensure_vae_latents and is reused across the 15 combos that
include it and their 5 seeds each.

Ranking pool for the final top 5: the 15 combo results here, PLUS the 8
single-branch results already run in Phase 1 (top-2 x 4 branches) -- those
aren't re-run, their raw per-seed rows are pulled straight from
PHASE1_RESULTS_JSONL and pooled into the same ranking.

Usage:
    python scripts/orchestrator_phase2.py --run
    python scripts/orchestrator_phase2.py --run --dry-run
    python scripts/orchestrator_phase2.py --summary
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_results import aggregate_by_config, load_runs  # noqa: E402
from experiment_config import (  # noqa: E402
    ALL_MODALITIES,
    BASE_DIR,
    KAN_RUNS_DIR,
    PHASE1_RESULTS_JSONL,
    PHASE1_TOP_JSON,
    PHASE2_RESULTS_JSONL,
    PHASE2_TOP_JSON,
    RANKING_METRIC,
    SEEDS,
)
from experiment_runner import (  # noqa: E402
    ensure_vae_latents,
    execute_and_log,
    load_ok_run_keys,
    python_executable,
)


def all_nonempty_combos(modalities: List[str]) -> List[List[str]]:
    combos = []
    for r in range(1, len(modalities) + 1):
        combos.extend(list(c) for c in itertools.combinations(modalities, r))
    return combos


COMBOS = all_nonempty_combos(ALL_MODALITIES)


def combo_label(combo: List[str]) -> str:
    return "_".join(combo)


def run_key_for(combo: List[str], seed: int) -> str:
    return f"{combo_label(combo)}__seed{seed}"


def load_phase1_dims() -> Dict[str, int]:
    if not PHASE1_TOP_JSON.exists():
        raise FileNotFoundError(
            f"{PHASE1_TOP_JSON} not found. Run scripts/orchestrator_phase1.py --run first."
        )
    with open(PHASE1_TOP_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {branch: entries[0]["dim"] for branch, entries in data["by_branch"].items()}


def build_kan_command(combo: List[str], dims: Dict[str, int], seed: int) -> List[str]:
    cmd = [python_executable(), "main.py", "--merge_vae_latents", "--train_kan"]

    for branch in ALL_MODALITIES:
        if branch not in combo:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(dims[branch])]

    output_dir = KAN_RUNS_DIR / "phase2" / combo_label(combo) / f"seed{seed}"
    cmd += [
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
    ]
    return cmd


def run_sweep(dims: Dict[str, int], dry_run: bool) -> None:
    total = len(COMBOS) * len(SEEDS)
    print(f"Phase 2: {len(COMBOS)} combos x {len(SEEDS)} seeds = {total} runs")
    print(f"Using Phase 1 dims: {dims}")
    print(f"Results: {PHASE2_RESULTS_JSONL}")

    ok_keys = load_ok_run_keys(PHASE2_RESULTS_JSONL) if not dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = n_skip = n_failed = idx = 0

    for combo in COMBOS:
        ensure_vae_latents(combo, dims, dry_run=dry_run)

        for seed in SEEDS:
            idx += 1
            key = run_key_for(combo, seed)
            label = f"[{idx:03d}/{total}] {key}"

            if dry_run:
                cmd = build_kan_command(combo, dims, seed)
                print(f"{label}\n  $ {' '.join(cmd)}")
                continue

            if key in ok_keys:
                print(f"{label} SKIP (already completed)")
                n_skip += 1
                continue

            cmd = build_kan_command(combo, dims, seed)
            print(f"{label} RUN\n  $ {' '.join(cmd)}")

            record = execute_and_log(
                run_key=key,
                cmd=cmd,
                jsonl_path=PHASE2_RESULTS_JSONL,
                meta={
                    "phase": "phase2",
                    "active_extractors": combo,
                    "seed": seed,
                    "overrides": {},
                    "kan_output_dir": str(cmd[cmd.index("--kan_output_dir") + 1]),
                },
            )

            if record["status"] == "ok":
                n_run += 1
                print(f"  OK in {record['elapsed_seconds']}s -- {record['results_json']}")
            else:
                n_failed += 1
                print(f"  FAILED -- {record['error']}")

    if dry_run:
        print(f"\ndry-run: {total} runs planned (not executed).")
        return

    print(f"\nPhase 2 complete (this invocation): {n_run} new runs, {n_skip} skipped, {n_failed} failed.")
    print(f"Total accumulated in {PHASE2_RESULTS_JSONL}: {len(load_ok_run_keys(PHASE2_RESULTS_JSONL))}/{total} ok.")


def _phase1_pool_rows() -> pd.DataFrame:
    """Raw per-seed rows for Phase 1's top-2-per-branch (8 (branch, dim)
    pairs) -- pulled from PHASE1_RESULTS_JSONL, not re-run, so they can be
    pooled into the same ranking as Phase 2's own combo runs."""
    if not PHASE1_TOP_JSON.exists():
        return pd.DataFrame()
    with open(PHASE1_TOP_JSON, "r", encoding="utf-8") as f:
        top = json.load(f)

    winning_pairs = {
        (branch, entry["dim"]) for branch, entries in top["by_branch"].items() for entry in entries
    }

    df = load_runs(PHASE1_RESULTS_JSONL)
    if df.empty:
        return df
    mask = df.apply(lambda row: (row.get("branch"), row.get("dim")) in winning_pairs, axis=1)
    return df[mask]


def summarize() -> None:
    phase2_df = load_runs(PHASE2_RESULTS_JSONL)
    phase1_pool = _phase1_pool_rows()
    combined = pd.concat([phase2_df, phase1_pool], ignore_index=True) if not phase1_pool.empty else phase2_df

    if combined.empty:
        print("No successful runs logged yet -- nothing to rank.")
        return

    ranking = aggregate_by_config(combined, group_by="extractors", metric=RANKING_METRIC)
    top5 = ranking.head(5)

    print(f"\n=== Phase 2 top 5 (15 combos + 8 Phase 1 solo winners, by {RANKING_METRIC}) ===")
    entries = []
    for i, row in top5.iterrows():
        combo = [m for m in ALL_MODALITIES if m in row["config"].split("+")]
        subset = combined[combined["active_extractors"].apply(lambda c: "+".join(c)) == row["config"]]
        if len(combo) == 1 and "dim" in subset.columns and subset["dim"].notna().any():
            dims = {combo[0]: int(subset["dim"].dropna().iloc[0])}
        else:
            dims = load_phase1_dims()
            dims = {b: dims[b] for b in combo}

        test_col = f"test_{RANKING_METRIC}_mean"
        test_str = f"  (test {RANKING_METRIC}={row[test_col]:.4f})" if test_col in row and pd.notna(row[test_col]) else ""
        print(f"  {i + 1}. {row['config']}  dims={dims}  val {RANKING_METRIC}_mean={row[f'{RANKING_METRIC}_mean']:.4f} "
              f"+/- {row[f'{RANKING_METRIC}_std']:.4f} (n={int(row[f'{RANKING_METRIC}_count'])}){test_str}")
        entries.append({
            "active_extractors": combo,
            "latent_dims": dims,
            f"{RANKING_METRIC}_mean": float(row[f"{RANKING_METRIC}_mean"]),
            f"{RANKING_METRIC}_std": float(row[f"{RANKING_METRIC}_std"]),
            "n_runs": int(row[f"{RANKING_METRIC}_count"]),
            **({f"test_{RANKING_METRIC}_mean": float(row[test_col])} if test_col in row and pd.notna(row[test_col]) else {}),
        })

    PHASE2_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE2_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "top": entries}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {PHASE2_TOP_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: extractor combinations using Phase 1's winning dims")
    parser.add_argument("--run", action="store_true", help="Run the full sweep (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate results/orchestrator_phase2.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not any([args.run, args.summary]):
        parser.error("Pass at least one of --run, --summary")

    if args.run:
        dims = load_phase1_dims()
        run_sweep(dims, dry_run=args.dry_run)
        if not args.dry_run:
            summarize()

    if args.summary and not args.run:
        summarize()


if __name__ == "__main__":
    main()
