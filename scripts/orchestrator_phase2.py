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

Phase 2 does NOT filter down to a top-K before handing off to Phase 3 --
all 15 combos (each fully resolved: extractors + Phase 1's winning dims)
are written to results/phase2_top.json and all 15 get hyperparameter-tuned
in Phase 3. Filtering on raw/default-hyperparameter performance before
hyperparameter tuning would let a combo that's mediocre under the default
KAN/VAE settings get discarded before it ever had a chance to shine under a
different setting (see e.g. 2026-09-03: with the leaky context branch, the
5-combo cutoff meant only context-containing combos ever reached Phase 3's
hyperparameter sweep). The only "top" selection left is Phase 1's dimension
choice per branch and, downstream, Phase 3's own top-5 after hyperparameters
have had a fair shot at every combo.

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


def summarize() -> None:
    phase2_df = load_runs(PHASE2_RESULTS_JSONL)

    if phase2_df.empty:
        print("No successful runs logged yet -- nothing to rank.")
        return

    dims = load_phase1_dims()
    ranking = aggregate_by_config(phase2_df, group_by="extractors", metric=RANKING_METRIC)

    print(f"\n=== Phase 2 -- all {len(ranking)} combos, ranked by val {RANKING_METRIC} "
          f"(all pass through to Phase 3, none filtered out here) ===")
    entries = []
    for i, row in ranking.iterrows():
        combo = [m for m in ALL_MODALITIES if m in row["config"].split("+")]
        combo_dims = {b: dims[b] for b in combo}

        test_col = f"test_{RANKING_METRIC}_mean"
        test_str = f"  (test {RANKING_METRIC}={row[test_col]:.4f})" if test_col in row and pd.notna(row[test_col]) else ""
        print(f"  {i + 1}. {row['config']}  dims={combo_dims}  val {RANKING_METRIC}_mean={row[f'{RANKING_METRIC}_mean']:.4f} "
              f"+/- {row[f'{RANKING_METRIC}_std']:.4f} (n={int(row[f'{RANKING_METRIC}_count'])}){test_str}")
        entries.append({
            "active_extractors": combo,
            "latent_dims": combo_dims,
            f"{RANKING_METRIC}_mean": float(row[f"{RANKING_METRIC}_mean"]),
            f"{RANKING_METRIC}_std": float(row[f"{RANKING_METRIC}_std"]),
            "n_runs": int(row[f"{RANKING_METRIC}_count"]),
            **({f"test_{RANKING_METRIC}_mean": float(row[test_col])} if test_col in row and pd.notna(row[test_col]) else {}),
        })

    PHASE2_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE2_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "top": entries}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {PHASE2_TOP_JSON} ({len(entries)} combos, all forwarded to Phase 3)")


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
