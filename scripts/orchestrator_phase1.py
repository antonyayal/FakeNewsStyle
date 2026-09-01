# scripts/orchestrator_phase1.py
# -*- coding: utf-8 -*-
"""
Phase 1: latent dimension per branch, in isolation -- for each of the 4
modalities, only that one extractor is active (the other 3 are
--exclude_{branch}) while its latent dimension is swept over
experiment_config.PHASE1_DIM_CANDIDATES[branch], x the 5 fixed seeds from
experiment_config.SEEDS.

17 configs (5 semantic + 3 emotion + 4 style + 5 context) x 5 seeds = 85 runs.
Each (branch, dim) pair needs its own VAE, trained once via
experiment_runner.ensure_vae_latents and reused across its 5 seeds -- VAE
training has no seed and doesn't depend on any other branch being active.

Ranking: this doesn't compare branches against each other (a single-branch
model is expected to underperform any multi-branch combo -- that's Phase
2's job) -- it only ranks *dimensions within the same branch*, to pick each
branch's best latent size before Phase 2 tests extractor combinations.

Checkpointing: identical to the old orchestrators, via
experiment_config.PHASE1_RESULTS_JSONL and run_key.

Usage:
    python scripts/orchestrator_phase1.py --run                    # runs the 85 (or whichever are missing)
    python scripts/orchestrator_phase1.py --run --dry-run          # print the commands without executing
    python scripts/orchestrator_phase1.py --summary                # aggregate results/orchestrator_phase1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_results import aggregate_by_config, load_runs  # noqa: E402
from experiment_config import (  # noqa: E402
    ALL_MODALITIES,
    BASE_DIR,
    DEFAULT_LATENT_DIM,
    KAN_RUNS_DIR,
    PHASE1_DIM_CANDIDATES,
    PHASE1_RESULTS_JSONL,
    PHASE1_TOP_JSON,
    PHASE1_TOP_K,
    RANKING_METRIC,
    SEEDS,
)
from experiment_runner import (  # noqa: E402
    ensure_vae_latents,
    execute_and_log,
    load_ok_run_keys,
    python_executable,
)


def all_branch_dim_pairs() -> List[Dict[str, Any]]:
    pairs = []
    for branch in ALL_MODALITIES:
        for dim in PHASE1_DIM_CANDIDATES[branch]:
            pairs.append({"branch": branch, "dim": dim})
    return pairs


PAIRS = all_branch_dim_pairs()


def run_key_for(branch: str, dim: int, seed: int) -> str:
    return f"{branch}__dim{dim}__seed{seed}"


def build_kan_command(branch: str, dim: int, seed: int) -> List[str]:
    cmd = [python_executable(), "main.py", "--merge_vae_latents", "--train_kan"]

    for modality in ALL_MODALITIES:
        if modality != branch:
            cmd.append(f"--exclude_{modality}")
        cmd += [f"--{modality}_latent_dim", str(dim if modality == branch else DEFAULT_LATENT_DIM[modality])]

    output_dir = KAN_RUNS_DIR / "phase1" / branch / f"dim{dim}" / f"seed{seed}"
    cmd += [
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
    ]
    return cmd


def run_sweep(dry_run: bool) -> None:
    total = len(PAIRS) * len(SEEDS)
    print(f"Phase 1: {len(PAIRS)} (branch, dim) pairs x {len(SEEDS)} seeds = {total} runs")
    print(f"Results: {PHASE1_RESULTS_JSONL}")

    ok_keys = load_ok_run_keys(PHASE1_RESULTS_JSONL) if not dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = n_skip = n_failed = idx = 0

    for pair in PAIRS:
        branch, dim = pair["branch"], pair["dim"]
        ensure_vae_latents([branch], {branch: dim}, dry_run=dry_run)

        for seed in SEEDS:
            idx += 1
            key = run_key_for(branch, dim, seed)
            label = f"[{idx:03d}/{total}] {key}"

            if dry_run:
                cmd = build_kan_command(branch, dim, seed)
                print(f"{label}\n  $ {' '.join(cmd)}")
                continue

            if key in ok_keys:
                print(f"{label} SKIP (already completed)")
                n_skip += 1
                continue

            cmd = build_kan_command(branch, dim, seed)
            print(f"{label} RUN\n  $ {' '.join(cmd)}")

            record = execute_and_log(
                run_key=key,
                cmd=cmd,
                jsonl_path=PHASE1_RESULTS_JSONL,
                meta={
                    "phase": "phase1",
                    "branch": branch,
                    "dim": dim,
                    "active_extractors": [branch],
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

    print(f"\nPhase 1 complete (this invocation): {n_run} new runs, {n_skip} skipped, {n_failed} failed.")
    print(f"Total accumulated in {PHASE1_RESULTS_JSONL}: {len(load_ok_run_keys(PHASE1_RESULTS_JSONL))}/{total} ok.")


def summarize() -> None:
    df = load_runs(PHASE1_RESULTS_JSONL)
    if df.empty:
        print(f"No successful runs logged yet in {PHASE1_RESULTS_JSONL}.")
        return

    ranking = aggregate_by_config(df, group_by="branch_dim", metric=RANKING_METRIC)
    if ranking.empty:
        print("No successful runs -- nothing to rank.")
        return

    by_branch: Dict[str, List[Dict[str, Any]]] = {}
    for branch in ALL_MODALITIES:
        prefix = f"{branch}::dim"
        branch_ranking = ranking[ranking["config"].str.startswith(prefix)].head(PHASE1_TOP_K)

        print(f"\n=== {branch} -- ranking by {RANKING_METRIC} (top {PHASE1_TOP_K} kept) ===")
        entries = []
        for rank, (_, row) in enumerate(branch_ranking.iterrows(), start=1):
            dim = int(row["config"].split("::dim", 1)[1])
            print(f"  {rank}. dim={dim}  {RANKING_METRIC}_mean={row[f'{RANKING_METRIC}_mean']:.4f} "
                  f"+/- {row[f'{RANKING_METRIC}_std']:.4f} (n={int(row[f'{RANKING_METRIC}_count'])})")
            entries.append({
                "dim": dim,
                f"{RANKING_METRIC}_mean": float(row[f"{RANKING_METRIC}_mean"]),
                f"{RANKING_METRIC}_std": float(row[f"{RANKING_METRIC}_std"]),
                "n_runs": int(row[f"{RANKING_METRIC}_count"]),
            })
        by_branch[branch] = entries

    PHASE1_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metric": RANKING_METRIC, "seeds": SEEDS, "top_k": PHASE1_TOP_K, "by_branch": by_branch}
    with open(PHASE1_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {PHASE1_TOP_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: latent dimension per branch, in isolation")
    parser.add_argument("--run", action="store_true", help="Run the full sweep (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate results/orchestrator_phase1.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not any([args.run, args.summary]):
        parser.error("Pass at least one of --run, --summary")

    if args.run:
        run_sweep(dry_run=args.dry_run)
        if not args.dry_run:
            summarize()

    if args.summary and not args.run:
        summarize()


if __name__ == "__main__":
    main()
