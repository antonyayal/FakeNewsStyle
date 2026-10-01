# scripts/phase1b_branch_dim_sweep.py
# -*- coding: utf-8 -*-
"""
Phase 1b (addendum to Phase 1): does main.py's default latent dimension for
the `semantic`, `style`, and `emotion` branches actually beat the
alternative, under the disciplined 3-seed protocol -- closing a gap between
the paper's stated goal ("best hyperparameters per branch", Table 5.2 /
single-branch performance) and what the current 3-phase plan
(scripts/orchestrator_phase{1,2,3}.py) actually tests: only `context`'s
latent dimension is swept there; `semantic`/`emotion`/`style` are frozen at
main.py's defaults (128/16/16) on the strength of an older, less disciplined
sweep (results_old_2/results_old_3) that is re-examined here, not just
cited.

That older sweep is suspicious on inspection:
  - semantic: dim=64 (mean test F1 ~0.743-0.745) clearly beats the current
    default dim=128 (~0.674-0.720), and dim>=256 collapses to F1=0.0 across
    every seed in that sweep -- an instability, not "no effect".
  - style: dim=32 (~0.723-0.733) edges out the current default dim=16
    (~0.716-0.723) by ~0.01-0.03 F1.
  - emotion: dims 8/16/23 land within ~0.01 F1 of each other in that same
    sweep -- the "barely moves F1" claim holds there, but is checked here
    too (8 vs.\ 16) so all three branches go through the same protocol
    instead of two being re-verified and one only cited from the old sweep.

This mirrors orchestrator_phase1.py's design (one branch alone, --exclude_*
on the other three, FINAL_HPARAMS, fixed kan_hidden_dim, the SAME 3 fixed
SEEDS from experiment_config.py, standard split -- matching what Table 5.2
reports) but sweeps BRANCH_DIMS below instead of context, and reuses the
SHARED default VAE-latent cache (ensure_vae_latents) since these are
ordinary latent-dim values, not isolated non-default hyperparameters.

  3 branches x 2 dims x 3 seeds = 18 runs (semantic's dim=128, style's
  dim=16, and emotion's dim=16 are the current defaults and are likely
  already cached from other phases, so this typically trains only 3 new
  VAEs: semantic@64, style@32, emotion@8).

Checkpoint/resume via results/orchestrator_phase1b.jsonl + run_key.

Usage:
    python scripts/phase1b_branch_dim_sweep.py --run
    python scripts/phase1b_branch_dim_sweep.py --run --dry-run
    python scripts/phase1b_branch_dim_sweep.py --summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_results import aggregate_by_config, load_runs, pairwise_wilcoxon  # noqa: E402
from experiment_config import (  # noqa: E402
    BASE_DIR,
    KAN_RUNS_DIR,
    PHASE2_MERGED_DIR,
    RANKING_METRIC,
    SEEDS,
    VAE_LATENTS_DIR,
)
from experiment_plan import build_kan_cmd, merge_kan_inputs  # noqa: E402
from experiment_runner import (  # noqa: E402
    ensure_vae_latents,
    execute_and_log,
    load_ok_run_keys,
)

# hidden_dim doesn't matter for choosing the VAE dim -- fix it, same value
# Phase 1 used for context.
PHASE1B_HIDDEN_DIM = 64

# Current main.py defaults included explicitly, so the sweep directly
# confirms or refutes them rather than only testing alternatives.
BRANCH_DIMS: Dict[str, List[int]] = {
    "semantic": [64, 128],
    "style": [16, 32],
    "emotion": [8, 16],
}

RESULTS_JSONL = BASE_DIR / "results" / "orchestrator_phase1b.jsonl"
TOP_JSON = BASE_DIR / "results" / "phase1b_top.json"
MERGED_ROOT = PHASE2_MERGED_DIR / "phase1b_branch_only"


def run_key_for(branch: str, dim: int, seed: int) -> str:
    return f"{branch}__dim{dim}__seed{seed}"


def run_sweep(dry_run: bool) -> None:
    pairs = [(branch, dim) for branch, dims in BRANCH_DIMS.items() for dim in dims]
    total = len(pairs) * len(SEEDS)
    print(f"Phase 1b: {len(pairs)} (branch, dim) pairs x {len(SEEDS)} seeds = {total} runs")
    print(f"Results: {RESULTS_JSONL}")

    ok_keys = load_ok_run_keys(RESULTS_JSONL) if not dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = n_skip = n_failed = idx = 0

    for branch, dim in pairs:
        print(f"\n=== Phase 1b -- {branch} @ dim={dim} ===")
        ensure_vae_latents([branch], {branch: dim}, dry_run=dry_run)
        latent_dir = VAE_LATENTS_DIR / branch / f"latent{dim}"
        merged_dir = MERGED_ROOT / branch / f"dim{dim}"
        pkl_paths = merge_kan_inputs([branch], {branch: latent_dir}, merged_dir, dry_run)

        for seed in SEEDS:
            idx += 1
            key = run_key_for(branch, dim, seed)
            label = f"[{idx:02d}/{total}] {key}"
            output_dir = KAN_RUNS_DIR / "phase1b" / branch / f"dim{dim}" / f"seed{seed}"
            cmd = build_kan_cmd(
                combo=[branch], latent_dims={branch: dim}, pkl_paths=pkl_paths,
                hidden_dim=PHASE1B_HIDDEN_DIM, seed=seed, output_dir=output_dir,
            )

            if dry_run:
                print(f"{label}\n  $ {' '.join(cmd)}")
                continue
            if key in ok_keys:
                print(f"{label} SKIP (already completed)")
                n_skip += 1
                continue

            print(f"{label} RUN")
            record = execute_and_log(
                run_key=key, cmd=cmd, jsonl_path=RESULTS_JSONL,
                meta={
                    "phase": "phase1b",
                    "branch": branch,
                    "dim": dim,
                    "active_extractors": [branch],
                    "seed": seed,
                    "kan_output_dir": str(output_dir.relative_to(BASE_DIR)),
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

    print(f"\nPhase 1b complete (this invocation): {n_run} new, {n_skip} skipped, {n_failed} failed.")
    print(f"Total accumulated in {RESULTS_JSONL}: {len(load_ok_run_keys(RESULTS_JSONL))}/{total} ok.")


def summarize() -> None:
    df = load_runs(RESULTS_JSONL)
    if df.empty:
        print(f"No successful runs logged yet in {RESULTS_JSONL}.")
        return

    ranking = aggregate_by_config(df, group_by="branch_dim", metric=RANKING_METRIC)
    if ranking.empty:
        print("No successful runs -- nothing to rank.")
        return

    payload: Dict[str, Any] = {"metric": RANKING_METRIC, "seeds": SEEDS}
    for branch in BRANCH_DIMS:
        prefix = f"{branch}::dim"
        branch_ranking = ranking[ranking["config"].str.startswith(prefix)]
        print(f"\n=== {branch} -- ranking by {RANKING_METRIC} ===")
        entries: List[Dict[str, Any]] = []
        for rank, (_, row) in enumerate(branch_ranking.iterrows(), start=1):
            dim = int(row["config"].split("::dim", 1)[1])
            test_col = f"test_{RANKING_METRIC}_mean"
            test_str = f"  (test {RANKING_METRIC}={row[test_col]:.4f})" if test_col in row and row.notna()[test_col] else ""
            print(f"  {rank}. dim={dim}  val {RANKING_METRIC}_mean={row[f'{RANKING_METRIC}_mean']:.4f} "
                  f"+/- {row[f'{RANKING_METRIC}_std']:.4f} (n={int(row[f'{RANKING_METRIC}_count'])}){test_str}")
            entries.append({
                "dim": dim,
                f"{RANKING_METRIC}_mean": float(row[f"{RANKING_METRIC}_mean"]),
                f"{RANKING_METRIC}_std": float(row[f"{RANKING_METRIC}_std"]),
                "n_runs": int(row[f"{RANKING_METRIC}_count"]),
                **({f"test_{RANKING_METRIC}_mean": float(row[test_col])}
                   if test_col in row and row.notna()[test_col] else {}),
            })
        payload[branch] = entries

        wilcox = pairwise_wilcoxon(df, group_by="branch_dim", ranking=branch_ranking, top_n=len(entries), metric=RANKING_METRIC)
        if not wilcox.empty:
            print(f"  Wilcoxon (paired by seed):")
            for _, row in wilcox.iterrows():
                p_str = f"{row['p_value']:.4f}" if row["p_value"] is not None else f"n/a ({row['note']})"
                print(f"    {row['config_a']} vs {row['config_b']}: p={p_str}")
            payload[f"{branch}_wilcoxon"] = wilcox.to_dict(orient="records")

    TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(TOP_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {TOP_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1b: re-check semantic/style latent-dim defaults under the 3-seed protocol")
    parser.add_argument("--run", action="store_true", help="Run the sweep (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate results/orchestrator_phase1b.jsonl")
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
