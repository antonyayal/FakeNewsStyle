# scripts/orchestrator_phase1.py
# -*- coding: utf-8 -*-
"""
Phase 1 (Option A): context VAE latent dimension, one identity mode at a time.

Only the `context` branch is swept here. The other three branches are fixed
at main.py's default latent dims in every later phase -- an earlier full
per-branch sweep (old Phase 1) showed the latent size moves F1 by less than
seed noise within a branch, so re-sweeping all four is not worth the runs.

Two independent sub-sweeps, `context` alone (`--exclude_*` on the other 3):
  - identity ON  : Source Name/Source Link hash embeddings on (main.py
                   defaults), dims PHASE1_CONTEXT_ON_DIMS.
  - identity OFF : --context_source_name_dim 0 --context_source_link_dim 0,
                   dims PHASE1_CONTEXT_OFF_DIMS (capped at the 23-d
                   identity-free raw dimension).

(5 + 4) dims x SEEDS runs. Each (mode, dim) trains one VAE, reused across
its seeds. Ranking picks the best dim *within each mode* -- the two modes
are never compared here (that contrast is Phase 2/3's job).

Checkpoint/resume via results/orchestrator_phase1.jsonl + run_key.

Usage:
    python scripts/orchestrator_phase1.py --run
    python scripts/orchestrator_phase1.py --run --dry-run
    python scripts/orchestrator_phase1.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_results import aggregate_by_config, load_runs  # noqa: E402
from experiment_config import (  # noqa: E402
    BASE_DIR,
    KAN_RUNS_DIR,
    PHASE1_CONTEXT_OFF_DIMS,
    PHASE1_CONTEXT_ON_DIMS,
    PHASE1_RESULTS_JSONL,
    PHASE1_TOP_JSON,
    PHASE1_TOP_K,
    PHASE2_MERGED_DIR,
    RANKING_METRIC,
    SEEDS,
    VAE_LATENTS_DIR,
)
from experiment_plan import build_kan_cmd, ensure_idfree_context, merge_kan_inputs  # noqa: E402
from experiment_runner import (  # noqa: E402
    ensure_vae_latents,
    execute_and_log,
    load_ok_run_keys,
)

# hidden_dim doesn't matter for choosing the VAE dim -- fix it at one value.
PHASE1_HIDDEN_DIM = 64

MODES = [
    ("on", PHASE1_CONTEXT_ON_DIMS),
    ("off", PHASE1_CONTEXT_OFF_DIMS),
]
MERGED_ROOT = PHASE2_MERGED_DIR / "phase1_context_only"


def run_key_for(mode: str, dim: int, seed: int) -> str:
    return f"ctx{mode}__dim{dim}__seed{seed}"


def ensure_context_latents(mode: str, dim: int, dry_run: bool) -> Path:
    """Latent dir for `context` at (mode, dim), training the VAE if missing."""
    if mode == "on":
        ensure_vae_latents(["context"], {"context": dim}, dry_run=dry_run)
        return VAE_LATENTS_DIR / "context" / f"latent{dim}"
    return ensure_idfree_context(dim, fold_idx=None, dry_run=dry_run)


def run_sweep(dry_run: bool) -> None:
    pairs = [(mode, dim) for mode, dims in MODES for dim in dims]
    total = len(pairs) * len(SEEDS)
    print(f"Phase 1: {len(pairs)} (identity, dim) pairs x {len(SEEDS)} seeds = {total} runs")
    print(f"Results: {PHASE1_RESULTS_JSONL}")

    ok_keys = load_ok_run_keys(PHASE1_RESULTS_JSONL) if not dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = n_skip = n_failed = idx = 0

    for mode, dim in pairs:
        print(f"\n=== Phase 1 -- context identity {mode.upper()} @ dim={dim} ===")
        latent_dir = ensure_context_latents(mode, dim, dry_run)
        merged_dir = MERGED_ROOT / f"ctx{mode}" / f"dim{dim}"
        pkl_paths = merge_kan_inputs(["context"], {"context": latent_dir}, merged_dir, dry_run)

        for seed in SEEDS:
            idx += 1
            key = run_key_for(mode, dim, seed)
            label = f"[{idx:03d}/{total}] {key}"
            output_dir = KAN_RUNS_DIR / "phase1" / f"ctx{mode}" / f"dim{dim}" / f"seed{seed}"
            cmd = build_kan_cmd(
                combo=["context"], latent_dims={"context": dim}, pkl_paths=pkl_paths,
                hidden_dim=PHASE1_HIDDEN_DIM, seed=seed, output_dir=output_dir,
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
                run_key=key, cmd=cmd, jsonl_path=PHASE1_RESULTS_JSONL,
                meta={
                    "phase": "phase1",
                    "branch": f"context_{mode}",
                    "dim": dim,
                    "context_identity": mode,
                    "active_extractors": ["context"],
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

    print(f"\nPhase 1 complete (this invocation): {n_run} new, {n_skip} skipped, {n_failed} failed.")
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

    payload: Dict[str, Any] = {"metric": RANKING_METRIC, "seeds": SEEDS, "top_k": PHASE1_TOP_K}
    for mode in ("on", "off"):
        prefix = f"context_{mode}::dim"
        mode_ranking = ranking[ranking["config"].str.startswith(prefix)].head(PHASE1_TOP_K)
        print(f"\n=== context identity {mode.upper()} -- ranking by {RANKING_METRIC} (top {PHASE1_TOP_K}) ===")
        entries: List[Dict[str, Any]] = []
        for rank, (_, row) in enumerate(mode_ranking.iterrows(), start=1):
            dim = int(row["config"].split("::dim", 1)[1])
            test_col = f"test_{RANKING_METRIC}_mean"
            test_str = f"  (test {RANKING_METRIC}={row[test_col]:.4f})" if test_col in row and pd.notna(row[test_col]) else ""
            print(f"  {rank}. dim={dim}  val {RANKING_METRIC}_mean={row[f'{RANKING_METRIC}_mean']:.4f} "
                  f"+/- {row[f'{RANKING_METRIC}_std']:.4f} (n={int(row[f'{RANKING_METRIC}_count'])}){test_str}")
            entries.append({
                "dim": dim,
                f"{RANKING_METRIC}_mean": float(row[f"{RANKING_METRIC}_mean"]),
                f"{RANKING_METRIC}_std": float(row[f"{RANKING_METRIC}_std"]),
                "n_runs": int(row[f"{RANKING_METRIC}_count"]),
                **({f"test_{RANKING_METRIC}_mean": float(row[test_col])}
                   if test_col in row and pd.notna(row[test_col]) else {}),
            })
        if entries:
            payload[f"context_{mode}"] = entries[0]
            payload[f"context_{mode}_top"] = entries

    PHASE1_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE1_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {PHASE1_TOP_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 (Option A): context VAE latent dimension per identity mode")
    parser.add_argument("--run", action="store_true", help="Run the sweep (resumable)")
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
