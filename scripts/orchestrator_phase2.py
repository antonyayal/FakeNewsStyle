# scripts/orchestrator_phase2.py
# -*- coding: utf-8 -*-
"""
Phase 2 (Option A): standard-split evaluation -- the in-distribution result.

Runs every extractor combination, unfiltered:
  - the 15 non-empty subsets of {semantic, emotion, style, context};
  - each combo that includes `context` is run twice, with the Source/Domain
    hash embeddings ON and OFF (identity-free);
  - each of those 23 combo-variants at both kan_hidden_dim values;
  - x SEEDS.
= 23 x 2 x 3 = 138 KAN runs on the fixed train/val/test split.

VAE + KAN hyperparameters are fixed (experiment_config.FINAL_HPARAMS); only
kan_hidden_dim varies. Non-context branches and identity-ON context read the
shared VAE cache (data/05_vae_latents/); identity-OFF context is trained into
its own isolated cache (data/05_vae_latents_idfree/). Per prep-unit the
branch latents are concatenated once into KAN-ready PKLs, then the
hidden_dim x seed grid is pure KAN training on top.

Context dims come from results/phase1_top.json (fallbacks in
experiment_config if Phase 1 hasn't run). Checkpoint/resume via
results/orchestrator_phase2.jsonl + run_key.

Usage:
    python scripts/orchestrator_phase2.py --run
    python scripts/orchestrator_phase2.py --run --dry-run
    python scripts/orchestrator_phase2.py --summary
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

from experiment_config import (  # noqa: E402
    BASE_DIR,
    HIDDEN_DIM_GRID,
    KAN_RUNS_DIR,
    PHASE2_MERGED_DIR,
    PHASE2_RESULTS_JSONL,
    PHASE2_TOP_JSON,
    RANKING_METRIC,
    SEEDS,
    VAE_LATENTS_DIR,
)
from experiment_plan import (  # noqa: E402
    build_kan_cmd,
    build_prep_units,
    entry_label,
    ensure_idfree_context,
    iter_kan_entries,
    load_context_dims,
    merge_kan_inputs,
    total_kan_runs,
)
from experiment_runner import ensure_vae_latents, execute_and_log, load_ok_run_keys  # noqa: E402


def prepare_standard(unit: Dict[str, Any], dry_run: bool) -> Dict[str, Path]:
    """Branch latents -> KAN-ready {split}.pkl for one prep unit."""
    combo = unit["active_extractors"]
    latent_dims = unit["latent_dims"]
    mode = unit["context_identity"]

    latent_dirs: Dict[str, Path] = {}
    non_context = [b for b in combo if b != "context"]
    shared = non_context + (["context"] if mode == "on" else [])
    if shared:
        ensure_vae_latents(shared, latent_dims, dry_run=dry_run)
        for b in shared:
            latent_dirs[b] = VAE_LATENTS_DIR / b / f"latent{latent_dims[b]}"
    if mode == "off":
        latent_dirs["context"] = ensure_idfree_context(latent_dims["context"], fold_idx=None, dry_run=dry_run)

    merged_dir = PHASE2_MERGED_DIR / unit["prep_label"]
    return merge_kan_inputs(combo, latent_dirs, merged_dir, dry_run)


def run_sweep(dry_run: bool) -> None:
    on_dim, off_dim = load_context_dims()
    prep_units = build_prep_units(on_dim, off_dim)
    total = total_kan_runs(prep_units)
    print(f"Phase 2: {len(prep_units)} combo-variants x {len(HIDDEN_DIM_GRID)} "
          f"hidden_dim x {len(SEEDS)} seeds = {total} runs")
    print(f"context dims: identity ON = {on_dim}, identity OFF = {off_dim}")
    print(f"Results: {PHASE2_RESULTS_JSONL}")

    ok_keys = load_ok_run_keys(PHASE2_RESULTS_JSONL) if not dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = n_skip = n_failed = idx = 0
    prepared: Dict[str, Dict[str, Path]] = {}

    for unit, hidden_dim in iter_kan_entries(prep_units):
        combo = unit["active_extractors"]
        latent_dims = unit["latent_dims"]
        mode = unit["context_identity"]
        plabel = unit["prep_label"]
        elabel = entry_label(combo, mode, hidden_dim)

        if plabel not in prepared:
            print(f"\n=== Phase 2 -- prep {plabel} ===")
            prepared[plabel] = prepare_standard(unit, dry_run)
        pkl_paths = prepared[plabel]

        for seed in SEEDS:
            idx += 1
            key = f"{elabel}__seed{seed}"
            run_label = f"[{idx:03d}/{total}] {key}"
            output_dir = KAN_RUNS_DIR / "phase2" / elabel / f"seed{seed}"
            cmd = build_kan_cmd(
                combo=combo, latent_dims=latent_dims, pkl_paths=pkl_paths,
                hidden_dim=hidden_dim, seed=seed, output_dir=output_dir,
            )

            if dry_run:
                print(f"{run_label}\n  $ {' '.join(cmd)}")
                continue
            if key in ok_keys:
                print(f"{run_label} SKIP (already completed)")
                n_skip += 1
                continue

            print(f"{run_label} RUN")
            record = execute_and_log(
                run_key=key, cmd=cmd, jsonl_path=PHASE2_RESULTS_JSONL,
                meta={
                    "phase": "phase2",
                    "entry_label": elabel,
                    "prep_label": plabel,
                    "active_extractors": combo,
                    "context_identity": mode,
                    "hidden_dim": hidden_dim,
                    "latent_dims": latent_dims,
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

    print(f"\nPhase 2 complete (this invocation): {n_run} new, {n_skip} skipped, {n_failed} failed.")
    print(f"Total accumulated in {PHASE2_RESULTS_JSONL}: {len(load_ok_run_keys(PHASE2_RESULTS_JSONL))}/{total} ok.")


def summarize() -> None:
    if not PHASE2_RESULTS_JSONL.exists():
        print(f"No results yet: {PHASE2_RESULTS_JSONL}")
        return

    rows: List[Dict[str, Any]] = []
    with open(PHASE2_RESULTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") != "ok" or not record.get("metrics"):
                continue
            rows.append({
                "entry_label": record["entry_label"],
                "active_extractors": record.get("active_extractors"),
                "context_identity": record.get("context_identity"),
                "hidden_dim": record.get("hidden_dim"),
                "seed": record.get("seed"),
                RANKING_METRIC: record["metrics"].get(RANKING_METRIC),
                f"test_{RANKING_METRIC}": (record.get("test_metrics") or {}).get(RANKING_METRIC),
                "test_accuracy": (record.get("test_metrics") or {}).get("accuracy"),
            })
    if not rows:
        print("No successful runs logged yet.")
        return

    df = pd.DataFrame(rows)
    ranked: List[Dict[str, Any]] = []
    for label, group in df.groupby("entry_label"):
        ranked.append({
            "entry_label": label,
            "active_extractors": group.iloc[0]["active_extractors"],
            "context_identity": group.iloc[0]["context_identity"],
            "hidden_dim": int(group.iloc[0]["hidden_dim"]),
            "n_runs": int(len(group)),
            f"{RANKING_METRIC}_mean": float(group[RANKING_METRIC].mean()),
            f"{RANKING_METRIC}_std": float(group[RANKING_METRIC].std()),
            f"test_{RANKING_METRIC}_mean": float(group[f"test_{RANKING_METRIC}"].mean()),
            f"test_{RANKING_METRIC}_std": float(group[f"test_{RANKING_METRIC}"].std()),
            "test_accuracy_mean": float(group["test_accuracy"].mean()),
        })
    ranked.sort(key=lambda r: r[f"{RANKING_METRIC}_mean"], reverse=True)

    print(f"\n=== Phase 2 -- all {len(ranked)} combo-variants, ranked by val {RANKING_METRIC} ===")
    for i, r in enumerate(ranked, start=1):
        print(f"  {i:2d}. {r['entry_label']:<42} "
              f"val {RANKING_METRIC}={r[f'{RANKING_METRIC}_mean']:.4f}  "
              f"test {RANKING_METRIC}={r[f'test_{RANKING_METRIC}_mean']:.4f}")

    PHASE2_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE2_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "seeds": SEEDS, "results": ranked}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {PHASE2_TOP_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 (Option A): standard-split evaluation of all combo-variants")
    parser.add_argument("--run", action="store_true", help="Run the sweep (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate results/orchestrator_phase2.jsonl")
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
