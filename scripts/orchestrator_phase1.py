# scripts/orchestrator_phase1.py
# -*- coding: utf-8 -*-
"""
Phase 1: full sweep of expert combinations (the 15 non-empty subsets of
{semantic, emotion, style, context}) x the 10 fixed seeds from
experiment_config.SEEDS = 150 runs.

The VAE trains ONCE (at the default latent dimensions, for all 4 branches)
before the sweep -- VAE training is independent per branch in main.py
(doesn't depend on which other branches are active) and has no configurable
seed, so retraining it per run would only add uncontrolled noise to a
comparison that's meant to be paired by seed. Each of the 150 runs only
executes `--merge_vae_latents --train_kan`, reusing those cached latents.

Checkpointing: before launching a run, experiment_config.PHASE1_RESULTS_JSONL
is checked; if a line with the same run_key and status "ok" already exists,
it's skipped. Resuming after a crash is simply running this same script
again.

Usage:
    python scripts/orchestrator_phase1.py              # runs the 150 (or whichever are missing)
    python scripts/orchestrator_phase1.py --dry-run     # just prints the plan
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import (  # noqa: E402
    ALL_MODALITIES,
    BASE_DIR,
    DEFAULT_LATENT_DIM,
    KAN_RUNS_DIR,
    PHASE1_RESULTS_JSONL,
    SEEDS,
    VAE_LATENTS_DIR,
)
from experiment_runner import (  # noqa: E402
    execute_and_log,
    latent_cache_is_fresh,
    load_ok_run_keys,
    python_executable,
    run_main_command,
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


def default_vae_latents_missing() -> List[str]:
    missing = []
    for branch in ALL_MODALITIES:
        dim = DEFAULT_LATENT_DIM[branch]
        if not latent_cache_is_fresh(branch, dim, VAE_LATENTS_DIR):
            missing.append(f"{branch} (latent{dim}, missing or stale vs. current corpus)")
    return missing


def build_vae_prep_command() -> List[str]:
    cmd = [python_executable(), "main.py", "--run_vaes"]
    for branch in ALL_MODALITIES:
        cmd += [f"--{branch}_latent_dim", str(DEFAULT_LATENT_DIM[branch])]
    return cmd


def ensure_default_vae_latents(dry_run: bool) -> None:
    missing = default_vae_latents_missing()
    if not missing:
        print("Default VAE latents already exist for all 4 branches -- reusing them.")
        return

    print("Default VAE latents missing, will be trained once:")
    for m in missing:
        print(f"  - {m}")

    cmd = build_vae_prep_command()
    print(f"  $ {' '.join(cmd)}")

    if dry_run:
        print("  (dry-run: not executing)")
        return

    outcome = run_main_command(cmd, require_results_json=False)
    if outcome["error"] is not None:
        raise RuntimeError(f"Failed training default VAE: {outcome['error']}")
    print(f"  OK in {outcome['elapsed_seconds']}s")


def build_kan_command(combo: List[str], seed: int) -> List[str]:
    cmd = [python_executable(), "main.py", "--merge_vae_latents", "--train_kan"]

    for modality in ALL_MODALITIES:
        if modality not in combo:
            cmd.append(f"--exclude_{modality}")

    for branch in ALL_MODALITIES:
        cmd += [f"--{branch}_latent_dim", str(DEFAULT_LATENT_DIM[branch])]

    output_dir = KAN_RUNS_DIR / "phase1" / combo_label(combo) / f"seed{seed}"
    cmd += [
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Phase 1: sweep of expert combos x seeds")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running anything")
    args = parser.parse_args()

    total = len(COMBOS) * len(SEEDS)
    print(f"Phase 1: {len(COMBOS)} combos x {len(SEEDS)} seeds = {total} runs")
    print(f"Results: {PHASE1_RESULTS_JSONL}")

    ensure_default_vae_latents(dry_run=args.dry_run)

    ok_keys = load_ok_run_keys(PHASE1_RESULTS_JSONL) if not args.dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = 0
    n_skip = 0
    n_failed = 0
    idx = 0

    for combo in COMBOS:
        for seed in SEEDS:
            idx += 1
            key = run_key_for(combo, seed)
            label = f"[{idx:03d}/{total}] {key}"

            if args.dry_run:
                cmd = build_kan_command(combo, seed)
                print(f"{label}\n  $ {' '.join(cmd)}")
                continue

            if key in ok_keys:
                print(f"{label} SKIP (already completed)")
                n_skip += 1
                continue

            cmd = build_kan_command(combo, seed)
            print(f"{label} RUN\n  $ {' '.join(cmd)}")

            record = execute_and_log(
                run_key=key,
                cmd=cmd,
                jsonl_path=PHASE1_RESULTS_JSONL,
                meta={
                    "phase": "phase1",
                    "group": None,
                    "candidate_label": None,
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

    if args.dry_run:
        print(f"\ndry-run: {total} runs planned (not executed).")
        return

    print(f"\nPhase 1 complete (this invocation): {n_run} new runs, {n_skip} skipped, {n_failed} failed.")
    print(f"Total accumulated in {PHASE1_RESULTS_JSONL}: {len(load_ok_run_keys(PHASE1_RESULTS_JSONL))}/{total} ok.")


if __name__ == "__main__":
    main()
