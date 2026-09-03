# scripts/orchestrator_phase3.py
# -*- coding: utf-8 -*-
"""
Phase 3: VAE regularization + KAN hyperparameters, fused into a sweep of 18
variants: a shared baseline -- main.py's own defaults -- plus one candidate
per non-default value of vae_beta, vae_dropout, kan_num_basis,
kan_hidden_dim (covering the full {8, 16, 32, 64, 128} range, 64 being the
baseline), kan_weight_decay (one-knob-at-a-time), plus one combined variant
that overrides several of those knobs together; see
experiment_config.PHASE3_CANDIDATES. kan_lr / kan_batch_size are not swept
-- a prior sweep already found the default wins for both.

Applied to all 15 configs in results/phase2_top.json (Phase 2 forwards every
extractor combo, not a pre-filtered subset -- see orchestrator_phase2.py):
15 configs x 18 variants x 5 seeds = 1350 runs. Ranking pool = all 15 x 18 =
270 (config, variant) combinations -> the BEST variant PER combo is kept
(never a flat top-N across combos, which would let one dominant combo
occupy every slot) -> all 15 combos, each fully resolved (extractors +
latent dims + its own best hyperparameters), advance to Phase 4/5.

Variants matching the baseline (vae_beta=1.0, vae_dropout=0.1) reuse the
shared default VAE cache via --merge_vae_latents (experiment_runner.
ensure_vae_latents trains only what's missing). Variants that change
vae_beta/vae_dropout train an isolated VAE and merge it manually
(experiment_runner.resolve_kan_input / merge_latents_manual), since
main.py's --merge_vae_latents always reads from the default cache path.

Usage:
    python scripts/orchestrator_phase3.py --run
    python scripts/orchestrator_phase3.py --run --dry-run
    python scripts/orchestrator_phase3.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_results import aggregate_by_config, load_runs  # noqa: E402
from experiment_config import (  # noqa: E402
    ALL_MODALITIES,
    BASE_DIR,
    DEFAULT_LATENT_DIM,
    FIXED_KAN_BASELINE,
    KAN_RUNS_DIR,
    PHASE2_TOP_JSON,
    PHASE3_BASELINE,
    PHASE3_CANDIDATES,
    PHASE3_RESULTS_JSONL,
    PHASE3_TOP_JSON,
    PHASE3_VAE_DATA_DIR,
    PHASE3_VAE_MERGED_DIR,
    PHASE3_VAE_MODEL_DIR,
    RANKING_METRIC,
    SEEDS,
)
from experiment_runner import (  # noqa: E402
    execute_and_log,
    load_ok_run_keys,
    python_executable,
    resolve_kan_input,
)


def load_phase2_top() -> List[Dict[str, Any]]:
    if not PHASE2_TOP_JSON.exists():
        raise FileNotFoundError(f"{PHASE2_TOP_JSON} not found. Run scripts/orchestrator_phase2.py --run first.")
    with open(PHASE2_TOP_JSON, "r", encoding="utf-8") as f:
        return json.load(f)["top"]


def config_label(combo: List[str]) -> str:
    return "_".join(m for m in ALL_MODALITIES if m in combo)


def build_kan_command(
    combo: List[str], latent_dims: Dict[str, int], effective: Dict[str, Any], seed: int,
    output_dir: Path, kan_pkl_paths: Optional[Dict[str, Path]] = None,
) -> List[str]:
    cmd = [python_executable(), "main.py"]
    if kan_pkl_paths is None:
        cmd.append("--merge_vae_latents")
    cmd.append("--train_kan")

    for branch in ALL_MODALITIES:
        if branch not in combo:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(latent_dims.get(branch, DEFAULT_LATENT_DIM[branch]))]

    if kan_pkl_paths is not None:
        cmd += [
            "--kan_train_pkl", str(kan_pkl_paths["train"]),
            "--kan_val_pkl", str(kan_pkl_paths["val"]),
            "--kan_test_pkl", str(kan_pkl_paths["test"]),
        ]

    cmd += [
        "--kan_num_basis", str(effective["kan_num_basis"]),
        "--kan_hidden_dim", str(effective["kan_hidden_dim"]),
        "--kan_dropout", str(FIXED_KAN_BASELINE["kan_dropout"]),
        "--kan_epochs", str(FIXED_KAN_BASELINE["kan_epochs"]),
        "--kan_patience", str(FIXED_KAN_BASELINE["kan_patience"]),
        "--kan_batch_size", str(FIXED_KAN_BASELINE["kan_batch_size"]),
        "--kan_lr", str(FIXED_KAN_BASELINE["kan_lr"]),
        "--kan_weight_decay", str(effective["kan_weight_decay"]),
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
        # Repeated here even though this subprocess doesn't always retrain the
        # VAE: main.py's Step 10 logs vae_hyperparams from its own args
        # regardless of what actually trained it.
        "--vae_beta", str(effective["vae_beta"]),
        "--vae_dropout", str(effective["vae_dropout"]),
    ]
    return cmd


def run_sweep(configs: List[Dict[str, Any]], dry_run: bool) -> None:
    total = len(configs) * len(PHASE3_CANDIDATES) * len(SEEDS)
    print(f"Phase 3: {len(configs)} configs x {len(PHASE3_CANDIDATES)} variants x {len(SEEDS)} seeds = {total} runs")
    print(f"Results: {PHASE3_RESULTS_JSONL}")

    ok_keys = load_ok_run_keys(PHASE3_RESULTS_JSONL) if not dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = n_skip = n_failed = idx = 0

    for cfg in configs:
        combo = cfg["active_extractors"]
        latent_dims = cfg["latent_dims"]
        label = config_label(combo)

        for variant_label, override in PHASE3_CANDIDATES.items():
            effective = {**PHASE3_BASELINE, **override}
            kan_pkl_paths = resolve_kan_input(
                combo, label, variant_label,
                {**effective, "latent": latent_dims},
                PHASE3_VAE_DATA_DIR, PHASE3_VAE_MODEL_DIR, PHASE3_VAE_MERGED_DIR,
                dry_run=dry_run,
            )

            for seed in SEEDS:
                idx += 1
                key = f"{label}__{variant_label}__seed{seed}"
                run_label = f"[{idx:04d}/{total}] {key}"
                output_dir = KAN_RUNS_DIR / "phase3" / label / variant_label / f"seed{seed}"
                cmd = build_kan_command(combo, latent_dims, effective, seed, output_dir, kan_pkl_paths=kan_pkl_paths)

                if dry_run:
                    print(f"{run_label}\n  $ {' '.join(cmd)}")
                    continue

                if key in ok_keys:
                    print(f"{run_label} SKIP (already completed)")
                    n_skip += 1
                    continue

                print(f"{run_label} RUN\n  $ {' '.join(cmd)}")
                record = execute_and_log(
                    run_key=key,
                    cmd=cmd,
                    jsonl_path=PHASE3_RESULTS_JSONL,
                    meta={
                        "phase": "phase3",
                        "config_label": label,
                        "group": label,
                        "candidate_label": variant_label,
                        "active_extractors": combo,
                        "seed": seed,
                        "overrides": effective,
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

    print(f"\nPhase 3 complete (this invocation): {n_run} new runs, {n_skip} skipped, {n_failed} failed.")
    print(f"Total accumulated in {PHASE3_RESULTS_JSONL}: {len(load_ok_run_keys(PHASE3_RESULTS_JSONL))}/{total} ok.")


def summarize(configs: List[Dict[str, Any]]) -> None:
    df = load_runs(PHASE3_RESULTS_JSONL)
    if df.empty:
        print("No successful runs logged yet -- nothing to rank.")
        return

    ranking = aggregate_by_config(df, group_by="candidate", metric=RANKING_METRIC)
    # aggregate_by_config already sorts descending by {RANKING_METRIC}_mean, so
    # keeping the first row per extractor-combo label is that combo's best
    # hyperparameter variant. One winner PER combo -- never a flat top-5 across
    # all (combo, variant) pairs, which could (and did) let a single dominant
    # combo occupy every slot and starve every other combo of ever finishing
    # hyperparameter tuning.
    ranking["_label"] = ranking["config"].str.split("::", n=1).str[0]
    best_per_combo = ranking.drop_duplicates(subset="_label", keep="first")

    latent_by_label = {config_label(cfg["active_extractors"]): cfg["latent_dims"] for cfg in configs}

    print(f"\n=== Phase 3 -- best hyperparameter variant per combo ({len(configs)} configs x "
          f"{len(PHASE3_CANDIDATES)} variants = {len(configs) * len(PHASE3_CANDIDATES)} combos tried, "
          f"{len(best_per_combo)} combos advancing, by {RANKING_METRIC}) ===")
    entries = []
    for i, (_, row) in enumerate(best_per_combo.iterrows()):
        label, variant_label = row["config"].split("::", 1)
        combo = [m for m in ALL_MODALITIES if m in label.split("_")]
        effective = {**PHASE3_BASELINE, **PHASE3_CANDIDATES[variant_label]}

        test_col = f"test_{RANKING_METRIC}_mean"
        test_str = f"  (test {RANKING_METRIC}={row[test_col]:.4f})" if test_col in row and pd.notna(row[test_col]) else ""
        print(f"  {i + 1}. {label} / {variant_label}  val {RANKING_METRIC}_mean={row[f'{RANKING_METRIC}_mean']:.4f} "
              f"+/- {row[f'{RANKING_METRIC}_std']:.4f} (n={int(row[f'{RANKING_METRIC}_count'])}){test_str}")
        entries.append({
            "active_extractors": combo,
            "latent_dims": latent_by_label[label],
            "variant_label": variant_label,
            **effective,
            **FIXED_KAN_BASELINE,
            f"{RANKING_METRIC}_mean": float(row[f"{RANKING_METRIC}_mean"]),
            f"{RANKING_METRIC}_std": float(row[f"{RANKING_METRIC}_std"]),
            "n_runs": int(row[f"{RANKING_METRIC}_count"]),
            **({f"test_{RANKING_METRIC}_mean": float(row[test_col])} if test_col in row and pd.notna(row[test_col]) else {}),
        })

    PHASE3_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE3_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "top": entries}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {PHASE3_TOP_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: fused VAE-reg + KAN hyperparameter sweep on Phase 2's top 5")
    parser.add_argument("--run", action="store_true", help="Run the full sweep (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate results/orchestrator_phase3.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not any([args.run, args.summary]):
        parser.error("Pass at least one of --run, --summary")

    configs = load_phase2_top()

    if args.run:
        run_sweep(configs, dry_run=args.dry_run)
        if not args.dry_run:
            summarize(configs)

    if args.summary and not args.run:
        summarize(configs)


if __name__ == "__main__":
    main()
