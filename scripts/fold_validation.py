# scripts/fold_validation.py
# -*- coding: utf-8 -*-
"""
Shared engine behind orchestrator_phase4.py (kfold, normal CV) and
orchestrator_phase5.py (source_disjoint CV) -- the two are ~90% identical
(same protocol: take all 15 configs from results/phase3_top.json -- one per
extractor combo, each already carrying its own best Phase 3 hyperparameters
-- and repeat each across N folds x the 5 fixed SEEDS, purely as a
generalization/robustness check on already-chosen models, exploring nothing
new), differing only in which of main.py's --corpus_mode partitions they
read.

Per fold, per config, two kinds of calls:
  1. once: --preprocess_text --extract_{active branches} --run_vaes
     --merge_vae_latents (builds that fold's corpus/features/VAE/merged
     latents for that config's combo+hyperparams; a marker file records
     which config was used, so a stale merge from a different config is
     rebuilt rather than silently reused)
  2. once per seed: --train_kan --kan_seed s (reads the same merged latents
     via the same --corpus_mode/index, no explicit --kan_train_pkl needed)

Since every extractor combo is validated across the same N folds, the
merged-latents directory is namespaced per config, not just per fold:
    {merged_cv_dir}/seed{S}_n{N}/fold{k}/{entry_label}/

Produces two outputs:
  - {phase}_per_fold.json: full per-config x per-fold stats (mean/std/min/max
    over the 5 seeds), for both the VAL split (selection metric, bare names)
    and TEST split (test_-prefixed, reported only, never used to pick a
    winner).
  - {phase}_top.json: ALL combos' results, ranked by mean VAL metric across
    all N folds x 5 seeds combined (not one result per fold) -- plus each
    combo's TEST metric alongside, for an honest read of its held-out
    performance. Deliberately not collapsed to a single global winner: Phase
    3 already gave every combo a fair shot at its own hyperparameters, so
    that diversity is preserved through fold validation instead of being
    reduced here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import ALL_MODALITIES, BASE_DIR, DEFAULT_LATENT_DIM, RANKING_METRIC, SEEDS  # noqa: E402
from experiment_runner import (  # noqa: E402
    execute_and_log,
    load_ok_run_keys,
    python_executable,
    run_main_command,
)


def config_label(combo: List[str]) -> str:
    return "_".join(m for m in ALL_MODALITIES if m in combo)


def entry_label(cfg: Dict[str, Any]) -> str:
    return f"{config_label(cfg['active_extractors'])}__{cfg['variant_label']}"


def _corpus_mode_flags(corpus_mode: str, n_folds: int, split_seed: int, fold_idx: int) -> List[str]:
    if corpus_mode == "kfold":
        return ["--corpus_mode", "kfold", "--kfold_n", str(n_folds),
                "--kfold_index", str(fold_idx), "--kfold_split_seed", str(split_seed)]
    elif corpus_mode == "source_disjoint":
        return ["--corpus_mode", "source_disjoint", "--source_split_n", str(n_folds),
                "--source_split_index", str(fold_idx), "--source_split_seed", str(split_seed)]
    else:
        raise ValueError(f"Unknown corpus_mode: {corpus_mode!r}")


def _merged_dir(merged_cv_dir: Path, split_seed: int, n_folds: int, fold_idx: int, label: str) -> Path:
    return merged_cv_dir / f"seed{split_seed}_n{n_folds}" / f"fold{fold_idx}" / label


def _marker_path(merged_cv_dir: Path, split_seed: int, n_folds: int, fold_idx: int, label: str) -> Path:
    return _merged_dir(merged_cv_dir, split_seed, n_folds, fold_idx, label) / "_marker.json"


def fold_is_ready(merged_cv_dir: Path, split_seed: int, n_folds: int, fold_idx: int, cfg: Dict[str, Any]) -> bool:
    label = entry_label(cfg)
    d = _merged_dir(merged_cv_dir, split_seed, n_folds, fold_idx, label)
    if not all((d / f"{s}.pkl").exists() for s in ["train", "val", "test"]):
        return False

    marker = _marker_path(merged_cv_dir, split_seed, n_folds, fold_idx, label)
    if not marker.exists():
        return False
    with open(marker, "r", encoding="utf-8") as f:
        saved = json.load(f)
    return saved == cfg


def prepare_fold(
    corpus_mode: str, n_folds: int, split_seed: int, fold_idx: int,
    merged_cv_dir: Path, cfg: Dict[str, Any], dry_run: bool,
) -> None:
    label = entry_label(cfg)
    if fold_is_ready(merged_cv_dir, split_seed, n_folds, fold_idx, cfg):
        print(f"  Fold {fold_idx} [{label}]: merged latents already exist for this config, skipping.")
        return

    combo, latent_dims = cfg["active_extractors"], cfg["latent_dims"]

    cmd = [python_executable(), "main.py",
           *_corpus_mode_flags(corpus_mode, n_folds, split_seed, fold_idx), "--preprocess_text"]
    for branch in ALL_MODALITIES:
        cmd.append(f"--extract_{branch}" if branch in combo else f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(latent_dims.get(branch, DEFAULT_LATENT_DIM[branch]))]
    cmd += [
        "--run_vaes",
        "--vae_beta", str(cfg["vae_beta"]),
        "--vae_dropout", str(cfg["vae_dropout"]),
        "--merge_vae_latents",
    ]

    print(f"  Fold {fold_idx} [{label}]: building corpus + features + VAE + merged latents")
    print(f"    $ {' '.join(cmd)}")
    if dry_run:
        return

    outcome = run_main_command(cmd, require_results_json=False)
    if outcome["error"] is not None:
        raise RuntimeError(f"Fold {fold_idx} [{label}]: preparation failed: {outcome['error']}")
    print(f"    OK in {outcome['elapsed_seconds']}s")

    marker = _marker_path(merged_cv_dir, split_seed, n_folds, fold_idx, label)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def build_kan_command(
    corpus_mode: str, n_folds: int, split_seed: int, fold_idx: int,
    cfg: Dict[str, Any], seed: int, output_dir: Path,
) -> List[str]:
    combo, latent_dims = cfg["active_extractors"], cfg["latent_dims"]
    cmd = [python_executable(), "main.py",
           *_corpus_mode_flags(corpus_mode, n_folds, split_seed, fold_idx), "--train_kan"]
    for branch in ALL_MODALITIES:
        if branch not in combo:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(latent_dims.get(branch, DEFAULT_LATENT_DIM[branch]))]
    cmd += [
        "--kan_num_basis", str(cfg["kan_num_basis"]),
        "--kan_hidden_dim", str(cfg["kan_hidden_dim"]),
        "--kan_dropout", str(cfg["kan_dropout"]),
        "--kan_epochs", str(cfg["kan_epochs"]),
        "--kan_batch_size", str(cfg["kan_batch_size"]),
        "--kan_lr", str(cfg["kan_lr"]),
        "--kan_weight_decay", str(cfg["kan_weight_decay"]),
        "--kan_patience", str(cfg["kan_patience"]),
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
        "--vae_beta", str(cfg["vae_beta"]),
        "--vae_dropout", str(cfg["vae_dropout"]),
    ]
    return cmd


def run_sweep(
    *, corpus_mode: str, n_folds: int, split_seed: int, configs: List[Dict[str, Any]],
    results_jsonl: Path, kan_runs_dir: Path, merged_cv_dir: Path, phase_name: str, dry_run: bool,
) -> None:
    total = len(configs) * n_folds * len(SEEDS)
    print(f"{phase_name}: {len(configs)} configs x {n_folds} folds x {len(SEEDS)} seeds = {total} runs")
    print(f"Results: {results_jsonl}")

    ok_keys = load_ok_run_keys(results_jsonl) if not dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = n_skip = n_failed = idx = 0

    for cfg in configs:
        label = entry_label(cfg)
        for fold_idx in range(n_folds):
            print(f"\n=== {phase_name} -- {label} -- fold {fold_idx}/{n_folds - 1} ===")
            prepare_fold(corpus_mode, n_folds, split_seed, fold_idx, merged_cv_dir, cfg, dry_run=dry_run)

            for seed in SEEDS:
                idx += 1
                key = f"{label}__fold{fold_idx}__seed{seed}"
                run_label = f"[{idx:04d}/{total}] {key}"
                output_dir = kan_runs_dir / label / f"fold{fold_idx}" / f"seed{seed}"
                cmd = build_kan_command(corpus_mode, n_folds, split_seed, fold_idx, cfg, seed, output_dir)

                if dry_run:
                    print(f"    [{key}]\n      $ {' '.join(cmd)}")
                    continue

                if key in ok_keys:
                    print(f"    [{key}] SKIP (already completed)")
                    n_skip += 1
                    continue

                print(f"    [{key}] RUN")
                record = execute_and_log(
                    run_key=key,
                    cmd=cmd,
                    jsonl_path=results_jsonl,
                    meta={
                        "phase": phase_name,
                        "entry_label": label,
                        "fold": fold_idx,
                        "seed": seed,
                        "active_extractors": cfg["active_extractors"],
                        "config": cfg,
                    },
                )
                if record["status"] == "ok":
                    n_run += 1
                    print(f"      OK in {record['elapsed_seconds']}s -- {record['results_json']}")
                else:
                    n_failed += 1
                    print(f"      FAILED -- {record['error']}")

    if dry_run:
        print(f"\ndry-run: {total} runs planned (not executed).")
        return

    print(f"\n{phase_name} complete (this invocation): {n_run} new runs, {n_skip} skipped, {n_failed} failed.")
    print(f"Total accumulated in {results_jsonl}: {len(load_ok_run_keys(results_jsonl))}/{total} ok.")


def summarize(
    *, results_jsonl: Path, per_fold_json: Path, top_json: Path, phase_name: str,
) -> None:
    if not results_jsonl.exists():
        print(f"No results yet: {results_jsonl}")
        return

    rows = []
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") != "ok" or not record.get("metrics"):
                continue
            rows.append({
                "entry_label": record["entry_label"],
                "fold": record["fold"],
                "seed": record["seed"],
                "active_extractors": record.get("active_extractors"),
                "config": record.get("config"),
                **record["metrics"],  # val split -- bare names, used for ranking/winner selection
                **{f"test_{k}": v for k, v in (record.get("test_metrics") or {}).items()},
            })

    if not rows:
        print("No successful runs logged yet.")
        return

    df = pd.DataFrame(rows)

    per_label_summary: Dict[str, Any] = {}
    global_rows = []
    test_col = f"test_{RANKING_METRIC}"
    has_test = test_col in df.columns

    for label, group in df.groupby("entry_label"):
        print(f"\n=== {label} -- per-fold val {RANKING_METRIC} (n={len(SEEDS)} seeds each) ===")
        per_fold = group.groupby("fold")[RANKING_METRIC].agg(["mean", "std", "min", "max", "count"])
        print(per_fold.to_string())

        global_stats = {
            f"{RANKING_METRIC}_mean": float(group[RANKING_METRIC].mean()),
            f"{RANKING_METRIC}_std": float(group[RANKING_METRIC].std()),
            f"{RANKING_METRIC}_min": float(group[RANKING_METRIC].min()),
            f"{RANKING_METRIC}_max": float(group[RANKING_METRIC].max()),
            "n_runs": int(len(group)),
            "n_folds": int(group["fold"].nunique()),
        }
        if has_test:
            global_stats[f"test_{RANKING_METRIC}_mean"] = float(group[test_col].mean())
            global_stats[f"test_{RANKING_METRIC}_std"] = float(group[test_col].std())
        test_str = f"  (test {RANKING_METRIC}={global_stats[f'test_{RANKING_METRIC}_mean']:.4f})" if has_test else ""
        print(f"  global: mean={global_stats[f'{RANKING_METRIC}_mean']:.4f} "
              f"std={global_stats[f'{RANKING_METRIC}_std']:.4f} (n={global_stats['n_runs']}){test_str}")

        per_label_summary[label] = {
            "active_extractors": group.iloc[0]["active_extractors"],
            "config": group.iloc[0]["config"],
            "per_fold": {
                str(fold): {stat: (float(v) if pd.notna(v) else None) for stat, v in row.items()}
                for fold, row in per_fold.iterrows()
            },
            "global": global_stats,
        }
        global_rows.append({"entry_label": label, "config": group.iloc[0]["config"], **global_stats})

    per_fold_json.parent.mkdir(parents=True, exist_ok=True)
    with open(per_fold_json, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "configs": per_label_summary}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {per_fold_json}")

    # One result PER combo, not a single collapsed global winner -- Phase 3
    # already gave every combo a fair shot at its own best hyperparameters,
    # so the extractor-combo diversity should survive all the way through
    # fold validation instead of being reduced to one "winner" here.
    ranked = sorted(global_rows, key=lambda r: r[f"{RANKING_METRIC}_mean"], reverse=True)
    print(f"\n=== {phase_name} -- all {len(ranked)} combos, ranked by val {RANKING_METRIC} ===")
    for i, r in enumerate(ranked, start=1):
        test_str = f"  (test {RANKING_METRIC}={r[f'test_{RANKING_METRIC}_mean']:.4f})" if f"test_{RANKING_METRIC}_mean" in r else ""
        print(f"  {i}. {r['entry_label']}  {RANKING_METRIC}_mean={r[f'{RANKING_METRIC}_mean']:.4f}{test_str}")

    top_json.parent.mkdir(parents=True, exist_ok=True)
    with open(top_json, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "results": ranked}, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {top_json}")
