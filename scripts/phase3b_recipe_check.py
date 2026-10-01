# scripts/phase3b_recipe_check.py
# -*- coding: utf-8 -*-
"""
Phase 3b (addendum to the 3-phase Option A plan): does the old Phase-3-sweep
"winning" hyperparameter recipe (results_old, commit 31471893, test F1 up to
~0.86 on the standard split) generalize better than the current frozen
FINAL_HPARAMS -- once Source Name/Source Link identity is off and evaluation is done on
genuinely held-out sources?

Context: results_old's best all-4-branch run used
    semantic=64, emotion=8, style=8, context=32 (identity ON), vae_dropout=0.3,
    kan_num_basis=8, kan_hidden_dim=16, one lucky seed among hundreds tried.
scripts/experiment_config.py's FINAL_HPARAMS concluded (from that same old
sweep) that nothing beat main.py's defaults by more than seed noise -- but
that conclusion was drawn in a regime dominated by context/identity leakage
variance (dataset_source_label_leakage memory), so it was never actually
tested in the leakage-free (identity-off) regime this script targets.

This is NOT a re-run of the old ~1350-run grid. It isolates exactly one
question -- does the old recipe's smaller capacity + heavier regularization
help once context can no longer memorize outlet identity -- on a small,
pre-registered set of combo x recipe pairs, using the SAME 3 fixed seeds as
every other phase (SEEDS in experiment_config.py; this is what the doctoral
committee asked for, not a wider seed search) and Phase 3's source-disjoint
folds (the only honest evaluation surface).

Scope (deliberately small -- 3 combos x 2 recipes x 3 seeds x 5 folds = 90
KAN runs):
    combos  -- semantic+emotion+style, semantic+style,
               semantic+emotion+style+context (context identity OFF)
    recipes -- "current" (FINAL_HPARAMS: sem128/emo16/sty16, dropout=0.1,
               num_basis=16) vs. "old_resultsold" (sem64/emo8/sty8,
               dropout=0.3, num_basis=8). kan_hidden_dim is held fixed at 16
               (the old recipe's value) in BOTH arms, so hidden_dim -- already
               swept elsewhere -- doesn't confound this comparison.
    context (when present) -- identity OFF, latent dim fixed at Phase 1's
               off-winner (results/phase1_top.json) in BOTH recipes, since
               Phase 1 already optimized that dimension independently; only
               the non-context branches' capacity/regularization varies.

VAE latents for the swept branches are cached in an ISOLATED tree
(data/05_vae_latents_recipecmp/{recipe}/...) so this never touches or
invalidates the shared FINAL_HPARAMS cache Phase 2/3 rely on. Raw features
are read from Phase 3's existing shared per-fold cache (SOURCE_CV_RAW_DIR) --
recipe-independent, so nothing is re-extracted.

Usage:
    python scripts/phase3b_recipe_check.py --run
    python scripts/phase3b_recipe_check.py --run --dry-run
    python scripts/phase3b_recipe_check.py --summary
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
    KAN_RUNS_DIR,
    SEEDS,
    SOURCE_CV_RAW_DIR,
    SOURCE_SPLIT_N_FOLDS,
    SOURCE_SPLIT_SEED,
    source_cv_fold_dir,
)
from experiment_plan import ensure_idfree_context, load_context_dims  # noqa: E402
from experiment_runner import (  # noqa: E402
    execute_and_log,
    latent_cache_is_fresh,
    load_ok_run_keys,
    merge_latents_manual,
    python_executable,
    run_main_command,
)

PHASE_NAME = "phase3b_recipe_check"
RESULTS_JSONL = BASE_DIR / "results" / "phase3b_recipe_check.jsonl"
TOP_JSON = BASE_DIR / "results" / "phase3b_recipe_check_top.json"

RECIPE_VAE_DATA_DIR = BASE_DIR / "data" / "05_vae_latents_recipecmp"
RECIPE_VAE_MODEL_DIR = BASE_DIR / "models" / "vae_recipecmp"
RECIPE_MERGED_DIR = BASE_DIR / "data" / "06_vae_latents_merged_recipecmp"

RANKING_METRIC = "f1"

# ---- the two recipes under comparison -------------------------------------
# kan_hidden_dim fixed at 16 in both arms (the old recipe's value) so the
# already-swept hidden_dim axis can't confound this comparison.
RECIPES: Dict[str, Dict[str, Any]] = {
    "current": {
        "latent_dims": {"semantic": 128, "emotion": 16, "style": 16},
        "vae_dropout": 0.1,
        "kan_num_basis": 16,
        "kan_hidden_dim": 16,
    },
    "old_resultsold": {
        "latent_dims": {"semantic": 64, "emotion": 8, "style": 8},
        "vae_dropout": 0.3,
        "kan_num_basis": 8,
        "kan_hidden_dim": 16,
    },
}
# Unchanged between recipes -- same as FINAL_HPARAMS.
COMMON_KAN = {
    "kan_dropout": 0.2,
    "kan_weight_decay": 1e-4,
    "kan_epochs": 100,
    "kan_patience": 15,
    "kan_batch_size": 32,
    "kan_lr": 1e-3,
}
VAE_BETA = 1.0  # same in results_old's winning run and in FINAL_HPARAMS

COMBOS: List[List[str]] = [
    ["semantic", "emotion", "style"],
    ["semantic", "style"],
    ["semantic", "emotion", "style", "context"],
]


def combo_label(combo: List[str]) -> str:
    return "_".join(combo)


def entry_label(combo: List[str], recipe_name: str) -> str:
    base = combo_label(combo)
    if "context" in combo:
        base += "__ctxoff"
    return f"{base}__{recipe_name}"


def _source_disjoint_flags(fold_idx: int) -> List[str]:
    return ["--corpus_mode", "source_disjoint",
            "--source_split_n", str(SOURCE_SPLIT_N_FOLDS),
            "--source_split_index", str(fold_idx),
            "--source_split_seed", str(SOURCE_SPLIT_SEED)]


def prepare_fold(
    combo: List[str], recipe_name: str, recipe_cfg: Dict[str, Any],
    context_off_dim: int, fold_idx: int, dry_run: bool,
) -> Dict[str, Path]:
    """Build fold `fold_idx`'s KAN-ready PKLs for (combo, recipe): isolated
    VAE latents for the swept branches (semantic/emotion/style), reusing
    Phase 3's already-cached identity-off context latents unchanged."""
    non_context = [b for b in combo if b != "context"]
    latent_dims = {b: recipe_cfg["latent_dims"][b] for b in non_context}
    if "context" in combo:
        latent_dims["context"] = context_off_dim

    plabel = entry_label(combo, recipe_name)
    merged_dir = source_cv_fold_dir(RECIPE_MERGED_DIR, fold_idx) / plabel

    marker = merged_dir / "_marker.json"
    if not dry_run and all((merged_dir / f"{s}.pkl").exists() for s in ("train", "val", "test")) and marker.exists():
        with open(marker, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("latent_dims") == latent_dims and saved.get("vae_dropout") == recipe_cfg["vae_dropout"]:
            print(f"  fold {fold_idx} [{plabel}]: KAN inputs already prepared, skip")
            return {s: merged_dir / f"{s}.pkl" for s in ("train", "val", "test")}

    vae_data_dir = source_cv_fold_dir(RECIPE_VAE_DATA_DIR / recipe_name, fold_idx)
    vae_model_dir = source_cv_fold_dir(RECIPE_VAE_MODEL_DIR / recipe_name, fold_idx)
    fold_raw = source_cv_fold_dir(SOURCE_CV_RAW_DIR, fold_idx)

    latent_dirs: Dict[str, Path] = {}

    stale = [
        b for b in non_context
        if dry_run or not latent_cache_is_fresh(b, latent_dims[b], vae_data_dir, raw_dir_override=fold_raw / b)
    ]
    if stale:
        cmd = [python_executable(), "main.py", *_source_disjoint_flags(fold_idx), "--run_vaes"]
        for b in ("semantic", "emotion", "style"):
            if b not in non_context:
                cmd.append(f"--exclude_{b}")
            cmd += [f"--{b}_latent_dim", str(latent_dims.get(b, recipe_cfg["latent_dims"].get(b, 16)))]
        cmd.append("--exclude_context")
        cmd += [
            "--vae_beta", str(VAE_BETA),
            "--vae_dropout", str(recipe_cfg["vae_dropout"]),
            "--vae_data_output_dir", str(vae_data_dir.relative_to(BASE_DIR)),
            "--vae_model_output_dir", str(vae_model_dir.relative_to(BASE_DIR)),
        ]
        print(f"  fold {fold_idx} [{plabel}]: training isolated VAE for {stale} "
              f"(dropout={recipe_cfg['vae_dropout']}, dims={latent_dims})")
        print(f"    $ {' '.join(cmd)}")
        if not dry_run:
            outcome = run_main_command(cmd, require_results_json=False)
            if outcome["error"] is not None:
                raise RuntimeError(f"fold {fold_idx} [{plabel}]: VAE prep failed: {outcome['error']}")
            print(f"      OK in {outcome['elapsed_seconds']}s")

    for b in non_context:
        latent_dirs[b] = vae_data_dir / b / f"latent{latent_dims[b]}"

    if "context" in combo:
        latent_dirs["context"] = ensure_idfree_context(context_off_dim, fold_idx=fold_idx, dry_run=dry_run)

    if dry_run:
        pkl_paths = {s: merged_dir / f"{s}.pkl" for s in ("train", "val", "test")}
    else:
        pkl_paths = merge_latents_manual(combo, latent_dirs, merged_dir)
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({"latent_dims": latent_dims, "vae_dropout": recipe_cfg["vae_dropout"]}, f, indent=2)
    return pkl_paths


def build_kan_cmd(
    *, combo: List[str], pkl_paths: Dict[str, Path], recipe_cfg: Dict[str, Any],
    seed: int, output_dir: Path, fold_idx: int,
) -> List[str]:
    cmd = [python_executable(), "main.py", *_source_disjoint_flags(fold_idx), "--train_kan"]
    for b in ("semantic", "emotion", "style", "context"):
        if b not in combo:
            cmd.append(f"--exclude_{b}")
    cmd += [
        "--kan_train_pkl", str(pkl_paths["train"]),
        "--kan_val_pkl", str(pkl_paths["val"]),
        "--kan_test_pkl", str(pkl_paths["test"]),
        "--kan_hidden_dim", str(recipe_cfg["kan_hidden_dim"]),
        "--kan_num_basis", str(recipe_cfg["kan_num_basis"]),
        "--kan_dropout", str(COMMON_KAN["kan_dropout"]),
        "--kan_weight_decay", str(COMMON_KAN["kan_weight_decay"]),
        "--kan_epochs", str(COMMON_KAN["kan_epochs"]),
        "--kan_patience", str(COMMON_KAN["kan_patience"]),
        "--kan_batch_size", str(COMMON_KAN["kan_batch_size"]),
        "--kan_lr", str(COMMON_KAN["kan_lr"]),
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
        "--vae_beta", str(VAE_BETA),
        "--vae_dropout", str(recipe_cfg["vae_dropout"]),
    ]
    return cmd


def run_sweep(dry_run: bool) -> None:
    on_dim, off_dim = load_context_dims()
    n_folds = SOURCE_SPLIT_N_FOLDS
    total = len(COMBOS) * len(RECIPES) * len(SEEDS) * n_folds
    print(f"Phase 3b: {len(COMBOS)} combos x {len(RECIPES)} recipes x {len(SEEDS)} seeds x {n_folds} folds = {total} runs")
    print(f"context identity-off dim (from phase1_top.json): {off_dim}")
    print(f"Results: {RESULTS_JSONL}")

    ok_keys = load_ok_run_keys(RESULTS_JSONL) if not dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = n_skip = n_failed = idx = 0

    for fold_idx in range(n_folds):
        prepared: Dict[str, Dict[str, Path]] = {}
        for combo in COMBOS:
            for recipe_name, recipe_cfg in RECIPES.items():
                plabel = entry_label(combo, recipe_name)
                if plabel not in prepared:
                    print(f"\n=== Phase 3b -- fold {fold_idx}/{n_folds - 1} -- prep {plabel} ===")
                    prepared[plabel] = prepare_fold(combo, recipe_name, recipe_cfg, off_dim, fold_idx, dry_run)
                pkl_paths = prepared[plabel]

                for seed in SEEDS:
                    idx += 1
                    key = f"{plabel}__fold{fold_idx}__seed{seed}"
                    run_label = f"[{idx:04d}/{total}] {key}"
                    output_dir = KAN_RUNS_DIR / "phase3b_recipecmp" / plabel / f"fold{fold_idx}" / f"seed{seed}"
                    cmd = build_kan_cmd(
                        combo=combo, pkl_paths=pkl_paths, recipe_cfg=recipe_cfg,
                        seed=seed, output_dir=output_dir, fold_idx=fold_idx,
                    )

                    if dry_run:
                        print(f"{run_label}\n  $ {' '.join(cmd)}")
                        continue
                    if key in ok_keys:
                        n_skip += 1
                        continue

                    print(f"{run_label} RUN")
                    record = execute_and_log(
                        run_key=key, cmd=cmd, jsonl_path=RESULTS_JSONL,
                        meta={
                            "phase": PHASE_NAME,
                            "entry_label": plabel,
                            "active_extractors": combo,
                            "recipe": recipe_name,
                            "recipe_cfg": recipe_cfg,
                            "context_identity": "off" if "context" in combo else None,
                            "fold": fold_idx,
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

    print(f"\nPhase 3b complete (this invocation): {n_run} new, {n_skip} skipped, {n_failed} failed.")
    print(f"Total accumulated in {RESULTS_JSONL}: {len(load_ok_run_keys(RESULTS_JSONL))}/{total} ok.")


def summarize() -> None:
    if not RESULTS_JSONL.exists():
        print(f"No results yet: {RESULTS_JSONL}")
        return

    rows: List[Dict[str, Any]] = []
    with open(RESULTS_JSONL, "r", encoding="utf-8") as f:
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
                "recipe": record.get("recipe"),
                "fold": record.get("fold"),
                "seed": record.get("seed"),
                RANKING_METRIC: record["metrics"].get(RANKING_METRIC),
                f"test_{RANKING_METRIC}": (record.get("test_metrics") or {}).get(RANKING_METRIC),
            })
    if not rows:
        print("No successful runs logged yet.")
        return

    df = pd.DataFrame(rows)
    ranked: List[Dict[str, Any]] = []
    for label, group in df.groupby("entry_label"):
        entry = {
            "entry_label": label,
            "active_extractors": group.iloc[0]["active_extractors"],
            "recipe": group.iloc[0]["recipe"],
            "n_runs": int(len(group)),
            "n_folds": int(group["fold"].nunique()),
            f"{RANKING_METRIC}_mean": float(group[RANKING_METRIC].mean()),
            f"{RANKING_METRIC}_std": float(group[RANKING_METRIC].std()),
            f"test_{RANKING_METRIC}_mean": float(group[f"test_{RANKING_METRIC}"].mean()),
            f"test_{RANKING_METRIC}_std": float(group[f"test_{RANKING_METRIC}"].std()),
        }
        ranked.append(entry)
    ranked.sort(key=lambda r: r[f"{RANKING_METRIC}_mean"], reverse=True)

    print(f"\n=== Phase 3b -- {len(ranked)} combo-x-recipe entries, ranked by val {RANKING_METRIC} ===")
    for i, r in enumerate(ranked, start=1):
        print(f"  {i:2d}. {r['entry_label']:<40} "
              f"val {RANKING_METRIC}={r[f'{RANKING_METRIC}_mean']:.4f}+-{r[f'{RANKING_METRIC}_std']:.4f}  "
              f"test {RANKING_METRIC}={r[f'test_{RANKING_METRIC}_mean']:.4f}+-{r[f'test_{RANKING_METRIC}_std']:.4f}  "
              f"(n={r['n_runs']})")

    print("\n=== current vs. old_resultsold, per combo (val F1 mean) ===")
    by_combo: Dict[str, Dict[str, float]] = {}
    for r in ranked:
        combo_key = combo_label(r["active_extractors"])
        by_combo.setdefault(combo_key, {})[r["recipe"]] = r[f"{RANKING_METRIC}_mean"]
    for combo_key, vals in by_combo.items():
        cur = vals.get("current")
        old = vals.get("old_resultsold")
        if cur is not None and old is not None:
            print(f"  {combo_key:<30} current={cur:.4f}  old_resultsold={old:.4f}  delta={old - cur:+.4f}")

    with open(TOP_JSON, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "seeds": SEEDS, "results": ranked}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {TOP_JSON}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3b: does results_old's winning hyperparameter recipe beat "
                     "FINAL_HPARAMS under identity-off, source-disjoint evaluation?"
    )
    parser.add_argument("--run", action="store_true", help="Run the sweep (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate results/phase3b_recipe_check.jsonl")
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
