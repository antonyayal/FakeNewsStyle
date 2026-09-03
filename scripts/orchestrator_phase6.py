# scripts/orchestrator_phase6.py
# -*- coding: utf-8 -*-
"""
Phase 6: identity-free context control -- checks how much of context's
apparent contribution (and of any combo that includes it) is Source/Domain
outlet-memorization rather than genuine contextual signal. See
dataset_source_label_leakage memory and README's "Known Limitations &
Caveats": ~97% of train Sources publish only one label, so context's hash
embedding of Source/Domain lets the model shortcut "this outlet always
publishes Fake" instead of learning anything from Topic/age.

Re-runs, with context's Source/Domain embeddings switched off
(--context_source_dim 0 --context_domain_dim 0, keeping only Topic+age+
flags -- PHASE6_CONTEXT_SOURCE_DIM/DOMAIN_DIM in experiment_config.py):
  Stage A (Phase 1 equivalent): context alone, dim swept over
    PHASE6_CONTEXT_DIM_CANDIDATES, 5 seeds each -- picks the best
    identity-free context dim (results/phase6_context_top.json).
  Stage B (Phase 2 equivalent): the 15 on/off combos of
    {semantic,emotion,style,context}, each branch at its Phase 1 rank-1 dim
    (results/phase1_top.json) except context, which uses Stage A's winner,
    5 seeds each -- final ranking in results/phase6_top.json.

Fully isolated from the shared default cache Phase 1-5 read (raw features,
VAE, merged latents all live under *_phase6 paths via main.py's
--context_output_dir/--context_vae_input_dir/--vae_data_output_dir/
--vae_model_output_dir/--merge_output_dir) -- can run interleaved with
Phase 1-5 without touching them. Only "context" is retrained here; the
other 3 branches reuse Phase 1's already-cached shared VAE latents as-is.

Usage:
    python scripts/orchestrator_phase6.py --run --dry-run   # review the plan
    python scripts/orchestrator_phase6.py --run              # execute (resumable)
    python scripts/orchestrator_phase6.py --summary           # re-aggregate only
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_results import aggregate_by_config, load_runs  # noqa: E402
from experiment_config import (  # noqa: E402
    ALL_MODALITIES,
    BASE_DIR,
    DEFAULT_LATENT_DIM,
    FIXED_KAN_BASELINE,
    PHASE1_RESULTS_JSONL,
    PHASE1_TOP_JSON,
    PHASE3_BASELINE,
    PHASE4_N_FOLDS,
    PHASE4_SPLIT_SEED,
    PHASE4_TOP_JSON,
    PHASE5_N_FOLDS,
    PHASE5_SPLIT_SEED,
    PHASE5_TOP_JSON,
    PHASE3_CANDIDATES,
    PHASE3_VAE_DATA_DIR,
    PHASE3_VAE_MODEL_DIR,
    PHASE6_CONTEXT_DIM_CANDIDATES,
    PHASE6_CONTEXT_DOMAIN_DIM,
    PHASE6_CONTEXT_SOURCE_DIM,
    PHASE6_CONTEXT_TOP_JSON,
    PHASE6_KAN_RUNS_DIR,
    PHASE6_MERGED_DIR,
    PHASE6_RAW_DIR,
    PHASE6_RESULTS_JSONL,
    PHASE6_STAGE_C_TOP_JSON,
    PHASE6_STAGE_D_KAN_RUNS_DIR,
    PHASE6_STAGE_D_MERGED_DIR,
    PHASE6_STAGE_D_PER_FOLD_JSON,
    PHASE6_STAGE_D_RAW_DIR,
    PHASE6_STAGE_D_RESULTS_JSONL,
    PHASE6_STAGE_D_TOP_JSON,
    PHASE6_STAGE_D_VAE_DATA_DIR,
    PHASE6_STAGE_D_VAE_MODEL_DIR,
    PHASE6_STAGE_E_KAN_RUNS_DIR,
    PHASE6_STAGE_E_MERGED_DIR,
    PHASE6_STAGE_E_PER_FOLD_JSON,
    PHASE6_STAGE_E_RAW_DIR,
    PHASE6_STAGE_E_RESULTS_JSONL,
    PHASE6_STAGE_E_TOP_JSON,
    PHASE6_STAGE_E_VAE_DATA_DIR,
    PHASE6_STAGE_E_VAE_MODEL_DIR,
    PHASE6_TOP_JSON,
    PHASE6_TOP_K,
    PHASE6_VAE_DATA_DIR,
    PHASE6_VAE_MODEL_DIR,
    RANKING_METRIC,
    SEEDS,
    VAE_LATENTS_DIR,
)
from experiment_runner import (  # noqa: E402
    execute_and_log,
    is_default_vae_reg,
    latent_cache_is_fresh,
    load_ok_run_keys,
    merge_latents_manual,
    python_executable,
    resolve_kan_input,
    run_main_command,
)
from orchestrator_phase2 import load_phase1_dims  # noqa: E402

CONTEXT_RAW_DIR = PHASE6_RAW_DIR / "context"


def config_label(combo) -> str:
    return "_".join(m for m in ALL_MODALITIES if m in combo)


def extract_identity_free_context(dry_run: bool) -> None:
    cmd = [
        python_executable(), "main.py", "--extract_context",
        "--context_output_dir", str(CONTEXT_RAW_DIR.relative_to(BASE_DIR)),
        "--context_source_dim", str(PHASE6_CONTEXT_SOURCE_DIM),
        "--context_domain_dim", str(PHASE6_CONTEXT_DOMAIN_DIM),
    ]
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        print("  (dry-run: not executing)")
        return
    outcome = run_main_command(cmd, require_results_json=False)
    if outcome["error"] is not None:
        raise RuntimeError(f"extract_context (identity-free) failed: {outcome['error']}")
    print(f"  OK in {outcome['elapsed_seconds']}s")


def ensure_context_vae(dim: int, dry_run: bool) -> None:
    if not dry_run and latent_cache_is_fresh("context", dim, PHASE6_VAE_DATA_DIR):
        print(f"  context latent{dim}: already cached, skip")
        return
    cmd = [
        python_executable(), "main.py", "--run_vaes",
        "--exclude_semantic", "--exclude_emotion", "--exclude_style",
        "--context_latent_dim", str(dim),
        "--context_vae_input_dir", str(CONTEXT_RAW_DIR.relative_to(BASE_DIR)),
        "--vae_data_output_dir", str(PHASE6_VAE_DATA_DIR.relative_to(BASE_DIR)),
        "--vae_model_output_dir", str(PHASE6_VAE_MODEL_DIR.relative_to(BASE_DIR)),
    ]
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        print("  (dry-run: not executing)")
        return
    outcome = run_main_command(cmd, require_results_json=False)
    if outcome["error"] is not None:
        raise RuntimeError(f"context VAE (identity-free) @ dim={dim} failed: {outcome['error']}")
    print(f"  OK in {outcome['elapsed_seconds']}s")


def build_kan_command(combo, latent_dims, pkl_paths, seed, output_dir) -> list:
    cmd = [python_executable(), "main.py", "--train_kan"]
    for branch in ALL_MODALITIES:
        if branch not in combo:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(latent_dims.get(branch, DEFAULT_LATENT_DIM[branch]))]
    cmd += [
        "--kan_train_pkl", str(pkl_paths["train"]),
        "--kan_val_pkl", str(pkl_paths["val"]),
        "--kan_test_pkl", str(pkl_paths["test"]),
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
    ]
    return cmd


def run_stage_a(dry_run: bool) -> None:
    print("\n=== Phase 6 / Stage A -- identity-free context, dim sweep ===")
    extract_identity_free_context(dry_run)
    for dim in PHASE6_CONTEXT_DIM_CANDIDATES:
        ensure_context_vae(dim, dry_run)

    ok_keys = load_ok_run_keys(PHASE6_RESULTS_JSONL) if not dry_run else set()
    total = len(PHASE6_CONTEXT_DIM_CANDIDATES) * len(SEEDS)
    idx = 0
    for dim in PHASE6_CONTEXT_DIM_CANDIDATES:
        latent_dir = PHASE6_VAE_DATA_DIR / "context" / f"latent{dim}"
        merged_dir = PHASE6_MERGED_DIR / "stageA_context_only" / f"dim{dim}"
        pkl_paths = (
            {s: merged_dir / f"{s}.pkl" for s in ["train", "val", "test"]}
            if dry_run else merge_latents_manual(["context"], {"context": latent_dir}, merged_dir)
        )
        for seed in SEEDS:
            idx += 1
            key = f"stageA__dim{dim}__seed{seed}"
            cmd = build_kan_command(
                ["context"], {"context": dim}, pkl_paths, seed,
                PHASE6_KAN_RUNS_DIR / "stageA" / f"dim{dim}" / f"seed{seed}",
            )
            if dry_run:
                print(f"[{idx:03d}/{total}] {key}\n  $ {' '.join(cmd)}")
                continue
            if key in ok_keys:
                print(f"[{idx:03d}/{total}] {key} SKIP (already ok)")
                continue
            print(f"[{idx:03d}/{total}] {key} RUN")
            record = execute_and_log(
                run_key=key, cmd=cmd, jsonl_path=PHASE6_RESULTS_JSONL,
                meta={"stage": "A", "branch": "context", "dim": dim,
                      "active_extractors": ["context"], "seed": seed},
            )
            print(f"  {record['status']} ({record.get('elapsed_seconds')}s)")


def pick_best_context_dim() -> int:
    df = load_runs(PHASE6_RESULTS_JSONL)
    df = df[df.get("stage") == "A"] if not df.empty else df
    if df.empty:
        raise RuntimeError("No Stage A runs logged -- run Stage A first.")

    ranking = aggregate_by_config(df, group_by="branch_dim", metric=RANKING_METRIC)
    entries = []
    for _, row in ranking.head(PHASE6_TOP_K).iterrows():
        dim = int(row["config"].split("::dim", 1)[1])
        test_col = f"test_{RANKING_METRIC}_mean"
        entries.append({
            "dim": dim,
            f"{RANKING_METRIC}_mean": float(row[f"{RANKING_METRIC}_mean"]),
            f"{RANKING_METRIC}_std": float(row[f"{RANKING_METRIC}_std"]),
            "n_runs": int(row[f"{RANKING_METRIC}_count"]),
            **({f"test_{RANKING_METRIC}_mean": float(row[test_col])} if test_col in row and pd.notna(row[test_col]) else {}),
        })

    PHASE6_CONTEXT_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE6_CONTEXT_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "top_k": PHASE6_TOP_K, "entries": entries}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {PHASE6_CONTEXT_TOP_JSON}")

    best = entries[0]
    print(f"Best identity-free context dim: {best['dim']} (val {RANKING_METRIC}={best[f'{RANKING_METRIC}_mean']:.4f})")
    return best["dim"]


def run_stage_b(best_context_dim: int, dry_run: bool) -> None:
    print(f"\n=== Phase 6 / Stage B -- 15 combos, identity-free context @ dim={best_context_dim} ===")
    other_dims = load_phase1_dims()
    ok_keys = load_ok_run_keys(PHASE6_RESULTS_JSONL) if not dry_run else set()

    all_combos = []
    for r in range(1, len(ALL_MODALITIES) + 1):
        all_combos.extend(itertools.combinations(ALL_MODALITIES, r))

    context_latent_dir = PHASE6_VAE_DATA_DIR / "context" / f"latent{best_context_dim}"
    total = len(all_combos) * len(SEEDS)
    idx = 0

    for combo in all_combos:
        combo = list(combo)
        label = "_".join(combo)
        latent_dims = {b: (best_context_dim if b == "context" else other_dims[b]) for b in combo}
        latent_dirs = {
            b: (context_latent_dir if b == "context" else VAE_LATENTS_DIR / b / f"latent{other_dims[b]}")
            for b in combo
        }

        merged_dir = PHASE6_MERGED_DIR / "stageB_combos" / label
        pkl_paths = (
            {s: merged_dir / f"{s}.pkl" for s in ["train", "val", "test"]}
            if dry_run else merge_latents_manual(combo, latent_dirs, merged_dir)
        )

        for seed in SEEDS:
            idx += 1
            key = f"stageB__{label}__seed{seed}"
            cmd = build_kan_command(
                combo, latent_dims, pkl_paths, seed,
                PHASE6_KAN_RUNS_DIR / "stageB" / label / f"seed{seed}",
            )
            if dry_run:
                print(f"[{idx:03d}/{total}] {key}\n  $ {' '.join(cmd)}")
                continue
            if key in ok_keys:
                print(f"[{idx:03d}/{total}] {key} SKIP (already ok)")
                continue
            print(f"[{idx:03d}/{total}] {key} RUN")
            record = execute_and_log(
                run_key=key, cmd=cmd, jsonl_path=PHASE6_RESULTS_JSONL,
                meta={"stage": "B", "active_extractors": combo, "seed": seed},
            )
            print(f"  {record['status']} ({record.get('elapsed_seconds')}s)")


def summarize() -> None:
    df = load_runs(PHASE6_RESULTS_JSONL)
    if df.empty:
        print("No successful runs logged yet -- nothing to rank.")
        return
    dfb = df[df.get("stage") == "B"]
    if dfb.empty:
        print("Stage B not run yet -- nothing to rank.")
        return

    ranking = aggregate_by_config(dfb, group_by="extractors", metric=RANKING_METRIC)
    print(f"\n=== Phase 6 final ranking (identity-free context, by val {RANKING_METRIC}) ===")
    entries = []
    for i, row in ranking.iterrows():
        combo = [m for m in ALL_MODALITIES if m in row["config"].split("+")]
        test_col = f"test_{RANKING_METRIC}_mean"
        test_str = f"  (test {RANKING_METRIC}={row[test_col]:.4f})" if test_col in row and pd.notna(row[test_col]) else ""
        tag = "  <-- ALL FOUR" if len(combo) == 4 else ""
        print(f"  {i + 1}. {row['config']:40s} val {RANKING_METRIC}={row[f'{RANKING_METRIC}_mean']:.4f} "
              f"+/- {row[f'{RANKING_METRIC}_std']:.4f} (n={int(row[f'{RANKING_METRIC}_count'])}){test_str}{tag}")
        entries.append({
            "active_extractors": combo,
            f"{RANKING_METRIC}_mean": float(row[f"{RANKING_METRIC}_mean"]),
            f"{RANKING_METRIC}_std": float(row[f"{RANKING_METRIC}_std"]),
            "n_runs": int(row[f"{RANKING_METRIC}_count"]),
            **({f"test_{RANKING_METRIC}_mean": float(row[test_col])} if test_col in row and pd.notna(row[test_col]) else {}),
        })

    PHASE6_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE6_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "metric": RANKING_METRIC,
            "context_source_dim": PHASE6_CONTEXT_SOURCE_DIM,
            "context_domain_dim": PHASE6_CONTEXT_DOMAIN_DIM,
            "ranking": entries,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {PHASE6_TOP_JSON}")


# =============================================================================
# Stage C -- Phase 3 equivalent: VAE-reg + KAN hyperparameters on Stage B's
# top 5 (config, itself pooled with Stage A's top identity-free context dims
# and the original Phase 1's top semantic/emotion/style dims, mirroring
# orchestrator_phase2.py's _phase1_pool_rows()).
# =============================================================================

def load_stageB_pool() -> pd.DataFrame:
    dfB = load_runs(PHASE6_RESULTS_JSONL)
    dfB = dfB[dfB.get("stage") == "B"] if not dfB.empty else dfB

    dfA = load_runs(PHASE6_RESULTS_JSONL)
    dfA = dfA[dfA.get("stage") == "A"] if not dfA.empty else dfA
    if not dfA.empty and PHASE6_CONTEXT_TOP_JSON.exists():
        with open(PHASE6_CONTEXT_TOP_JSON, encoding="utf-8") as f:
            top_dims = {e["dim"] for e in json.load(f)["entries"]}
        dfA = dfA[dfA["dim"].isin(top_dims)]

    df1 = load_runs(PHASE1_RESULTS_JSONL)
    if not df1.empty:
        df1 = df1[df1["branch"] != "context"]
        if PHASE1_TOP_JSON.exists():
            with open(PHASE1_TOP_JSON, encoding="utf-8") as f:
                top1 = json.load(f)
            winning_pairs = {
                (b, e["dim"]) for b, es in top1["by_branch"].items() for e in es if b != "context"
            }
            df1 = df1[df1.apply(lambda r: (r.get("branch"), r.get("dim")) in winning_pairs, axis=1)]

    parts = [d for d in [dfB, dfA, df1] if not d.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_stageC_configs() -> list:
    pool = load_stageB_pool()
    if pool.empty:
        raise RuntimeError("No Stage B (or pool) runs logged -- run Stage A+B first.")
    ranking = aggregate_by_config(pool, group_by="extractors", metric=RANKING_METRIC)
    top5 = ranking.head(5)

    other_dims = load_phase1_dims()
    if not PHASE6_CONTEXT_TOP_JSON.exists():
        raise RuntimeError(f"{PHASE6_CONTEXT_TOP_JSON} not found -- run Stage A first.")
    with open(PHASE6_CONTEXT_TOP_JSON, encoding="utf-8") as f:
        best_ctx_dim = json.load(f)["entries"][0]["dim"]

    configs = []
    for _, row in top5.iterrows():
        combo = [m for m in ALL_MODALITIES if m in row["config"].split("+")]
        latent_dims = {b: (best_ctx_dim if b == "context" else other_dims[b]) for b in combo}
        configs.append({"active_extractors": combo, "latent_dims": latent_dims})
    return configs


def resolve_branch_dirs(combo, label: str, variant_label: str, effective: dict, latent_dims: dict, dry_run: bool) -> dict:
    """{branch: latent_dir} for every branch in combo. Non-context branches
    go through Phase 3's exact shared-cache/isolated-VAE mechanism
    (resolve_kan_input) -- its merged-pkl return is discarded, we only need
    training triggered so we can compute each branch's own dir with the same
    convention it uses internally. "context" is always trained isolated on
    the identity-free raw features, regardless of variant."""
    dirs = {}
    non_context = [b for b in combo if b != "context"]

    if non_context:
        resolve_kan_input(
            non_context, label, variant_label, {**effective, "latent": latent_dims},
            PHASE3_VAE_DATA_DIR, PHASE3_VAE_MODEL_DIR,
            PHASE6_MERGED_DIR / "stageC_scratch" / label / variant_label,
            dry_run,
        )
        if is_default_vae_reg(effective):
            for b in non_context:
                dirs[b] = VAE_LATENTS_DIR / b / f"latent{latent_dims[b]}"
        else:
            tag = f"beta{effective['vae_beta']}_drop{effective['vae_dropout']}"
            for b in non_context:
                dirs[b] = PHASE3_VAE_DATA_DIR / label / tag / b / f"latent{latent_dims[b]}"

    if "context" in combo:
        ctx_dim = latent_dims["context"]
        tag = f"beta{effective['vae_beta']}_drop{effective['vae_dropout']}"
        ctx_data_dir = PHASE6_VAE_DATA_DIR / "stageC" / tag
        ctx_model_dir = PHASE6_VAE_MODEL_DIR / "stageC" / tag
        if dry_run:
            print(f"    [context, isolated] would ensure VAE @ dim={ctx_dim} beta={effective['vae_beta']} "
                  f"dropout={effective['vae_dropout']} under {ctx_data_dir}")
        elif not latent_cache_is_fresh("context", ctx_dim, ctx_data_dir, raw_dir_override=CONTEXT_RAW_DIR):
            cmd = [
                python_executable(), "main.py", "--run_vaes",
                "--exclude_semantic", "--exclude_emotion", "--exclude_style",
                "--context_latent_dim", str(ctx_dim),
                "--context_vae_input_dir", str(CONTEXT_RAW_DIR.relative_to(BASE_DIR)),
                "--vae_beta", str(effective["vae_beta"]),
                "--vae_dropout", str(effective["vae_dropout"]),
                "--vae_data_output_dir", str(ctx_data_dir.relative_to(BASE_DIR)),
                "--vae_model_output_dir", str(ctx_model_dir.relative_to(BASE_DIR)),
            ]
            print(f"    $ {' '.join(cmd)}")
            outcome = run_main_command(cmd, require_results_json=False)
            if outcome["error"] is not None:
                raise RuntimeError(f"context VAE (Stage C, isolated) failed: {outcome['error']}")
            print(f"    OK in {outcome['elapsed_seconds']}s")
        dirs["context"] = ctx_data_dir / "context" / f"latent{ctx_dim}"

    return dirs


def build_stageC_kan_command(combo, latent_dims: dict, effective: dict, pkl_paths: dict, seed: int, output_dir: Path) -> list:
    cmd = [python_executable(), "main.py", "--train_kan"]
    for branch in ALL_MODALITIES:
        if branch not in combo:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(latent_dims.get(branch, DEFAULT_LATENT_DIM[branch]))]
    cmd += [
        "--kan_train_pkl", str(pkl_paths["train"]),
        "--kan_val_pkl", str(pkl_paths["val"]),
        "--kan_test_pkl", str(pkl_paths["test"]),
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
        "--vae_beta", str(effective["vae_beta"]),
        "--vae_dropout", str(effective["vae_dropout"]),
    ]
    return cmd


def run_stage_c(dry_run: bool) -> None:
    configs = load_stageC_configs()
    total = len(configs) * len(PHASE3_CANDIDATES) * len(SEEDS)
    print(f"\n=== Phase 6 / Stage C -- VAE-reg + KAN hyperparams on Stage B's top 5 ===")
    print(f"{len(configs)} configs x {len(PHASE3_CANDIDATES)} variants x {len(SEEDS)} seeds = {total} runs")

    ok_keys = load_ok_run_keys(PHASE6_RESULTS_JSONL) if not dry_run else set()
    idx = 0
    for cfg in configs:
        combo = cfg["active_extractors"]
        latent_dims = cfg["latent_dims"]
        label = config_label(combo)

        for variant_label, override in PHASE3_CANDIDATES.items():
            effective = {**PHASE3_BASELINE, **override}
            latent_dirs = resolve_branch_dirs(combo, label, variant_label, effective, latent_dims, dry_run)
            merged_dir = PHASE6_MERGED_DIR / "stageC" / label / variant_label
            pkl_paths = (
                {s: merged_dir / f"{s}.pkl" for s in ["train", "val", "test"]}
                if dry_run else merge_latents_manual(combo, latent_dirs, merged_dir)
            )

            for seed in SEEDS:
                idx += 1
                key = f"stageC__{label}__{variant_label}__seed{seed}"
                cmd = build_stageC_kan_command(
                    combo, latent_dims, effective, pkl_paths, seed,
                    PHASE6_KAN_RUNS_DIR / "stageC" / label / variant_label / f"seed{seed}",
                )
                if dry_run:
                    print(f"[{idx:04d}/{total}] {key}\n  $ {' '.join(cmd)}")
                    continue
                if key in ok_keys:
                    print(f"[{idx:04d}/{total}] {key} SKIP (already ok)")
                    continue
                print(f"[{idx:04d}/{total}] {key} RUN")
                record = execute_and_log(
                    run_key=key, cmd=cmd, jsonl_path=PHASE6_RESULTS_JSONL,
                    meta={"stage": "C", "config_label": label, "group": label,
                          "candidate_label": variant_label, "active_extractors": combo,
                          "seed": seed, "overrides": effective},
                )
                print(f"  {record['status']} ({record.get('elapsed_seconds')}s)")


def summarize_stage_c() -> list:
    df = load_runs(PHASE6_RESULTS_JSONL)
    dfc = df[df.get("stage") == "C"] if not df.empty else df
    if dfc.empty:
        print("Stage C not run yet -- nothing to rank.")
        return []

    ranking = aggregate_by_config(dfc, group_by="candidate", metric=RANKING_METRIC)
    top5 = ranking.head(5)
    other_dims = load_phase1_dims()
    with open(PHASE6_CONTEXT_TOP_JSON, encoding="utf-8") as f:
        best_ctx_dim = json.load(f)["entries"][0]["dim"]

    print(f"\n=== Phase 6 / Stage C top 5 (by val {RANKING_METRIC}) ===")
    entries = []
    for i, row in top5.iterrows():
        label, variant_label = row["config"].split("::", 1)
        combo = [m for m in ALL_MODALITIES if m in label.split("_")]
        effective = {**PHASE3_BASELINE, **PHASE3_CANDIDATES[variant_label]}
        latent_dims = {b: (best_ctx_dim if b == "context" else other_dims[b]) for b in combo}
        test_col = f"test_{RANKING_METRIC}_mean"
        test_str = f"  (test {RANKING_METRIC}={row[test_col]:.4f})" if test_col in row and pd.notna(row[test_col]) else ""
        print(f"  {i + 1}. {label} / {variant_label}  val {RANKING_METRIC}={row[f'{RANKING_METRIC}_mean']:.4f} "
              f"+/- {row[f'{RANKING_METRIC}_std']:.4f} (n={int(row[f'{RANKING_METRIC}_count'])}){test_str}")
        entries.append({
            "active_extractors": combo,
            "latent_dims": latent_dims,
            "variant_label": variant_label,
            **effective,
            **FIXED_KAN_BASELINE,
            f"{RANKING_METRIC}_mean": float(row[f"{RANKING_METRIC}_mean"]),
            f"{RANKING_METRIC}_std": float(row[f"{RANKING_METRIC}_std"]),
            "n_runs": int(row[f"{RANKING_METRIC}_count"]),
            **({f"test_{RANKING_METRIC}_mean": float(row[test_col])} if test_col in row and pd.notna(row[test_col]) else {}),
        })

    PHASE6_STAGE_C_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE6_STAGE_C_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "top": entries}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {PHASE6_STAGE_C_TOP_JSON}")
    return entries


# =============================================================================
# Stage D/E -- Phase 4/5 equivalent: kfold / source-disjoint fold validation
# of Stage C's top 5, with identity-free context.
#
# SAFETY: non-context branches reuse the SAME shared fold-aware cache Phase
# 4 (data/03_features_raw_cv, data/05_vae_latents_cv) / Phase 5
# (data/03_features_raw_source_cv, data/05_vae_latents_source_cv) read and
# OVERWRITE per fold (see fold_validation.py's prepare_fold: it assumes only
# one process touches that cache at a time). Running Stage D/E while the
# real Phase 4/5 is still executing would race on that shared cache and
# silently corrupt either run. require_phase_done() below refuses to start
# unless results/phase{4,5}_top.json already exist -- run Stage D/E only
# after Phase 4/5 have fully finished, never concurrently.
# =============================================================================

def require_phase_done(top_json: Path, phase_label: str) -> None:
    if not top_json.exists():
        raise RuntimeError(
            f"{top_json} does not exist yet -- {phase_label} must FINISH (not just start) "
            f"before Stage D/E can safely run, since both share the same per-fold cache "
            f"({phase_label} overwrites it fold-by-fold and assumes no concurrent reader/writer)."
        )


def _corpus_mode_flags(corpus_mode: str, n_folds: int, split_seed: int, fold_idx: int) -> list:
    if corpus_mode == "kfold":
        return ["--corpus_mode", "kfold", "--kfold_n", str(n_folds),
                 "--kfold_index", str(fold_idx), "--kfold_split_seed", str(split_seed)]
    elif corpus_mode == "source_disjoint":
        return ["--corpus_mode", "source_disjoint", "--source_split_n", str(n_folds),
                 "--source_split_index", str(fold_idx), "--source_split_seed", str(split_seed)]
    else:
        raise ValueError(f"Unknown corpus_mode: {corpus_mode!r}")


def _shared_fold_dirs(corpus_mode: str):
    if corpus_mode == "kfold":
        return (BASE_DIR / "data" / "03_features_raw_cv", BASE_DIR / "data" / "05_vae_latents_cv")
    return (BASE_DIR / "data" / "03_features_raw_source_cv", BASE_DIR / "data" / "05_vae_latents_source_cv")


def prepare_fold_phase6(
    *, corpus_mode: str, n_folds: int, split_seed: int, fold_idx: int, cfg: dict,
    label: str, ctx_raw_base: Path, ctx_vae_data_base: Path, ctx_vae_model_base: Path,
    dry_run: bool,
) -> dict:
    combo, latent_dims = cfg["active_extractors"], cfg["latent_dims"]
    non_context = [b for b in combo if b != "context"]
    shared_raw, shared_vae = _shared_fold_dirs(corpus_mode)

    if non_context:
        cmd = [python_executable(), "main.py",
               *_corpus_mode_flags(corpus_mode, n_folds, split_seed, fold_idx),
               "--preprocess_text", "--exclude_context",
               "--context_latent_dim", str(latent_dims.get("context", DEFAULT_LATENT_DIM["context"]))]
        for branch in ALL_MODALITIES:
            if branch == "context":
                continue
            cmd.append(f"--extract_{branch}" if branch in combo else f"--exclude_{branch}")
            cmd += [f"--{branch}_latent_dim", str(latent_dims.get(branch, DEFAULT_LATENT_DIM[branch]))]
        cmd += ["--run_vaes", "--vae_beta", str(cfg["vae_beta"]), "--vae_dropout", str(cfg["vae_dropout"])]
        print(f"  [{label}] fold{fold_idx}: preparing non-context branches (shared {corpus_mode} cache)")
        print(f"    $ {' '.join(cmd)}")
        if not dry_run:
            outcome = run_main_command(cmd, require_results_json=False)
            if outcome["error"] is not None:
                raise RuntimeError(f"Fold {fold_idx} [{label}]: non-context prep failed: {outcome['error']}")
            print(f"    OK in {outcome['elapsed_seconds']}s")

    latent_dirs = {b: shared_vae / b / f"latent{latent_dims[b]}" for b in non_context}

    if "context" in combo:
        ctx_dim = latent_dims["context"]
        fold_raw = ctx_raw_base / f"fold{fold_idx}"
        fold_vae_data = ctx_vae_data_base / f"fold{fold_idx}"
        fold_vae_model = ctx_vae_model_base / f"fold{fold_idx}"

        extract_cmd = [
            python_executable(), "main.py",
            *_corpus_mode_flags(corpus_mode, n_folds, split_seed, fold_idx), "--extract_context",
            "--context_output_dir", str(fold_raw.relative_to(BASE_DIR)),
            "--context_source_dim", str(PHASE6_CONTEXT_SOURCE_DIM),
            "--context_domain_dim", str(PHASE6_CONTEXT_DOMAIN_DIM),
        ]
        print(f"  [{label}] fold{fold_idx}: extracting identity-free context (isolated)")
        print(f"    $ {' '.join(extract_cmd)}")
        if not dry_run:
            outcome = run_main_command(extract_cmd, require_results_json=False)
            if outcome["error"] is not None:
                raise RuntimeError(f"Fold {fold_idx} [{label}]: context extraction failed: {outcome['error']}")
            print(f"    OK in {outcome['elapsed_seconds']}s")

        vae_cmd = [
            python_executable(), "main.py", "--run_vaes",
            "--exclude_semantic", "--exclude_emotion", "--exclude_style",
            "--context_latent_dim", str(ctx_dim),
            "--context_vae_input_dir", str(fold_raw.relative_to(BASE_DIR)),
            "--vae_beta", str(cfg["vae_beta"]), "--vae_dropout", str(cfg["vae_dropout"]),
            "--vae_data_output_dir", str(fold_vae_data.relative_to(BASE_DIR)),
            "--vae_model_output_dir", str(fold_vae_model.relative_to(BASE_DIR)),
        ]
        print(f"  [{label}] fold{fold_idx}: isolated identity-free context VAE")
        print(f"    $ {' '.join(vae_cmd)}")
        if dry_run:
            pass
        elif not latent_cache_is_fresh("context", ctx_dim, fold_vae_data, raw_dir_override=fold_raw):
            outcome = run_main_command(vae_cmd, require_results_json=False)
            if outcome["error"] is not None:
                raise RuntimeError(f"Fold {fold_idx} [{label}]: context VAE failed: {outcome['error']}")
            print(f"    OK in {outcome['elapsed_seconds']}s")
        else:
            print("    already cached, skip")
        latent_dirs["context"] = fold_vae_data / "context" / f"latent{ctx_dim}"

    return latent_dirs


def build_fold_kan_command(
    corpus_mode: str, n_folds: int, split_seed: int, fold_idx: int,
    cfg: dict, pkl_paths: dict, seed: int, output_dir: Path,
) -> list:
    combo, latent_dims = cfg["active_extractors"], cfg["latent_dims"]
    cmd = [python_executable(), "main.py",
           *_corpus_mode_flags(corpus_mode, n_folds, split_seed, fold_idx), "--train_kan"]
    for branch in ALL_MODALITIES:
        if branch not in combo:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(latent_dims.get(branch, DEFAULT_LATENT_DIM[branch]))]
    cmd += [
        "--kan_train_pkl", str(pkl_paths["train"]),
        "--kan_val_pkl", str(pkl_paths["val"]),
        "--kan_test_pkl", str(pkl_paths["test"]),
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


def run_fold_stage(
    *, corpus_mode: str, n_folds: int, split_seed: int, configs: list,
    results_jsonl: Path, kan_runs_dir: Path, merged_dir: Path,
    ctx_raw_base: Path, ctx_vae_data_base: Path, ctx_vae_model_base: Path,
    phase_name: str, dry_run: bool,
) -> None:
    total = len(configs) * n_folds * len(SEEDS)
    print(f"\n=== Phase 6 / {phase_name}: {len(configs)} configs x {n_folds} folds x {len(SEEDS)} seeds = {total} runs ===")
    ok_keys = load_ok_run_keys(results_jsonl) if not dry_run else set()
    idx = 0

    for cfg in configs:
        label = f"{config_label(cfg['active_extractors'])}__{cfg['variant_label']}"
        for fold_idx in range(n_folds):
            print(f"\n--- {phase_name} -- {label} -- fold {fold_idx}/{n_folds - 1} ---")
            latent_dirs = prepare_fold_phase6(
                corpus_mode=corpus_mode, n_folds=n_folds, split_seed=split_seed, fold_idx=fold_idx,
                cfg=cfg, label=label, ctx_raw_base=ctx_raw_base, ctx_vae_data_base=ctx_vae_data_base,
                ctx_vae_model_base=ctx_vae_model_base, dry_run=dry_run,
            )
            fold_merged_dir = merged_dir / f"fold{fold_idx}" / label
            pkl_paths = (
                {s: fold_merged_dir / f"{s}.pkl" for s in ["train", "val", "test"]}
                if dry_run else merge_latents_manual(cfg["active_extractors"], latent_dirs, fold_merged_dir)
            )

            for seed in SEEDS:
                idx += 1
                key = f"{label}__fold{fold_idx}__seed{seed}"
                cmd = build_fold_kan_command(
                    corpus_mode, n_folds, split_seed, fold_idx, cfg, pkl_paths, seed,
                    kan_runs_dir / label / f"fold{fold_idx}" / f"seed{seed}",
                )
                if dry_run:
                    print(f"    [{idx:04d}/{total}] {key}\n      $ {' '.join(cmd)}")
                    continue
                if key in ok_keys:
                    print(f"    [{idx:04d}/{total}] {key} SKIP (already ok)")
                    continue
                print(f"    [{idx:04d}/{total}] {key} RUN")
                record = execute_and_log(
                    run_key=key, cmd=cmd, jsonl_path=results_jsonl,
                    meta={"phase": phase_name, "entry_label": label, "fold": fold_idx, "seed": seed,
                          "active_extractors": cfg["active_extractors"], "config": cfg},
                )
                print(f"      {record['status']} ({record.get('elapsed_seconds')}s)")

    if not dry_run:
        print(f"\n{phase_name} complete (this invocation). "
              f"Total accumulated in {results_jsonl}: {len(load_ok_run_keys(results_jsonl))}/{total} ok.")


def summarize_fold_stage(*, results_jsonl: Path, per_fold_json: Path, top_json: Path, phase_name: str) -> None:
    if not results_jsonl.exists():
        print(f"No results yet: {results_jsonl}")
        return

    rows = []
    with open(results_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok" or not r.get("metrics"):
                continue
            rows.append({
                "entry_label": r["entry_label"], "fold": r["fold"], "seed": r["seed"],
                "active_extractors": r.get("active_extractors"), "config": r.get("config"),
                **r["metrics"],
                **{f"test_{k}": v for k, v in (r.get("test_metrics") or {}).items()},
            })

    if not rows:
        print("No successful runs logged yet.")
        return

    df = pd.DataFrame(rows)
    test_col = f"test_{RANKING_METRIC}"
    has_test = test_col in df.columns

    per_label_summary = {}
    global_rows = []
    for label, group in df.groupby("entry_label"):
        print(f"\n=== {label} -- per-fold val {RANKING_METRIC} (n={len(SEEDS)} seeds each) ===")
        per_fold = group.groupby("fold")[RANKING_METRIC].agg(["mean", "std", "min", "max", "count"])
        print(per_fold.to_string())

        global_stats = {
            f"{RANKING_METRIC}_mean": float(group[RANKING_METRIC].mean()),
            f"{RANKING_METRIC}_std": float(group[RANKING_METRIC].std()),
            f"{RANKING_METRIC}_min": float(group[RANKING_METRIC].min()),
            f"{RANKING_METRIC}_max": float(group[RANKING_METRIC].max()),
            "n_runs": int(len(group)), "n_folds": int(group["fold"].nunique()),
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

    winner = max(global_rows, key=lambda r: r[f"{RANKING_METRIC}_mean"])
    print(f"\n=== {phase_name} global winner: {winner['entry_label']} "
          f"({RANKING_METRIC}_mean={winner[f'{RANKING_METRIC}_mean']:.4f}) ===")

    top_json.parent.mkdir(parents=True, exist_ok=True)
    with open(top_json, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "winner": winner}, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {top_json}")


def run_stage_d(dry_run: bool) -> None:
    if not dry_run:
        require_phase_done(PHASE4_TOP_JSON, "Phase 4")
    configs = summarize_stage_c() if not PHASE6_STAGE_C_TOP_JSON.exists() else json.load(open(PHASE6_STAGE_C_TOP_JSON))["top"]
    run_fold_stage(
        corpus_mode="kfold", n_folds=PHASE4_N_FOLDS, split_seed=PHASE4_SPLIT_SEED, configs=configs,
        results_jsonl=PHASE6_STAGE_D_RESULTS_JSONL, kan_runs_dir=PHASE6_STAGE_D_KAN_RUNS_DIR,
        merged_dir=PHASE6_STAGE_D_MERGED_DIR, ctx_raw_base=PHASE6_STAGE_D_RAW_DIR,
        ctx_vae_data_base=PHASE6_STAGE_D_VAE_DATA_DIR, ctx_vae_model_base=PHASE6_STAGE_D_VAE_MODEL_DIR,
        phase_name="Stage D (kfold)", dry_run=dry_run,
    )
    if not dry_run:
        summarize_fold_stage(
            results_jsonl=PHASE6_STAGE_D_RESULTS_JSONL, per_fold_json=PHASE6_STAGE_D_PER_FOLD_JSON,
            top_json=PHASE6_STAGE_D_TOP_JSON, phase_name="Stage D (kfold)",
        )


def run_stage_e(dry_run: bool) -> None:
    if not dry_run:
        require_phase_done(PHASE5_TOP_JSON, "Phase 5")
    configs = summarize_stage_c() if not PHASE6_STAGE_C_TOP_JSON.exists() else json.load(open(PHASE6_STAGE_C_TOP_JSON))["top"]
    run_fold_stage(
        corpus_mode="source_disjoint", n_folds=PHASE5_N_FOLDS, split_seed=PHASE5_SPLIT_SEED, configs=configs,
        results_jsonl=PHASE6_STAGE_E_RESULTS_JSONL, kan_runs_dir=PHASE6_STAGE_E_KAN_RUNS_DIR,
        merged_dir=PHASE6_STAGE_E_MERGED_DIR, ctx_raw_base=PHASE6_STAGE_E_RAW_DIR,
        ctx_vae_data_base=PHASE6_STAGE_E_VAE_DATA_DIR, ctx_vae_model_base=PHASE6_STAGE_E_VAE_MODEL_DIR,
        phase_name="Stage E (source-disjoint)", dry_run=dry_run,
    )
    if not dry_run:
        summarize_fold_stage(
            results_jsonl=PHASE6_STAGE_E_RESULTS_JSONL, per_fold_json=PHASE6_STAGE_E_PER_FOLD_JSON,
            top_json=PHASE6_STAGE_E_TOP_JSON, phase_name="Stage E (source-disjoint)",
        )


def main():
    parser = argparse.ArgumentParser(description="Phase 6: identity-free context control (Source/Domain leakage)")
    parser.add_argument("--run", action="store_true", help="Run the selected stage(s) (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate existing results only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stage", choices=["ab", "c", "d", "e", "all"], default="ab",
        help="ab = Stage A+B (dim sweep + 15 combos, no other-phase dependency). "
             "c = Stage C (VAE-reg + KAN hyperparams on Stage B's top 5, requires Stage A+B done). "
             "d = Stage D (kfold validation, requires Stage C done AND Phase 4 fully finished). "
             "e = Stage E (source-disjoint validation, requires Stage C done AND Phase 5 fully finished). "
             "all = run ab, then c (d/e are never included in 'all' -- they need Phase 4/5 done first, "
             "run them explicitly once that's confirmed).",
    )
    args = parser.parse_args()

    if not any([args.run, args.summary]):
        parser.error("Pass at least one of --run, --summary")

    if args.run:
        if args.stage in ("ab", "all"):
            run_stage_a(dry_run=args.dry_run)
            if args.dry_run:
                run_stage_b(PHASE6_CONTEXT_DIM_CANDIDATES[0], dry_run=True)
            else:
                best_dim = pick_best_context_dim()
                run_stage_b(best_dim, dry_run=False)
                summarize()

        if args.stage in ("c", "all"):
            run_stage_c(dry_run=args.dry_run)
            if not args.dry_run:
                summarize_stage_c()

        if args.stage == "d":
            run_stage_d(dry_run=args.dry_run)

        if args.stage == "e":
            run_stage_e(dry_run=args.dry_run)

    if args.summary and not args.run:
        if args.stage in ("ab", "all"):
            summarize()
        if args.stage in ("c", "all"):
            summarize_stage_c()
        if args.stage == "d":
            summarize_fold_stage(
                results_jsonl=PHASE6_STAGE_D_RESULTS_JSONL, per_fold_json=PHASE6_STAGE_D_PER_FOLD_JSON,
                top_json=PHASE6_STAGE_D_TOP_JSON, phase_name="Stage D (kfold)",
            )
        if args.stage == "e":
            summarize_fold_stage(
                results_jsonl=PHASE6_STAGE_E_RESULTS_JSONL, per_fold_json=PHASE6_STAGE_E_PER_FOLD_JSON,
                top_json=PHASE6_STAGE_E_TOP_JSON, phase_name="Stage E (source-disjoint)",
            )


if __name__ == "__main__":
    main()
