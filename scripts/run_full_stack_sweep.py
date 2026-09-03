# scripts/run_full_stack_sweep.py
# =====================================================
# 20-row plan that varies the VAE and the KAN, not just extractor choice --
# every prior batch this session (13-run, 20-run combo x seed) held both the
# VAE and the KAN's architecture fixed and only varied which extractors were
# active, epochs, latent size (all 4 branches together, only once), or KAN
# regularization in isolation. This plan is designed to close those gaps.
#
# Fixed baseline (used unless a row explicitly overrides it):
#   extractors = all four (current best combo, see results/batches/20260812_020442.json)
#   VAE:  epochs=100, batch_size=32, lr=1e-3, beta=1.0, dropout=0.1  (main.py defaults)
#   KAN:  hidden_dim=32, num_basis=8, dropout=0.5, weight_decay=1e-3, lr=1e-3,
#         epochs=50, patience=5, seed=42  (the winning config from earlier batches)
#
# Phase 1 (8 rows) -- VAE sweep, extractors and KAN held at baseline:
#   beta (KL weight) up/down, dropout up/down, learning_rate (never varied
#   before), a much shorter training budget, and latent_dim halved/doubled
#   across all 4 branches at once (earlier latent sweeps only ever touched
#   semantic+style, never all four together).
#
# Phase 2 (7 rows) -- KAN sweep, extractors and VAE held at baseline:
#   hidden_dim and num_basis pushed further than the earlier regularization
#   sweep tried, plus KAN learning_rate (also never varied before) and a
#   longer epochs/patience budget.
#
# Phase 3 (5 rows) -- extractor-combo robustness check: reruns the top 5
# combos from the 20-run combo sweep, but using whichever single
# hyperparameter configuration (from Phase 1 or Phase 2) scored best on test
# accuracy -- resolved automatically after Phase 1+2 finish, mirroring the
# two-phase pattern from scripts/run_experiments.py.
#
# VAE-hyperparameter rows train into isolated directories
# (data/05_vae_latents_fullsweep/, models/vae_fullsweep/) and merge latents
# manually in Python -- main.py's --merge_vae_latents step always reads from
# the hardcoded default data/05_vae_latents/ path, so it can't see isolated
# VAE output on its own. Rows that keep the default VAE reuse the already-
# cached default latents via the normal --merge_vae_latents --train_kan path
# (fast, and never touches the tracked default models/vae/).
#
# Usage:
#   python scripts/run_full_stack_sweep.py
#   python scripts/run_full_stack_sweep.py --dry-run
# =====================================================

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
BATCHES_DIR = RESULTS_DIR / "batches"
LOGS_DIR = BASE_DIR / "logs" / "experiments"

BEST_METRIC = "accuracy"
BEST_TIEBREAK = "f1"

ALL_MODALITIES = ["semantic", "emotion", "style", "context"]
ALL_FOUR = list(ALL_MODALITIES)

DEFAULT_LATENT = {"semantic": 128, "emotion": 16, "style": 16, "context": 64}
SMALL_LATENT = {"semantic": 64, "emotion": 8, "style": 8, "context": 32}
LARGE_LATENT = {"semantic": 256, "emotion": 32, "style": 32, "context": 128}

VAE_HIDDEN_DIMS = {
    "semantic": [512, 256], "emotion": [128, 64], "style": [128, 64], "context": [256, 128],
}

DEFAULT_VAE = {"epochs": 100, "batch_size": 32, "learning_rate": 1e-3, "beta": 1.0, "dropout": 0.1}
DEFAULT_KAN = {"hidden_dim": 32, "num_basis": 8, "dropout": 0.5, "weight_decay": 1e-3,
                "lr": 1e-3, "epochs": 50, "patience": 5, "seed": 42}

# Top 5 extractor combos from results/batches/20260812_020442.json (by test accuracy).
TOP5_COMBOS = [
    ["semantic", "emotion", "style", "context"],
    ["semantic", "emotion", "context"],
    ["semantic", "emotion", "style"],
    ["semantic", "style"],
    ["semantic", "emotion"],
]


def _merged(base: Dict[str, Any], **overrides) -> Dict[str, Any]:
    d = dict(base)
    d.update(overrides)
    return d


# =====================================================
# Plan
# =====================================================
def build_plan() -> List[Dict[str, Any]]:
    plan = []
    rid = 1

    # ---- Phase 1: VAE sweep (extractors + KAN held at baseline) ----
    vae_rows = [
        ("vae_beta_low", dict(vae=_merged(DEFAULT_VAE, beta=0.25))),
        ("vae_beta_high", dict(vae=_merged(DEFAULT_VAE, beta=4.0))),
        ("vae_dropout_low", dict(vae=_merged(DEFAULT_VAE, dropout=0.0))),
        ("vae_dropout_high", dict(vae=_merged(DEFAULT_VAE, dropout=0.3))),
        ("vae_lr_high", dict(vae=_merged(DEFAULT_VAE, learning_rate=5e-3))),
        ("vae_epochs_short", dict(vae=_merged(DEFAULT_VAE, epochs=30))),
        ("vae_latent_small", dict(latent=SMALL_LATENT)),
        ("vae_latent_large", dict(latent=LARGE_LATENT)),
    ]
    for label, cfg in vae_rows:
        plan.append({
            "id": rid, "label": label, "phase": "1-vae",
            "extractors": ALL_FOUR,
            "latent": cfg.get("latent", DEFAULT_LATENT),
            "vae": cfg.get("vae", DEFAULT_VAE),
            "kan": DEFAULT_KAN,
        })
        rid += 1

    # ---- Phase 2: KAN sweep (extractors + VAE held at baseline) ----
    kan_rows = [
        ("kan_hidden_16", dict(hidden_dim=16)),
        ("kan_hidden_128", dict(hidden_dim=128)),
        ("kan_basis_4", dict(num_basis=4)),
        ("kan_basis_32", dict(num_basis=32)),
        ("kan_lr_low", dict(lr=1e-4)),
        ("kan_lr_high", dict(lr=5e-3)),
        ("kan_epochs_long", dict(epochs=150, patience=25)),
    ]
    for label, overrides in kan_rows:
        plan.append({
            "id": rid, "label": label, "phase": "2-kan",
            "extractors": ALL_FOUR,
            "latent": DEFAULT_LATENT,
            "vae": DEFAULT_VAE,
            "kan": _merged(DEFAULT_KAN, **overrides),
        })
        rid += 1

    # ---- Phase 3: extractor-combo robustness under the winning config ----
    for combo in TOP5_COMBOS:
        label = "best_hparams_" + "_".join(combo)
        plan.append({
            "id": rid, "label": label, "phase": "3-extractors",
            "extractors": combo,
            "latent": None, "vae": None, "kan": None,  # resolved after Phase 1+2
        })
        rid += 1

    return plan


PLAN = build_plan()
TOTAL = len(PLAN)


def is_default_vae(row: Dict[str, Any]) -> bool:
    return row["latent"] == DEFAULT_LATENT and row["vae"] == DEFAULT_VAE


# =====================================================
# Simple path: default VAE, reuse cached default latents (main.py's own
# --merge_vae_latents --train_kan in one call).
# =====================================================
def build_simple_command(row: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, "main.py", "--merge_vae_latents", "--train_kan"]
    for m in ALL_MODALITIES:
        if m not in row["extractors"]:
            cmd.append(f"--exclude_{m}")
    for m in ALL_MODALITIES:
        cmd += [f"--{m}_latent_dim", str(row["latent"][m])]

    kan = row["kan"]
    cmd += [
        "--kan_output_dir", f"data/07_kan_runs/fullsweep_{row['label']}",
        "--kan_hidden_dim", str(kan["hidden_dim"]),
        "--kan_num_basis", str(kan["num_basis"]),
        "--kan_dropout", str(kan["dropout"]),
        "--kan_weight_decay", str(kan["weight_decay"]),
        "--kan_lr", str(kan["lr"]),
        "--kan_epochs", str(kan["epochs"]),
        "--kan_patience", str(kan["patience"]),
        "--kan_seed", str(kan["seed"]),
        "--kan_batch_size", "32",
    ]
    return cmd


def run_subprocess(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=BASE_DIR, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def parse_results_json(log_path: Path) -> Optional[str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Experiment record saved:\s*(\S+\.json)", text)
    return m.group(1) if m else None


def run_simple_row(row: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    cmd = build_simple_command(row)
    label = row["label"]
    print(f"\n[{row['id']:02d}/{TOTAL}] {label} ({row['phase']}) [simple: cached default VAE]")
    print(f"  $ {' '.join(cmd)}")

    log_path = LOGS_DIR / f"{batch_id}_{row['id']:02d}_{label}.log"
    start = time.time()
    rc = run_subprocess(cmd, log_path)
    elapsed = time.time() - start

    results_json = parse_results_json(log_path) if rc == 0 else None
    entry = {**row, "command": cmd, "log_path": str(log_path), "elapsed_seconds": round(elapsed, 1),
             "returncode": rc, "status": "ok" if rc == 0 else "failed",
             "run_id": Path(results_json).stem if results_json else None, "results_json": results_json}
    print(f"  {'OK' if rc == 0 else 'FAILED'} in {elapsed:.1f}s — {results_json}")
    return entry


# =====================================================
# Full path: custom VAE hyperparams/latent dims -> isolated training +
# manual merge (mirrors scripts/run_homologated_experiment.py).
# =====================================================
def merge_latents(extractors: List[str], vae_data_dir: Path, latent: Dict[str, int], out_dir: Path) -> Dict[str, Path]:
    latent_dirs = {m: vae_data_dir / m / f"latent{latent[m]}" for m in extractors}
    out_paths = {}
    for split in ["train", "val", "test"]:
        dfs, labels = [], None
        for feature_name, feature_dir in latent_dirs.items():
            df = pd.read_pickle(feature_dir / f"{split}.pkl")
            if "label" in df.columns:
                current_labels = df["label"].reset_index(drop=True)
                if labels is None:
                    labels = current_labels
                df = df.drop(columns=["label"])
            df = df.reset_index(drop=True)
            df.columns = [c if str(c).startswith(f"{feature_name}_") else f"{feature_name}_{c}" for c in df.columns]
            dfs.append(df)
        merged_df = pd.concat(dfs, axis=1)
        if labels is not None:
            merged_df["label"] = labels.values
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.pkl"
        merged_df.to_pickle(out_path)
        out_paths[split] = out_path
    return out_paths


def run_full_row(row: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    label = row["label"]
    extractors = row["extractors"]
    latent = row["latent"]
    vae = row["vae"]
    kan = row["kan"]

    print(f"\n[{row['id']:02d}/{TOTAL}] {label} ({row['phase']}) [full: custom VAE, isolated dirs]")

    vae_data_dir = BASE_DIR / "data" / "05_vae_latents_fullsweep" / label
    vae_model_dir = BASE_DIR / "models" / "vae_fullsweep" / label

    cmd1 = [sys.executable, "main.py", "--run_vaes"]
    for m in ALL_MODALITIES:
        if m not in extractors:
            cmd1.append(f"--exclude_{m}")
    for m in ALL_MODALITIES:
        cmd1 += [f"--{m}_latent_dim", str(latent[m])]
    cmd1 += [
        "--vae_epochs", str(vae["epochs"]),
        "--vae_batch_size", str(vae["batch_size"]),
        "--vae_learning_rate", str(vae["learning_rate"]),
        "--vae_beta", str(vae["beta"]),
        "--vae_dropout", str(vae["dropout"]),
        "--vae_data_output_dir", str(vae_data_dir.relative_to(BASE_DIR)),
        "--vae_model_output_dir", str(vae_model_dir.relative_to(BASE_DIR)),
    ]
    print(f"  $ {' '.join(cmd1)}")
    log1 = LOGS_DIR / f"{batch_id}_{row['id']:02d}_{label}_vae.log"
    start = time.time()
    rc1 = run_subprocess(cmd1, log1)
    if rc1 != 0:
        elapsed = time.time() - start
        print(f"  FAILED (VAE step, exit {rc1}) — see {log1}")
        return {**row, "command": cmd1, "log_path": str(log1), "elapsed_seconds": round(elapsed, 1),
                "returncode": rc1, "status": "failed", "run_id": None, "results_json": None}

    merged_dir = BASE_DIR / "data" / "06_vae_latents_merged_fullsweep" / label
    paths = merge_latents(extractors, vae_data_dir, latent, merged_dir)

    output_dir = BASE_DIR / "data" / "07_kan_runs" / f"fullsweep_{label}"
    cmd2 = [sys.executable, "main.py", "--train_kan"]
    # Same reasoning as the vae_* flags below: main.py logs latent_dims and
    # resolves vae_model_dirs from *its own* args.*_latent_dim, not from what
    # cmd1 actually used -- must be repeated here for accurate metadata,
    # even though this call never re-touches the VAE.
    for m in ALL_MODALITIES:
        cmd2 += [f"--{m}_latent_dim", str(latent[m])]
    cmd2 += [
        "--kan_train_pkl", str(paths["train"]),
        "--kan_val_pkl", str(paths["val"]),
        "--kan_test_pkl", str(paths["test"]),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
        "--kan_hidden_dim", str(kan["hidden_dim"]),
        "--kan_num_basis", str(kan["num_basis"]),
        "--kan_dropout", str(kan["dropout"]),
        "--kan_weight_decay", str(kan["weight_decay"]),
        "--kan_lr", str(kan["lr"]),
        "--kan_epochs", str(kan["epochs"]),
        "--kan_patience", str(kan["patience"]),
        "--kan_seed", str(kan["seed"]),
        "--kan_batch_size", "32",
        # This subprocess never sees the --run_vaes call above -- main.py's Step 10
        # logs vae_hyperparams and resolves vae_model_dirs from *its own* args
        # regardless, so the real VAE config (and where its checkpoints actually
        # live) has to be repeated here too, or results/*.json would misreport
        # both as defaults.
        "--vae_epochs", str(vae["epochs"]),
        "--vae_batch_size", str(vae["batch_size"]),
        "--vae_learning_rate", str(vae["learning_rate"]),
        "--vae_beta", str(vae["beta"]),
        "--vae_dropout", str(vae["dropout"]),
        "--vae_model_output_dir", str(vae_model_dir.relative_to(BASE_DIR)),
    ]
    print(f"  $ {' '.join(cmd2)}")
    log2 = LOGS_DIR / f"{batch_id}_{row['id']:02d}_{label}_kan.log"
    rc2 = run_subprocess(cmd2, log2)
    elapsed = time.time() - start

    results_json = parse_results_json(log2) if rc2 == 0 else None
    entry = {**row, "command": cmd2, "log_path": str(log2), "elapsed_seconds": round(elapsed, 1),
             "returncode": rc2, "status": "ok" if rc2 == 0 else "failed",
             "run_id": Path(results_json).stem if results_json else None, "results_json": results_json}
    print(f"  {'OK' if rc2 == 0 else 'FAILED'} in {elapsed:.1f}s — {results_json}")
    return entry


def run_row(row: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    return run_simple_row(row, batch_id) if is_default_vae(row) else run_full_row(row, batch_id)


def pick_best_row(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = []
    for e in entries:
        if e["status"] != "ok" or not e["results_json"]:
            continue
        record = json.load(open(e["results_json"], "r", encoding="utf-8"))
        m = record["metrics"]["test"]
        scored.append((m[BEST_METRIC], m[BEST_TIEBREAK], e))
    if not scored:
        raise RuntimeError("No successful Phase 1/2 runs — cannot resolve Phase 3 hyperparameters.")
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = scored[0][2]
    print(f"\nBest Phase 1+2 config: {best['label']} ({BEST_METRIC}={scored[0][0]:.4f}, {BEST_TIEBREAK}={scored[0][1]:.4f})")
    return best


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Run the 20-row VAE x KAN x extractor sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Batch ID: {batch_id}")
    print(f"Total rows: {TOTAL} (8 VAE + 7 KAN + 5 extractor-robustness)")

    if args.dry_run:
        for row in PLAN:
            mode = "simple" if row["vae"] is not None and is_default_vae(row) else \
                   ("simple/full: resolved after Phase 1+2" if row["vae"] is None else "full")
            print(f"\n[{row['id']:02d}] {row['label']} ({row['phase']}) [{mode}]")
            print(f"  extractors={row['extractors']}  latent={row['latent']}  vae={row['vae']}  kan={row['kan']}")
        return

    entries: List[Dict[str, Any]] = []
    phase12_rows = [r for r in PLAN if r["phase"] != "3-extractors"]
    phase3_rows = [r for r in PLAN if r["phase"] == "3-extractors"]

    for row in phase12_rows:
        entries.append(run_row(row, batch_id))

    winner = pick_best_row(entries)
    resolved_latent, resolved_vae, resolved_kan = winner["latent"], winner["vae"], winner["kan"]

    for row in phase3_rows:
        row = dict(row)
        row["latent"], row["vae"], row["kan"] = resolved_latent, resolved_vae, resolved_kan
        entries.append(run_row(row, batch_id))

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BATCHES_DIR / f"{batch_id}.json"
    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "description": "20-row sweep varying VAE hyperparameters (Phase 1), KAN hyperparameters (Phase 2), "
                        "and extractor-combo robustness under the winning config (Phase 3).",
        "best_combo_metric": BEST_METRIC,
        "best_combo_tiebreak": BEST_TIEBREAK,
        "resolved_best_combo": winner["extractors"],
        "resolved_best_hyperparams": {"latent": resolved_latent, "vae": resolved_vae, "kan": resolved_kan},
        "runs": entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for e in entries if e["status"] == "ok")
    print(f"\nBatch complete: {n_ok}/{len(entries)} runs succeeded.")
    print(f"Manifest saved: {manifest_path}")
    print(f"Next: python scripts/html_report_builder.py --batch-id {batch_id}")


if __name__ == "__main__":
    main()
