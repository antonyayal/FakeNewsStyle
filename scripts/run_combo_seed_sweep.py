# scripts/run_combo_seed_sweep.py
# =====================================================
# 20-row test plan on the current default split (data/raw/*.xlsx, now the
# source-mixed "homologated" split promoted after run_homologated_experiment.py
# showed it closes the train->test generalization gap).
#
# Phase 1 (15 rows): the full 2^4-1 non-empty extractor-combo sweep (isolated,
# pairs, triples, all four) -- the original 13-run batch only covered 9 of
# these 15 combos (it skipped 3 of the 6 possible pairs and 3 of the 4
# possible triples), and none of them were run against this split. Fixed
# epochs/patience/latent_dims/KAN hyperparams (the winning config from
# results/batches/20260811_235144.json), seed=42.
#
# Phase 2 (5 rows): the best Phase 1 combo re-run with 5 different seeds, to
# quantify run-to-run variance -- scripts/run_regularization_sweep.py's
# reg_baseline row showed ~2pts of accuracy swing from seed alone at
# identical hyperparameters, which the original 13-run batch couldn't
# characterize since KAN training wasn't seeded yet at that point.
#
# Uses main.py's default directories throughout (no --exclude_* skips
# extraction/VAE steps -- only --merge_vae_latents --train_kan per row, reusing
# the already-cached default-dim VAE latents for every modality).
#
# Usage:
#   python scripts/run_combo_seed_sweep.py             # run the full batch
#   python scripts/run_combo_seed_sweep.py --dry-run     # print commands only
# =====================================================

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
BATCHES_DIR = RESULTS_DIR / "batches"
LOGS_DIR = BASE_DIR / "logs" / "experiments"

BEST_COMBO_METRIC = "accuracy"
BEST_COMBO_TIEBREAK = "f1"

ALL_MODALITIES = ["semantic", "emotion", "style", "context"]
DEFAULT_LATENT = {"semantic": 128, "emotion": 16, "style": 16, "context": 64}

FIXED_KAN_ARGS = [
    "--kan_epochs", "50",
    "--kan_patience", "5",
    "--kan_batch_size", "32",
    "--kan_hidden_dim", "32",
    "--kan_num_basis", "8",
    "--kan_dropout", "0.5",
    "--kan_weight_decay", "1e-3",
]

PHASE2_SEEDS = [7, 42, 123, 777, 2024]

# =====================================================
# Plan: Phase 1 = all 15 non-empty extractor combos; Phase 2 resolved after.
# =====================================================
def all_nonempty_combos(modalities: List[str]) -> List[List[str]]:
    combos = []
    for r in range(1, len(modalities) + 1):
        combos.extend(list(c) for c in itertools.combinations(modalities, r))
    return combos


def build_plan() -> List[Dict[str, Any]]:
    plan = []
    row_id = 1
    for combo in all_nonempty_combos(ALL_MODALITIES):
        label = "_".join(combo)
        plan.append({"id": row_id, "label": label, "phase": f"1-combo{len(combo)}", "extractors": combo, "seed": 42})
        row_id += 1
    for seed in PHASE2_SEEDS:
        plan.append({"id": row_id, "label": f"best_combo_seed{seed}", "phase": "2-seed", "extractors": None, "seed": seed})
        row_id += 1
    return plan


PLAN = build_plan()
TOTAL = len(PLAN)


# =====================================================
# Helpers (mirrors scripts/run_experiments.py)
# =====================================================
def build_command(row: Dict[str, Any]) -> List[str]:
    extractors = row["extractors"]
    cmd = [sys.executable, "main.py", "--merge_vae_latents", "--train_kan"]

    for modality in ALL_MODALITIES:
        if modality not in extractors:
            cmd.append(f"--exclude_{modality}")

    cmd += [
        "--semantic_latent_dim", str(DEFAULT_LATENT["semantic"]),
        "--emotion_latent_dim", str(DEFAULT_LATENT["emotion"]),
        "--style_latent_dim", str(DEFAULT_LATENT["style"]),
        "--context_latent_dim", str(DEFAULT_LATENT["context"]),
        "--kan_seed", str(row["seed"]),
        "--kan_output_dir", f"data/07_kan_runs/combosweep_{row['label']}",
    ]
    cmd += FIXED_KAN_ARGS
    return cmd


def run_row(row: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    cmd = build_command(row)
    label = row["label"]

    print(f"\n[{row['id']:02d}/{TOTAL}] {label} ({row['phase']}) — extractors={row['extractors']} seed={row['seed']}")
    print(f"  $ {' '.join(cmd)}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{batch_id}_{row['id']:02d}_{label}.log"

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=BASE_DIR, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    entry: Dict[str, Any] = {
        **row,
        "command": cmd,
        "log_path": str(log_path),
        "elapsed_seconds": round(elapsed, 1),
        "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "run_id": None,
        "results_json": None,
    }

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Experiment record saved:\s*(\S+\.json)", log_text)
    if m:
        entry["results_json"] = m.group(1)
        entry["run_id"] = Path(m.group(1)).stem

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) — see {log_path}")
    else:
        print(f"  OK in {elapsed:.1f}s — {entry['results_json']}")

    return entry


def pick_best_combo(phase1_entries: List[Dict[str, Any]]) -> List[str]:
    scored = []
    for e in phase1_entries:
        if e["status"] != "ok" or not e["results_json"]:
            continue
        with open(e["results_json"], "r", encoding="utf-8") as f:
            record = json.load(f)
        metrics = record["metrics"]["test"]
        scored.append((metrics[BEST_COMBO_METRIC], metrics[BEST_COMBO_TIEBREAK], e))

    if not scored:
        raise RuntimeError("No successful Phase 1 runs — cannot pick a best combination for Phase 2.")

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_entry = scored[0][2]

    print(
        f"\nBest Phase 1 combo: {best_entry['label']} "
        f"({BEST_COMBO_METRIC}={scored[0][0]:.4f}, {BEST_COMBO_TIEBREAK}={scored[0][1]:.4f})"
    )
    return best_entry["extractors"]


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Run the 20-row combo x seed sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Batch ID: {batch_id}")
    print(f"Total rows: {TOTAL} (15 combos + 5 seeds on the winning combo)")

    if args.dry_run:
        for row in PLAN:
            extractors = row["extractors"] if row["extractors"] else "<resolved after Phase 1>"
            print(f"\n[{row['id']:02d}] {row['label']} ({row['phase']}) extractors={extractors} seed={row['seed']}")
        return

    entries: List[Dict[str, Any]] = []
    phase1_rows = [r for r in PLAN if r["extractors"] is not None]
    phase2_rows = [r for r in PLAN if r["extractors"] is None]

    for row in phase1_rows:
        entries.append(run_row(row, batch_id))

    best_combo = pick_best_combo(entries)

    for row in phase2_rows:
        row = dict(row)
        row["extractors"] = best_combo
        entries.append(run_row(row, batch_id))

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BATCHES_DIR / f"{batch_id}.json"
    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "description": "Full 15-combo extractor sweep + 5-seed stability check on the winning combo, "
                        "run against the current default (source-mixed) split.",
        "best_combo_metric": BEST_COMBO_METRIC,
        "best_combo_tiebreak": BEST_COMBO_TIEBREAK,
        "resolved_best_combo": best_combo,
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
