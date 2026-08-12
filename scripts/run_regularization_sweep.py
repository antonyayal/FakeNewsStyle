# scripts/run_regularization_sweep.py
# =====================================================
# Follow-up batch to results/batches/20260811_235144.json.
#
# That batch showed a large, consistent train->test accuracy gap
# (e.g. semantic_style: train 0.949 / val 0.820 / test 0.692) across every
# extractor combination, epoch count, and latent size tried -- pointing at
# KAN overfitting rather than insufficient capacity or extractor choice.
# Context was also shown to hurt (context_only: F1=0.0; semantic+context
# worse than semantic alone), so it's dropped here.
#
# This batch holds extractors=semantic+style, latent_dims and epochs at
# the winning Phase-1 settings, and sweeps the KAN classifier's own
# regularization (dropout, weight_decay, hidden_dim, num_basis) --
# untouched in the previous batch, where it was fixed at
# dropout=0.5 / weight_decay=1e-3 / hidden_dim=32 / num_basis=8 throughout.
#
# Same runner pattern/manifest format as run_experiments.py, so
# scripts/comparison_report.py and scripts/html_report_builder.py can
# consume this batch without changes.
#
# Usage:
#   python scripts/run_regularization_sweep.py
#   python scripts/run_regularization_sweep.py --dry-run
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
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
BATCHES_DIR = RESULTS_DIR / "batches"
LOGS_DIR = BASE_DIR / "logs" / "experiments"

ALL_MODALITIES = ["semantic", "emotion", "style", "context"]
EXTRACTORS = ["semantic", "style"]  # fixed: best Phase-1 combo from the previous batch

# Winning Phase-1 settings from results/batches/20260811_235144.json, held fixed.
LATENT = {"semantic": 128, "emotion": 16, "style": 16, "context": 64}
EPOCHS = 50
PATIENCE = 5

# Baseline KAN hyperparams (== the previous batch's fixed values, used as row 1
# for an apples-to-apples reference point inside this batch).
BASE_HIDDEN_DIM = 32
BASE_NUM_BASIS = 8
BASE_DROPOUT = 0.5
BASE_WEIGHT_DECAY = 1e-3

# =====================================================
# Regularization sweep plan
# =====================================================
PLAN: List[Dict[str, Any]] = [
    {"id": 1, "label": "reg_baseline", "hidden_dim": BASE_HIDDEN_DIM, "num_basis": BASE_NUM_BASIS, "dropout": BASE_DROPOUT, "weight_decay": BASE_WEIGHT_DECAY},
    {"id": 2, "label": "dropout_07", "hidden_dim": BASE_HIDDEN_DIM, "num_basis": BASE_NUM_BASIS, "dropout": 0.7, "weight_decay": BASE_WEIGHT_DECAY},
    {"id": 3, "label": "dropout_08", "hidden_dim": BASE_HIDDEN_DIM, "num_basis": BASE_NUM_BASIS, "dropout": 0.8, "weight_decay": BASE_WEIGHT_DECAY},
    {"id": 4, "label": "weight_decay_5e3", "hidden_dim": BASE_HIDDEN_DIM, "num_basis": BASE_NUM_BASIS, "dropout": BASE_DROPOUT, "weight_decay": 5e-3},
    {"id": 5, "label": "weight_decay_1e2", "hidden_dim": BASE_HIDDEN_DIM, "num_basis": BASE_NUM_BASIS, "dropout": BASE_DROPOUT, "weight_decay": 1e-2},
    {"id": 6, "label": "dropout07_wd5e3", "hidden_dim": BASE_HIDDEN_DIM, "num_basis": BASE_NUM_BASIS, "dropout": 0.7, "weight_decay": 5e-3},
    {"id": 7, "label": "hidden_dim_16", "hidden_dim": 16, "num_basis": BASE_NUM_BASIS, "dropout": BASE_DROPOUT, "weight_decay": BASE_WEIGHT_DECAY},
    {"id": 8, "label": "hidden_dim_16_reg", "hidden_dim": 16, "num_basis": BASE_NUM_BASIS, "dropout": 0.6, "weight_decay": 5e-3},
    {"id": 9, "label": "num_basis_4", "hidden_dim": BASE_HIDDEN_DIM, "num_basis": 4, "dropout": BASE_DROPOUT, "weight_decay": BASE_WEIGHT_DECAY},
    {"id": 10, "label": "combined_reg", "hidden_dim": 16, "num_basis": 4, "dropout": 0.6, "weight_decay": 5e-3},
]
TOTAL = len(PLAN)


# =====================================================
# Helpers (mirrors run_experiments.py)
# =====================================================
def build_command(row: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, "main.py", "--merge_vae_latents", "--train_kan"]

    for modality in ALL_MODALITIES:
        if modality not in EXTRACTORS:
            cmd.append(f"--exclude_{modality}")

    cmd += [
        "--semantic_latent_dim", str(LATENT["semantic"]),
        "--emotion_latent_dim", str(LATENT["emotion"]),
        "--style_latent_dim", str(LATENT["style"]),
        "--context_latent_dim", str(LATENT["context"]),
        "--kan_epochs", str(EPOCHS),
        "--kan_patience", str(PATIENCE),
        "--kan_output_dir", f"data/07_kan_runs/regsweep_{row['label']}",
        "--kan_batch_size", "32",
        "--kan_hidden_dim", str(row["hidden_dim"]),
        "--kan_num_basis", str(row["num_basis"]),
        "--kan_dropout", str(row["dropout"]),
        "--kan_weight_decay", str(row["weight_decay"]),
    ]
    return cmd


def run_row(row: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    cmd = build_command(row)
    label = row["label"]

    print(f"\n[{row['id']:02d}/{TOTAL}] {label} — hidden_dim={row['hidden_dim']} num_basis={row['num_basis']} "
          f"dropout={row['dropout']} weight_decay={row['weight_decay']}")
    print(f"  $ {' '.join(cmd)}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{batch_id}_{row['id']:02d}_{label}.log"

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=BASE_DIR, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    entry: Dict[str, Any] = {
        **row,
        "phase": "3-regularization",
        "extractors": EXTRACTORS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "latent": LATENT,
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


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Run the KAN regularization sweep batch (semantic+style fixed)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Batch ID: {batch_id}")
    print(f"Total rows: {TOTAL}")
    print(f"Fixed: extractors={EXTRACTORS}, latent={LATENT}, epochs={EPOCHS}, patience={PATIENCE}")

    if args.dry_run:
        for row in PLAN:
            print(f"\n[{row['id']:02d}] {row['label']}")
            print(f"  {' '.join(build_command(row))}")
        return

    entries = [run_row(row, batch_id) for row in PLAN]

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BATCHES_DIR / f"{batch_id}.json"

    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "description": "KAN regularization sweep (dropout/weight_decay/hidden_dim/num_basis) on the fixed semantic+style combo, "
                        "following up on the 26pt train->test gap found in batch 20260811_235144.",
        "best_combo_metric": None,
        "best_combo_tiebreak": None,
        "resolved_best_combo": EXTRACTORS,
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
