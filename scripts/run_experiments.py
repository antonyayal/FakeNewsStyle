# scripts/run_experiments.py
# =====================================================
# Standalone experiment queue runner for FakeNewsStyle.
#
# Runs a fixed batch of main.py invocations sequentially (no Claude Code
# involved) and records which results/{run_id}.json each row produced, so
# scripts/comparison_report.py can build a report scoped to this batch.
#
# Phase 1 (rows with a concrete extractor list) always runs first.
# Phase 2 rows (extractors=None) are resolved automatically once Phase 1
# finishes, using the best-performing Phase 1 combination by
# BEST_COMBO_METRIC (primary) / BEST_COMBO_TIEBREAK (tiebreak).
#
# Usage:
#   python scripts/run_experiments.py             # run the full batch
#   python scripts/run_experiments.py --dry-run     # print commands only
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

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
BATCHES_DIR = RESULTS_DIR / "batches"
LOGS_DIR = BASE_DIR / "logs" / "experiments"

BEST_COMBO_METRIC = "accuracy"
BEST_COMBO_TIEBREAK = "f1"

# Fixed KAN hyperparameters, held constant across the whole batch —
# only extractors / epochs+patience / latent_dims vary per row.
FIXED_KAN_ARGS = [
    "--kan_batch_size", "32",
    "--kan_hidden_dim", "32",
    "--kan_num_basis", "8",
    "--kan_dropout", "0.5",
    "--kan_weight_decay", "1e-3",
]

ALL_MODALITIES = ["semantic", "emotion", "style", "context"]

DEFAULT_LATENT = {"semantic": 128, "emotion": 16, "style": 16, "context": 64}
SMALL_LATENT = {"semantic": 64, "emotion": 8, "style": 8, "context": 32}
LARGE_LATENT = {"semantic": 256, "emotion": 32, "style": 32, "context": 128}

# =====================================================
# The approved 13-row test plan
# =====================================================
PLAN: List[Dict[str, Any]] = [
    # ---- Phase 1: extractor combinations (fixed epochs/patience/latent_dim) ----
    {"id": 1, "label": "semantic_only", "phase": "1-isolated", "extractors": ["semantic"], "epochs": 50, "patience": 5, "latent": DEFAULT_LATENT},
    {"id": 2, "label": "emotion_only", "phase": "1-isolated", "extractors": ["emotion"], "epochs": 50, "patience": 5, "latent": DEFAULT_LATENT},
    {"id": 3, "label": "style_only", "phase": "1-isolated", "extractors": ["style"], "epochs": 50, "patience": 5, "latent": DEFAULT_LATENT},
    {"id": 4, "label": "context_only", "phase": "1-isolated", "extractors": ["context"], "epochs": 50, "patience": 5, "latent": DEFAULT_LATENT},
    {"id": 5, "label": "semantic_style", "phase": "1-partial2", "extractors": ["semantic", "style"], "epochs": 50, "patience": 5, "latent": DEFAULT_LATENT},
    {"id": 6, "label": "semantic_emotion", "phase": "1-partial2", "extractors": ["semantic", "emotion"], "epochs": 50, "patience": 5, "latent": DEFAULT_LATENT},
    {"id": 7, "label": "semantic_context", "phase": "1-partial2", "extractors": ["semantic", "context"], "epochs": 50, "patience": 5, "latent": DEFAULT_LATENT},
    {"id": 8, "label": "semantic_emotion_style", "phase": "1-partial3", "extractors": ["semantic", "emotion", "style"], "epochs": 50, "patience": 5, "latent": DEFAULT_LATENT},
    {"id": 9, "label": "all_extractors", "phase": "1-full", "extractors": ["semantic", "emotion", "style", "context"], "epochs": 50, "patience": 5, "latent": DEFAULT_LATENT},

    # ---- Phase 2a: epoch/patience variation (extractors resolved after Phase 1) ----
    {"id": 10, "label": "epochs_short", "phase": "2a-epochs", "extractors": None, "epochs": 15, "patience": 3, "latent": DEFAULT_LATENT},
    {"id": 11, "label": "epochs_long", "phase": "2a-epochs", "extractors": None, "epochs": 100, "patience": 20, "latent": DEFAULT_LATENT},

    # ---- Phase 2b: latent-dim variation (extractors resolved after Phase 1) ----
    {"id": 12, "label": "latent_small", "phase": "2b-latent", "extractors": None, "epochs": 50, "patience": 5, "latent": SMALL_LATENT},
    {"id": 13, "label": "latent_large", "phase": "2b-latent", "extractors": None, "epochs": 50, "patience": 5, "latent": LARGE_LATENT},
]


# =====================================================
# Helpers
# =====================================================
def build_command(row: Dict[str, Any]) -> List[str]:
    extractors = row["extractors"]
    latent = row["latent"]

    cmd = [sys.executable, "main.py", "--merge_vae_latents", "--train_kan"]

    for modality in ALL_MODALITIES:
        if modality not in extractors:
            cmd.append(f"--exclude_{modality}")

    cmd += [
        "--semantic_latent_dim", str(latent["semantic"]),
        "--emotion_latent_dim", str(latent["emotion"]),
        "--style_latent_dim", str(latent["style"]),
        "--context_latent_dim", str(latent["context"]),
        "--kan_epochs", str(row["epochs"]),
        "--kan_patience", str(row["patience"]),
        "--kan_output_dir", f"data/07_kan_runs/sweep_{row['label']}",
    ]
    cmd += FIXED_KAN_ARGS

    return cmd


def run_row(row: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    cmd = build_command(row)
    label = row["label"]

    print(f"\n[{row['id']:02d}/13] {label} ({row['phase']}) — extractors={row['extractors']}")
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
    parser = argparse.ArgumentParser(description="Run the FakeNewsStyle experiment batch")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Batch ID: {batch_id}")
    print(f"Total rows: {len(PLAN)}")

    if args.dry_run:
        for row in PLAN:
            print(f"\n[{row['id']:02d}] {row['label']} ({row['phase']})")
            print(f"  extractors={row['extractors']}")
            print(f"  {' '.join(build_command(row) if row['extractors'] else ['<resolved after Phase 1>'])}")
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
    print(f"Next: python scripts/comparison_report.py --batch {manifest_path}")


if __name__ == "__main__":
    main()
