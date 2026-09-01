# scripts/orchestrator_phase4.py
# -*- coding: utf-8 -*-
"""
Phase 4: robustness to partition -- standard stratified k-fold (NOT
source-disjoint, see orchestrator_phase5.py for that) over the pooled
corpus. Takes the top 5 (config, variant) entries from results/phase3_top.json
and repeats each across 5 folds x the 5 fixed SEEDS, purely as a
generalization check -- explores nothing new. See scripts/fold_validation.py
for the shared mechanics (also used by orchestrator_phase5.py).

Usage:
    python scripts/orchestrator_phase4.py --run
    python scripts/orchestrator_phase4.py --run --dry-run
    python scripts/orchestrator_phase4.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import (  # noqa: E402
    PHASE3_TOP_JSON,
    PHASE4_KAN_RUNS_DIR,
    PHASE4_MERGED_CV_DIR,
    PHASE4_N_FOLDS,
    PHASE4_PER_FOLD_JSON,
    PHASE4_RESULTS_JSONL,
    PHASE4_SPLIT_SEED,
    PHASE4_TOP_JSON,
)
from fold_validation import run_sweep, summarize  # noqa: E402

PHASE_NAME = "phase4"
CORPUS_MODE = "kfold"


def load_phase3_top() -> List[Dict[str, Any]]:
    if not PHASE3_TOP_JSON.exists():
        raise FileNotFoundError(f"{PHASE3_TOP_JSON} not found. Run scripts/orchestrator_phase3.py --run first.")
    with open(PHASE3_TOP_JSON, "r", encoding="utf-8") as f:
        return json.load(f)["top"]


def main():
    parser = argparse.ArgumentParser(description="Phase 4: validate Phase 3's top 5 across normal k-folds")
    parser.add_argument("--run", action="store_true", help="Run the full sweep (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate results/orchestrator_phase4.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not any([args.run, args.summary]):
        parser.error("Pass at least one of --run, --summary")

    configs = load_phase3_top()

    if args.run:
        run_sweep(
            corpus_mode=CORPUS_MODE, n_folds=PHASE4_N_FOLDS, split_seed=PHASE4_SPLIT_SEED,
            configs=configs, results_jsonl=PHASE4_RESULTS_JSONL, kan_runs_dir=PHASE4_KAN_RUNS_DIR,
            merged_cv_dir=PHASE4_MERGED_CV_DIR, phase_name=PHASE_NAME, dry_run=args.dry_run,
        )
        if not args.dry_run:
            summarize(
                results_jsonl=PHASE4_RESULTS_JSONL, per_fold_json=PHASE4_PER_FOLD_JSON,
                top_json=PHASE4_TOP_JSON, phase_name=PHASE_NAME,
            )

    if args.summary and not args.run:
        summarize(
            results_jsonl=PHASE4_RESULTS_JSONL, per_fold_json=PHASE4_PER_FOLD_JSON,
            top_json=PHASE4_TOP_JSON, phase_name=PHASE_NAME,
        )


if __name__ == "__main__":
    main()
