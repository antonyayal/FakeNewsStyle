# scripts/orchestrator_phase5.py
# -*- coding: utf-8 -*-
"""
Phase 5: robustness to leakage by outlet (Source) -- source-disjoint folds
(StratifiedGroupKFold + GroupShuffleSplit grouped by Source, see
src/data/source_split_corpus.py), so no news outlet appears in more than one
of a fold's train/val/test. Takes the SAME top 5 (config, variant) entries
from results/phase3_top.json as orchestrator_phase4.py (independent branch,
not chained after Phase 4) and repeats each across 5 folds x the 5 fixed
SEEDS. See scripts/fold_validation.py for the shared mechanics.

This is the definitive test of the Source/Domain leakage documented in
README.md's "Known Limitations & Caveats": if test F1 collapses toward the
identity-free ablation ceiling even with every branch active, that confirms
prior high F1 numbers were largely outlet memorization, not genuine
style/semantic signal.

Usage:
    python scripts/orchestrator_phase5.py --run
    python scripts/orchestrator_phase5.py --run --dry-run
    python scripts/orchestrator_phase5.py --summary
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
    PHASE5_KAN_RUNS_DIR,
    PHASE5_MERGED_CV_DIR,
    PHASE5_N_FOLDS,
    PHASE5_PER_FOLD_JSON,
    PHASE5_RESULTS_JSONL,
    PHASE5_SPLIT_SEED,
    PHASE5_TOP_JSON,
)
from fold_validation import run_sweep, summarize  # noqa: E402

PHASE_NAME = "phase5"
CORPUS_MODE = "source_disjoint"


def load_phase3_top() -> List[Dict[str, Any]]:
    if not PHASE3_TOP_JSON.exists():
        raise FileNotFoundError(f"{PHASE3_TOP_JSON} not found. Run scripts/orchestrator_phase3.py --run first.")
    with open(PHASE3_TOP_JSON, "r", encoding="utf-8") as f:
        return json.load(f)["top"]


def main():
    parser = argparse.ArgumentParser(description="Phase 5: validate Phase 3's top 5 across source-disjoint folds")
    parser.add_argument("--run", action="store_true", help="Run the full sweep (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate results/orchestrator_phase5.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not any([args.run, args.summary]):
        parser.error("Pass at least one of --run, --summary")

    configs = load_phase3_top()

    if args.run:
        run_sweep(
            corpus_mode=CORPUS_MODE, n_folds=PHASE5_N_FOLDS, split_seed=PHASE5_SPLIT_SEED,
            configs=configs, results_jsonl=PHASE5_RESULTS_JSONL, kan_runs_dir=PHASE5_KAN_RUNS_DIR,
            merged_cv_dir=PHASE5_MERGED_CV_DIR, phase_name=PHASE_NAME, dry_run=args.dry_run,
        )
        if not args.dry_run:
            summarize(
                results_jsonl=PHASE5_RESULTS_JSONL, per_fold_json=PHASE5_PER_FOLD_JSON,
                top_json=PHASE5_TOP_JSON, phase_name=PHASE_NAME,
            )

    if args.summary and not args.run:
        summarize(
            results_jsonl=PHASE5_RESULTS_JSONL, per_fold_json=PHASE5_PER_FOLD_JSON,
            top_json=PHASE5_TOP_JSON, phase_name=PHASE_NAME,
        )


if __name__ == "__main__":
    main()
