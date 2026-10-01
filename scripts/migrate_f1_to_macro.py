# scripts/migrate_f1_to_macro.py
# -*- coding: utf-8 -*-
"""
One-off migration: switches the historical "f1" field from Fake-class F1 to
macro-F1 (mean of the Fake and Real per-class F1), matching the change made to
src/evaluation/metrics.py on 2026-09-29. The old value is kept as "f1_fake".

Split-level metrics are recomputed exactly from the stored confusion matrix
(tn/fp/fn/tp) -- no retraining. KAN training early-stops on val loss, so no
model or prediction changes; only the reported/ranked F1 does.

Per-Domain breakdowns (results/*.json "domain_breakdown") only store n/accuracy/f1,
so they are recomputed from the run's predictions.pkl + test corpus Topic column,
but only when those predictions verifiably belong to that run (confusion matrix
and ROC-AUC match the record, per-domain n match). Otherwise the domain entry is
left unconverted and reported, since a kan_output_dir may have been reused.

Scope (the active results only; results_old*/ are frozen archives, untouched):
  - results/*.json                      metrics.{train,val,test} + domain_breakdown
  - results/*.jsonl                     each record's metrics / test_metrics
  - data/07_kan_runs/**/{split}_metrics.{json,csv}, all_metrics.pkl

Idempotent: a metrics dict that already has "f1_fake" is skipped.

Usage:
    python scripts/migrate_f1_to_macro.py            # dry run, prints a summary
    python scripts/migrate_f1_to_macro.py --apply    # writes the changes
Afterwards, regenerate the *_top.json summaries with each orchestrator's --summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
KAN_RUNS_DIR = BASE_DIR / "data" / "07_kan_runs"
SPLITS = ("train", "val", "test")

stats = Counter()


def macro_f1_from_cm(tn: int, fp: int, fn: int, tp: int) -> float:
    f1_fake = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f1_real = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
    return (f1_fake + f1_real) / 2


def convert_metrics(m) -> bool:
    """In place: f1 -> macro, old f1 -> f1_fake. Returns True if changed."""
    if not isinstance(m, dict) or "f1" not in m or "f1_fake" in m:
        return False
    if not all(k in m for k in ("tn", "fp", "fn", "tp")):
        stats["metrics_without_confusion_matrix"] += 1
        return False
    tn, fp, fn, tp = (int(m[k]) for k in ("tn", "fp", "fn", "tp"))
    f1_fake = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    if m["f1"] is not None and abs(f1_fake - m["f1"]) > 1e-9:
        # the stored f1 should be exactly the Fake-class F1 of its own matrix
        stats["f1_mismatch_with_cm"] += 1
        return False
    m["f1_fake"] = m["f1"]
    m["f1"] = macro_f1_from_cm(tn, fp, fn, tp)
    return True


# ---- per-Domain breakdown ----

_run_commands: dict[str, list] = {}


def load_run_commands() -> None:
    for path in RESULTS_DIR.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("run_id") and rec.get("command"):
                _run_commands[rec["run_id"]] = rec["command"]


def _arg(cmd: list, flag: str):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_corpus_pkl_for(run_id: str) -> Path:
    """Mirror of main.py's default test corpus PKL (PROCESSED_BY_MODEL_DIR/test.pkl)."""
    cmd = _run_commands.get(run_id, [])
    explicit = _arg(cmd, "--kan_test_corpus_pkl")
    if explicit:
        return Path(explicit)
    if _arg(cmd, "--corpus_mode") == "source_disjoint":
        seed = _arg(cmd, "--source_split_seed")
        n = _arg(cmd, "--source_split_n")
        idx = _arg(cmd, "--source_split_index")
        return BASE_DIR / "data" / "02_corpus_clean_source_cv" / f"seed{seed}_n{n}" / f"fold{idx}" / "test.pkl"
    if _arg(cmd, "--corpus_mode") == "kfold":
        seed = _arg(cmd, "--kfold_split_seed")
        n = _arg(cmd, "--kfold_n")
        idx = _arg(cmd, "--kfold_index")
        return BASE_DIR / "data" / "02_corpus_clean_cv" / f"seed{seed}_n{n}" / f"fold{idx}" / "test.pkl"
    return BASE_DIR / "data" / "02_corpus_clean" / "test.pkl"


_topic_cache: dict[Path, list] = {}


def convert_domain_breakdown(data: dict) -> bool:
    db = data.get("domain_breakdown")
    if not isinstance(db, dict) or not db:
        return False
    if all("f1_fake" in v for v in db.values()):
        return False

    test_m = (data.get("metrics") or {}).get("test") or {}
    out_dir = (data.get("paths") or {}).get("kan_output_dir")
    preds_path = BASE_DIR / out_dir / "predictions.pkl" if out_dir else None
    corpus_pkl = test_corpus_pkl_for(data.get("run_id", ""))
    if preds_path is None or not preds_path.exists() or not corpus_pkl.exists():
        stats["domain_skipped_missing_files"] += 1
        return False

    preds = pd.read_pickle(preds_path)["test"]
    y_true = np.asarray(preds["y_true"]).astype(int)
    y_prob = np.asarray(preds["y_prob"]).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    # predictions must belong to this run (kan_output_dir can be reused by a later run)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    same_cm = (tn, fp, fn, tp) == tuple(int(test_m.get(k, -1)) for k in ("tn", "fp", "fn", "tp"))
    same_auc = abs(roc_auc_score(y_true, y_prob) - test_m.get("roc_auc", -1)) < 1e-9
    if not (same_cm and same_auc):
        stats["domain_skipped_predictions_not_this_run"] += 1
        return False

    if corpus_pkl not in _topic_cache:
        _topic_cache[corpus_pkl] = pd.read_pickle(corpus_pkl)["Topic"].tolist()
    domains = np.asarray(_topic_cache[corpus_pkl])
    if len(domains) != len(y_true):
        stats["domain_skipped_length_mismatch"] += 1
        return False

    new_db = {}
    for domain, old in db.items():
        mask = domains == domain
        yt, yp = y_true[mask], y_pred[mask]
        f1_fake = float(f1_score(yt, yp, zero_division=0))
        if int(mask.sum()) != old.get("n") or abs(f1_fake - old.get("f1", -1)) > 1e-9:
            stats["domain_skipped_breakdown_mismatch"] += 1
            return False
        new_db[domain] = {**old, "f1": float(f1_score(yt, yp, average="macro", zero_division=0)), "f1_fake": f1_fake}

    data["domain_breakdown"] = new_db
    return True


# ---- file walkers ----

def migrate_results_json(apply: bool) -> None:
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in data or "run_id" not in data:
            continue  # *_top.json summaries: regenerated via --summary instead
        stats["results_json_seen"] += 1
        changed = False
        for split in SPLITS:
            changed |= convert_metrics((data["metrics"] or {}).get(split))
        dom_changed = convert_domain_breakdown(data)
        stats["domain_breakdown_converted"] += int(dom_changed)
        if changed or dom_changed:
            stats["results_json_changed"] += 1
            if apply:
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def migrate_jsonl(apply: bool) -> None:
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        lines_out, changed_file = [], False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                lines_out.append(line)
                continue
            rec = json.loads(line)
            changed = convert_metrics(rec.get("metrics")) | convert_metrics(rec.get("test_metrics"))
            stats["jsonl_records_changed"] += int(changed)
            changed_file |= changed
            lines_out.append(json.dumps(rec, ensure_ascii=False) if changed else line)
        if changed_file and apply:
            path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def migrate_kan_runs(apply: bool) -> None:
    for path in sorted(KAN_RUNS_DIR.rglob("*_metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if convert_metrics(m):
            stats["kan_metrics_json_changed"] += 1
            if apply:
                path.write_text(json.dumps(m, indent=4), encoding="utf-8")
                pd.DataFrame([m]).to_csv(path.with_suffix(".csv"), index=False)
    for path in sorted(KAN_RUNS_DIR.rglob("all_metrics.pkl")):
        d = pd.read_pickle(path)
        changed = False
        if isinstance(d, dict):
            for split in SPLITS:
                changed |= convert_metrics(d.get(split))
        if changed:
            stats["kan_all_metrics_pkl_changed"] += 1
            if apply:
                pd.to_pickle(d, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    args = parser.parse_args()

    load_run_commands()
    migrate_results_json(args.apply)
    migrate_jsonl(args.apply)
    migrate_kan_runs(args.apply)

    print(("APPLIED" if args.apply else "DRY RUN (use --apply to write)") + ":")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
