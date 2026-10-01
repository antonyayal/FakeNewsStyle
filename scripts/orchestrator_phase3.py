# scripts/orchestrator_phase3.py
# -*- coding: utf-8 -*-
"""
Phase 3 (Option A): source-disjoint validation -- the leakage-controlled result.

Same 23 combo-variants as Phase 2 (all 15 extractor combos; the 8 with
`context` doubled for identity ON/OFF), same kan_hidden_dim x SEEDS grid,
but partitioned so no news outlet (`Source`) appears in more than one of a
fold's train/val/test (StratifiedGroupKFold + GroupShuffleSplit grouped by
Source, src/data/source_split_corpus.py).

  23 combo-variants x 2 hidden_dim x 3 seeds x 5 folds = 690 KAN runs.

The contrast between this and Phase 2 quantifies how much of the standard-
split F1 was outlet memorization rather than genuine style/semantic signal
(see "Known Limitations & Caveats" in README.md).

Per (combo-variant, fold): one preparation pass builds that fold's corpus /
features / per-branch VAE latents and concatenates them into KAN-ready PKLs
(non-context branches + identity-ON context via main.py's shared source_cv
cache; identity-OFF context via its own isolated per-fold cache); then the
hidden_dim x seed grid is pure KAN training on top.

Checkpoint/resume via results/orchestrator_phase3.jsonl + run_key. A marker
file per (fold, prep unit) lets a resumed run skip preparation it already did.

Usage:
    python scripts/orchestrator_phase3.py --run
    python scripts/orchestrator_phase3.py --run --dry-run
    python scripts/orchestrator_phase3.py --summary
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
    ALL_MODALITIES,
    BASE_DIR,
    DEFAULT_LATENT_DIM,
    EXTRACT_DEVICE,
    KAN_RUNS_DIR,
    PHASE3_MERGED_DIR,
    PHASE3_PER_FOLD_JSON,
    PHASE3_RESULTS_JSONL,
    PHASE3_TOP_JSON,
    RANKING_METRIC,
    SEEDS,
    SOURCE_CV_RAW_DIR,
    SOURCE_CV_VAE_DIR,
    SOURCE_SPLIT_N_FOLDS,
    SOURCE_SPLIT_SEED,
    source_cv_fold_dir,
)
from experiment_plan import (  # noqa: E402
    build_kan_cmd,
    build_prep_units,
    entry_label,
    ensure_idfree_context,
    iter_kan_entries,
    load_context_dims,
    merge_kan_inputs,
    total_kan_runs,
)
from experiment_runner import (  # noqa: E402
    execute_and_log,
    latent_cache_is_fresh,
    load_ok_run_keys,
    python_executable,
    run_main_command,
)

PHASE_NAME = "phase3"


def _source_disjoint_flags(fold_idx: int) -> List[str]:
    return ["--corpus_mode", "source_disjoint",
            "--source_split_n", str(SOURCE_SPLIT_N_FOLDS),
            "--source_split_index", str(fold_idx),
            "--source_split_seed", str(SOURCE_SPLIT_SEED)]


def _marker_ok(merged_dir: Path, unit: Dict[str, Any]) -> bool:
    if not all((merged_dir / f"{s}.pkl").exists() for s in ("train", "val", "test")):
        return False
    marker = merged_dir / "_marker.json"
    if not marker.exists():
        return False
    with open(marker, "r", encoding="utf-8") as f:
        saved = json.load(f)
    return saved.get("latent_dims") == unit["latent_dims"] and saved.get("context_identity") == unit["context_identity"]


def prepare_fold(unit: Dict[str, Any], fold_idx: int, dry_run: bool) -> Dict[str, Path]:
    """Build fold `fold_idx`'s KAN-ready PKLs for one combo-variant."""
    combo = unit["active_extractors"]
    latent_dims = unit["latent_dims"]
    mode = unit["context_identity"]
    merged_dir = source_cv_fold_dir(PHASE3_MERGED_DIR, fold_idx) / unit["prep_label"]

    if not dry_run and _marker_ok(merged_dir, unit):
        print(f"  fold {fold_idx} [{unit['prep_label']}]: KAN inputs already prepared, skip")
        return {s: merged_dir / f"{s}.pkl" for s in ("train", "val", "test")}

    non_context = [b for b in combo if b != "context"]
    shared = non_context + (["context"] if mode == "on" else [])
    latent_dirs: Dict[str, Path] = {}
    fold_vae = source_cv_fold_dir(SOURCE_CV_VAE_DIR, fold_idx)
    fold_raw = source_cv_fold_dir(SOURCE_CV_RAW_DIR, fold_idx)

    if shared:
        stale = [
            b for b in shared
            if dry_run or not latent_cache_is_fresh(b, latent_dims[b], fold_vae, raw_dir_override=fold_raw / b)
        ]
        if stale:
            cmd = [python_executable(), "main.py", *_source_disjoint_flags(fold_idx), "--preprocess_text"]
            for b in ALL_MODALITIES:
                cmd.append(f"--extract_{b}" if b in shared else f"--exclude_{b}")
                cmd += [f"--{b}_latent_dim", str(latent_dims.get(b, DEFAULT_LATENT_DIM[b]))]
            cmd += ["--semantic_device", EXTRACT_DEVICE, "--emotion_device", EXTRACT_DEVICE]
            cmd += ["--run_vaes", "--vae_beta", "1.0", "--vae_dropout", "0.1"]
            print(f"  fold {fold_idx} [{unit['prep_label']}]: preparing {shared} (shared source_cv cache)")
            print(f"    $ {' '.join(cmd)}")
            if not dry_run:
                outcome = run_main_command(cmd, require_results_json=False)
                if outcome["error"] is not None:
                    raise RuntimeError(f"fold {fold_idx} [{unit['prep_label']}]: prep failed: {outcome['error']}")
                print(f"      OK in {outcome['elapsed_seconds']}s")
        for b in shared:
            latent_dirs[b] = fold_vae / b / f"latent{latent_dims[b]}"

    if mode == "off":
        latent_dirs["context"] = ensure_idfree_context(latent_dims["context"], fold_idx=fold_idx, dry_run=dry_run)

    pkl_paths = merge_kan_inputs(combo, latent_dirs, merged_dir, dry_run)

    if not dry_run:
        with open(merged_dir / "_marker.json", "w", encoding="utf-8") as f:
            json.dump({"latent_dims": latent_dims, "context_identity": mode}, f, indent=2)
    return pkl_paths


def run_sweep(dry_run: bool) -> None:
    on_dim, off_dim = load_context_dims()
    prep_units = build_prep_units(on_dim, off_dim)
    n_folds = SOURCE_SPLIT_N_FOLDS
    total = total_kan_runs(prep_units, n_folds=n_folds)
    print(f"Phase 3: {len(prep_units)} combo-variants x 2 hidden_dim x {len(SEEDS)} seeds x {n_folds} folds = {total} runs")
    print(f"context dims: identity ON = {on_dim}, identity OFF = {off_dim}")
    print(f"Results: {PHASE3_RESULTS_JSONL}")

    ok_keys = load_ok_run_keys(PHASE3_RESULTS_JSONL) if not dry_run else set()
    if ok_keys:
        print(f"Resuming: {len(ok_keys)}/{total} runs already completed, skipping.")

    n_run = n_skip = n_failed = idx = 0

    for fold_idx in range(n_folds):
        prepared: Dict[str, Dict[str, Path]] = {}
        for unit, hidden_dim in iter_kan_entries(prep_units):
            combo = unit["active_extractors"]
            latent_dims = unit["latent_dims"]
            mode = unit["context_identity"]
            plabel = unit["prep_label"]
            elabel = entry_label(combo, mode, hidden_dim)

            if plabel not in prepared:
                print(f"\n=== Phase 3 -- fold {fold_idx}/{n_folds - 1} -- prep {plabel} ===")
                prepared[plabel] = prepare_fold(unit, fold_idx, dry_run)
            pkl_paths = prepared[plabel]

            for seed in SEEDS:
                idx += 1
                key = f"{elabel}__fold{fold_idx}__seed{seed}"
                run_label = f"[{idx:04d}/{total}] {key}"
                output_dir = KAN_RUNS_DIR / "phase3" / elabel / f"fold{fold_idx}" / f"seed{seed}"
                cmd = build_kan_cmd(
                    combo=combo, latent_dims=latent_dims, pkl_paths=pkl_paths,
                    hidden_dim=hidden_dim, seed=seed, output_dir=output_dir,
                    extra_flags=_source_disjoint_flags(fold_idx),
                )

                if dry_run:
                    print(f"{run_label}\n  $ {' '.join(cmd)}")
                    continue
                if key in ok_keys:
                    n_skip += 1
                    continue

                print(f"{run_label} RUN")
                record = execute_and_log(
                    run_key=key, cmd=cmd, jsonl_path=PHASE3_RESULTS_JSONL,
                    meta={
                        "phase": PHASE_NAME,
                        "entry_label": elabel,
                        "prep_label": plabel,
                        "active_extractors": combo,
                        "context_identity": mode,
                        "hidden_dim": hidden_dim,
                        "latent_dims": latent_dims,
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

    print(f"\nPhase 3 complete (this invocation): {n_run} new, {n_skip} skipped, {n_failed} failed.")
    print(f"Total accumulated in {PHASE3_RESULTS_JSONL}: {len(load_ok_run_keys(PHASE3_RESULTS_JSONL))}/{total} ok.")


def summarize() -> None:
    if not PHASE3_RESULTS_JSONL.exists():
        print(f"No results yet: {PHASE3_RESULTS_JSONL}")
        return

    rows: List[Dict[str, Any]] = []
    with open(PHASE3_RESULTS_JSONL, "r", encoding="utf-8") as f:
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
                "context_identity": record.get("context_identity"),
                "hidden_dim": record.get("hidden_dim"),
                "fold": record.get("fold"),
                "seed": record.get("seed"),
                RANKING_METRIC: record["metrics"].get(RANKING_METRIC),
                f"test_{RANKING_METRIC}": (record.get("test_metrics") or {}).get(RANKING_METRIC),
                "test_accuracy": (record.get("test_metrics") or {}).get("accuracy"),
            })
    if not rows:
        print("No successful runs logged yet.")
        return

    df = pd.DataFrame(rows)
    per_label: Dict[str, Any] = {}
    ranked: List[Dict[str, Any]] = []
    for label, group in df.groupby("entry_label"):
        per_fold = group.groupby("fold")[RANKING_METRIC].agg(["mean", "std", "min", "max", "count"])
        print(f"\n=== {label} -- per-fold val {RANKING_METRIC} (n={len(SEEDS)} seeds each) ===")
        print(per_fold.to_string())
        entry = {
            "entry_label": label,
            "active_extractors": group.iloc[0]["active_extractors"],
            "context_identity": group.iloc[0]["context_identity"],
            "hidden_dim": int(group.iloc[0]["hidden_dim"]),
            "n_runs": int(len(group)),
            "n_folds": int(group["fold"].nunique()),
            f"{RANKING_METRIC}_mean": float(group[RANKING_METRIC].mean()),
            f"{RANKING_METRIC}_std": float(group[RANKING_METRIC].std()),
            f"test_{RANKING_METRIC}_mean": float(group[f"test_{RANKING_METRIC}"].mean()),
            f"test_{RANKING_METRIC}_std": float(group[f"test_{RANKING_METRIC}"].std()),
            "test_accuracy_mean": float(group["test_accuracy"].mean()),
        }
        per_label[label] = {
            **entry,
            "per_fold": {
                str(fold): {k: (float(v) if pd.notna(v) else None) for k, v in row.items()}
                for fold, row in per_fold.iterrows()
            },
        }
        ranked.append(entry)
    ranked.sort(key=lambda r: r[f"{RANKING_METRIC}_mean"], reverse=True)

    PHASE3_PER_FOLD_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE3_PER_FOLD_JSON, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "configs": per_label}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {PHASE3_PER_FOLD_JSON}")

    print(f"\n=== Phase 3 -- all {len(ranked)} combo-variants, ranked by val {RANKING_METRIC} ===")
    for i, r in enumerate(ranked, start=1):
        print(f"  {i:2d}. {r['entry_label']:<42} "
              f"val {RANKING_METRIC}={r[f'{RANKING_METRIC}_mean']:.4f}  "
              f"test {RANKING_METRIC}={r[f'test_{RANKING_METRIC}_mean']:.4f}")

    with open(PHASE3_TOP_JSON, "w", encoding="utf-8") as f:
        json.dump({"metric": RANKING_METRIC, "seeds": SEEDS, "results": ranked}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {PHASE3_TOP_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 (Option A): source-disjoint validation of all combo-variants")
    parser.add_argument("--run", action="store_true", help="Run the sweep (resumable)")
    parser.add_argument("--summary", action="store_true", help="Aggregate results/orchestrator_phase3.jsonl")
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
