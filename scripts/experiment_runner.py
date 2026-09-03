# scripts/experiment_runner.py
# -*- coding: utf-8 -*-
"""
Mechanics shared by orchestrator_phase{1..5}.py: launching main.py via
subprocess, capturing the results/{run_id}.json that src/experiments/
run_logger.py writes, appending a result line to the phase's centralized
JSON-lines (with checkpointing/resume based on run_key), and building the
VAE latents each phase's KAN runs read from -- either the shared default
cache (ensure_vae_latents) or an isolated per-(beta, dropout) one merged
manually (resolve_kan_input / merge_latents_manual), for candidates whose
vae_beta/vae_dropout aren't main.py's defaults.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from experiment_config import (
    ALL_MODALITIES,
    BASE_DIR,
    DEFAULT_LATENT_DIM,
    FEATURES_RAW_DIR,
    VAE_LATENTS_DIR,
)

DEFAULT_VAE_BETA = 1.0
DEFAULT_VAE_DROPOUT = 0.1

RESULTS_JSON_RE = re.compile(r"Experiment record saved:\s*(\S+\.json)")


def _row_count(pkl_path: Path) -> int:
    """Row count for either a DataFrame PKL (semantic/emotion, VAE latents)
    or a dict-payload PKL (style/context raw features, {"num_samples": N, ...})."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return int(obj["num_samples"])
    return len(obj)


def latent_cache_is_fresh(branch: str, dim: int, vae_latents_dir: Path) -> bool:
    """True if vae_latents_dir/{branch}/latent{dim}/{split}.pkl exists AND its
    row count matches the CURRENT data/03_features_raw/{branch}/{split}_{branch}.pkl
    -- i.e. these cached VAE latents were trained on the corpus that's on disk
    right now, not a stale snapshot from an earlier corpus revision.

    Callers previously trusted file *existence* alone (ensure_vae_latents /
    ensure_default_vae_latents), which silently reused latents trained on old
    corpus sizes (e.g. 971-row snapshots from May) after the corpus was cut
    down to 681 rows -- causing "Label length mismatch" crashes only once
    merged against a freshly trained branch, or worse, silently training/
    evaluating on stale data when every merged branch happened to be equally
    stale."""
    branch_dir = vae_latents_dir / branch / f"latent{dim}"
    for split in ["train", "val", "test"]:
        latent_pkl = branch_dir / f"{split}.pkl"
        if not latent_pkl.exists():
            return False

        raw_pkl = FEATURES_RAW_DIR / branch / f"{split}_{branch}.pkl"
        if not raw_pkl.exists():
            continue  # nothing to validate against -- existence is all we can check

        if _row_count(latent_pkl) != _row_count(raw_pkl):
            return False

    return True


def load_ok_run_keys(jsonl_path: Path) -> set:
    """Run keys with a successful ('ok') line already in the JSONL, so a
    resumed batch skips them. Missing file / unreadable lines are treated
    as "nothing done yet" rather than raising."""
    if not jsonl_path.exists():
        return set()

    ok_keys = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok" and "run_key" in record:
                ok_keys.add(record["run_key"])

    return ok_keys


def append_jsonl(jsonl_path: Path, record: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def run_main_command(cmd: List[str], require_results_json: bool = True) -> Dict[str, Any]:
    """Runs `python main.py ...` via subprocess, capturing stdout/stderr.
    Never raises on a failing run -- returns a dict describing what happened
    so callers can log it and keep going with the rest of the batch.

    require_results_json controls whether a missing "Experiment record
    saved: ..." line counts as a failure. That line is only printed by
    main.py's --train_kan step (see src/experiments/run_logger.py), so
    callers that only pass --run_vaes (ensure_vae_latents/resolve_kan_input
    in orchestrator_phase1.py/orchestrator_phase2.py) must pass
    require_results_json=False -- otherwise a successful VAE-only run
    (returncode 0, no results JSON to find) is misreported as failed.

    Returns both val_metrics and test_metrics (main.py's results JSON always
    has all three of train/val/test). val_metrics is what orchestrators must
    rank/select configs on -- test must never influence which config wins,
    only report how the already-chosen winner does on genuinely held-out
    data. (Before 2026-09-02 this returned test_metrics alone under the key
    "test_metrics" and callers used it for ranking -- see
    dataset_source_label_leakage / experiment_phases_status memory for why
    that was wrong: with ~860 KAN runs comparing configs, selecting by test
    F1 directly overfits every "winner" to the test set via search, on top
    of and independent from the Source/Domain leakage issue.)"""

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed = time.time() - start
        stdout, stderr = proc.stdout, proc.stderr
        returncode = proc.returncode
    except Exception as exc:  # e.g. main.py not found, permissions, etc.
        elapsed = time.time() - start
        return {
            "returncode": -1,
            "elapsed_seconds": round(elapsed, 1),
            "stdout": "",
            "stderr": "",
            "error": f"subprocess.run raised: {exc}",
            "results_json": None,
            "val_metrics": None,
            "test_metrics": None,
        }

    match = RESULTS_JSON_RE.search(stdout)
    results_json = match.group(1) if match else None

    error = None
    val_metrics = None
    test_metrics = None

    if returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-20:])
        error = f"main.py exited with code {returncode}. stderr tail:\n{tail}"
    elif results_json is None:
        if require_results_json:
            error = "main.py exited 0 but no 'Experiment record saved:' line found in stdout."
    else:
        results_path = Path(results_json)
        if not results_path.is_absolute():
            results_path = BASE_DIR / results_path
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            val_metrics = record["metrics"]["val"]
            test_metrics = record["metrics"]["test"]
        except Exception as exc:
            error = f"Could not read val/test metrics from {results_path}: {exc}"

    return {
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 1),
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "results_json": results_json,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def execute_and_log(
    *,
    run_key: str,
    cmd: List[str],
    jsonl_path: Path,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs cmd, then appends one JSON-lines record (ok or failed) merging
    `meta` (phase/group/config metadata) with the outcome. Never raises.

    "metrics" holds VAL split metrics -- this is what aggregate_results.py's
    RANKING_METRIC sorts/selects on. "test_metrics" holds the TEST split,
    carried along purely for final reporting on whichever config val
    already chose; it must never be used to pick a winner."""

    outcome = run_main_command(cmd)
    status = "ok" if outcome["error"] is None else "failed"

    record: Dict[str, Any] = {
        "run_key": run_key,
        **meta,
        "status": status,
        "error": outcome["error"],
        "command": cmd,
        "run_id": Path(outcome["results_json"]).stem if outcome["results_json"] else None,
        "results_json": outcome["results_json"],
        "elapsed_seconds": outcome["elapsed_seconds"],
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "metrics": outcome["val_metrics"],
        "test_metrics": outcome["test_metrics"],
    }

    append_jsonl(jsonl_path, record)
    return record


def python_executable() -> str:
    return sys.executable


def ensure_vae_latents(active_extractors: List[str], dims: Dict[str, int], dry_run: bool) -> None:
    """Default-beta/dropout path: reuses the shared cache at VAE_LATENTS_DIR,
    training only what's missing for this (active_extractors, dims) preset.
    Used by Phase 1 (one branch at a time) and Phase 2 (combos)."""
    missing = [
        f"{branch} (latent{dims[branch]}, missing or stale vs. current corpus)"
        for branch in active_extractors
        if not latent_cache_is_fresh(branch, dims[branch], VAE_LATENTS_DIR)
    ]

    if not missing:
        return

    print(f"    Missing VAE latents for this preset: {missing}")
    cmd = [python_executable(), "main.py", "--run_vaes"]
    for branch in ALL_MODALITIES:
        if branch not in active_extractors:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(dims.get(branch, DEFAULT_LATENT_DIM[branch]))]
    print(f"    $ {' '.join(cmd)}")

    if dry_run:
        print("    (dry-run: not executing)")
        return

    outcome = run_main_command(cmd, require_results_json=False)
    if outcome["error"] is not None:
        raise RuntimeError(f"Failed training VAE for {active_extractors} @ {dims}: {outcome['error']}")
    print(f"    OK in {outcome['elapsed_seconds']}s")


def is_default_vae_reg(
    effective: Dict[str, Any],
    default_beta: float = DEFAULT_VAE_BETA,
    default_dropout: float = DEFAULT_VAE_DROPOUT,
) -> bool:
    return effective["vae_beta"] == default_beta and effective["vae_dropout"] == default_dropout


def merge_latents_manual(
    active_extractors: List[str], latent_dirs: Dict[str, Path], out_dir: Path
) -> Dict[str, Path]:
    """Mirrors main.py's Step 9 (--merge_vae_latents) column-prefixing logic
    (same {branch}_ prefix convention, same label handling), but reads from
    arbitrary latent_dirs instead of the hardcoded data/05_vae_latents/ path
    -- needed for VAE beta/dropout candidates that main.py can't merge on
    its own."""
    out_paths = {}
    for split in ["train", "val", "test"]:
        dfs = []
        labels = None
        for branch in active_extractors:
            df = pd.read_pickle(latent_dirs[branch] / f"{split}.pkl")
            if "label" in df.columns:
                current_labels = df["label"].reset_index(drop=True)
                if labels is None:
                    labels = current_labels
                df = df.drop(columns=["label"])
            df = df.reset_index(drop=True)
            df.columns = [c if str(c).startswith(f"{branch}_") else f"{branch}_{c}" for c in df.columns]
            dfs.append(df)

        merged_df = pd.concat(dfs, axis=1)
        if labels is not None:
            merged_df["label"] = labels.values

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.pkl"
        merged_df.to_pickle(out_path)
        out_paths[split] = out_path

    return out_paths


def resolve_kan_input(
    combo: List[str],
    config_label_: str,
    variant_label: str,
    effective: Dict[str, Any],
    vae_data_base_dir: Path,
    vae_model_base_dir: Path,
    merged_base_dir: Path,
    dry_run: bool,
) -> Optional[Dict[str, Path]]:
    """Decides how a candidate's KAN input is produced. Returns None when
    (vae_beta, vae_dropout) match main.py's defaults -- callers should then
    pass --merge_vae_latents and let main.py read the shared cache (after
    ensure_vae_latents makes sure this latent preset exists in it). Returns
    a {split: path} dict when they don't -- an isolated VAE was trained (if
    missing) under vae_data_base_dir/vae_model_base_dir and manually merged
    into merged_base_dir, and callers should point --train_kan at those
    paths directly instead of --merge_vae_latents."""

    if is_default_vae_reg(effective):
        ensure_vae_latents(combo, effective["latent"], dry_run=dry_run)
        return None

    tag = f"beta{effective['vae_beta']}_drop{effective['vae_dropout']}"
    vae_data_dir = vae_data_base_dir / config_label_ / tag
    vae_model_dir = vae_model_base_dir / config_label_ / tag
    merged_dir = merged_base_dir / config_label_ / variant_label

    latent_dirs = {branch: vae_data_dir / branch / f"latent{effective['latent'][branch]}" for branch in combo}
    missing = [
        branch for branch in combo
        if not latent_cache_is_fresh(branch, effective["latent"][branch], vae_data_dir)
    ]

    if missing:
        print(f"    Missing isolated VAE (beta={effective['vae_beta']}, dropout={effective['vae_dropout']}): {missing}")
        cmd = [python_executable(), "main.py", "--run_vaes"]
        for branch in ALL_MODALITIES:
            if branch not in combo:
                cmd.append(f"--exclude_{branch}")
            cmd += [f"--{branch}_latent_dim", str(effective["latent"].get(branch, DEFAULT_LATENT_DIM[branch]))]
        cmd += [
            "--vae_beta", str(effective["vae_beta"]),
            "--vae_dropout", str(effective["vae_dropout"]),
            "--vae_data_output_dir", str(vae_data_dir.relative_to(BASE_DIR)),
            "--vae_model_output_dir", str(vae_model_dir.relative_to(BASE_DIR)),
        ]
        print(f"    $ {' '.join(cmd)}")

        if dry_run:
            print("    (dry-run: not executing)")
        else:
            outcome = run_main_command(cmd, require_results_json=False)
            if outcome["error"] is not None:
                raise RuntimeError(
                    f"Failed training isolated VAE for {combo} "
                    f"@ beta={effective['vae_beta']} dropout={effective['vae_dropout']}: {outcome['error']}"
                )
            print(f"    OK in {outcome['elapsed_seconds']}s")

    if dry_run:
        return {s: merged_dir / f"{s}.pkl" for s in ["train", "val", "test"]}

    return merge_latents_manual(combo, latent_dirs, merged_dir)
