# scripts/experiment_runner.py
# -*- coding: utf-8 -*-
"""
Mechanics shared by orchestrator_phase1.py and orchestrator_phase2.py:
launching main.py via subprocess, capturing the results/{run_id}.json that
src/experiments/run_logger.py writes, and appending a result line to the
phase's centralized JSON-lines -- with checkpointing/resume based on run_key.
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

from experiment_config import BASE_DIR, FEATURES_RAW_DIR

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
    (returncode 0, no results JSON to find) is misreported as failed."""

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
            "test_metrics": None,
        }

    match = RESULTS_JSON_RE.search(stdout)
    results_json = match.group(1) if match else None

    error = None
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
            test_metrics = record["metrics"]["test"]
        except Exception as exc:
            error = f"Could not read test metrics from {results_path}: {exc}"

    return {
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 1),
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "results_json": results_json,
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
    `meta` (phase/group/config metadata) with the outcome. Never raises."""

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
        "metrics": outcome["test_metrics"],
    }

    append_jsonl(jsonl_path, record)
    return record


def python_executable() -> str:
    return sys.executable
