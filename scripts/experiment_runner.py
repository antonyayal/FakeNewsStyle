# scripts/experiment_runner.py
# -*- coding: utf-8 -*-
"""
Mecánica compartida por orchestrator_phase1.py y orchestrator_phase2.py:
lanzar main.py vía subprocess, capturar el results/{run_id}.json que escribe
src/experiments/run_logger.py, y apendear una línea de resultado al JSON-lines
centralizado de la fase -- con checkpointing/resume basado en run_key.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_config import BASE_DIR

RESULTS_JSON_RE = re.compile(r"Experiment record saved:\s*(\S+\.json)")


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


def run_main_command(cmd: List[str]) -> Dict[str, Any]:
    """Runs `python main.py ...` via subprocess, capturing stdout/stderr.
    Never raises on a failing run -- returns a dict describing what happened
    so callers can log it and keep going with the rest of the batch."""

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
