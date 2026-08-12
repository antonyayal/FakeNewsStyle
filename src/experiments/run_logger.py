# src/experiments/run_logger.py
# -*- coding: utf-8 -*-
"""
Writes one JSON record per pipeline run to results/, capturing the
configuration and metrics needed to compare experiments later with
scripts/report_builder.py.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Canonical modality order. Must match the concatenation order used in
# main.py Step 9 (VAE latent merge) so weight-slicing in report_builder.py
# lines up with the columns actually fed into the KAN classifier.
MODALITY_ORDER = ["semantic", "emotion", "style", "context"]


def hash_files(paths: List[Path], algo: str = "md5") -> Optional[str]:
    """
    Combined hash of a set of files (e.g. the exact train/val/test PKLs fed
    into a KAN run), so two runs can be checked for having used identical
    data regardless of which corpus variant/split produced it. Returns None
    if any path is missing rather than raising, since this is best-effort
    traceability, not a required part of a run succeeding.
    """
    hasher = hashlib.new(algo)
    for path in sorted(Path(p) for p in paths):
        if not path.exists():
            return None
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _get_git_commit_hash(repo_dir: Path) -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def log_experiment_result(
    *,
    active_extractors: List[str],
    latent_dims: Dict[str, int],
    vae_epochs_requested: int,
    kan_epochs_requested: int,
    kan_epochs_run: int,
    vae_hyperparams: Dict[str, Any],
    kan_hyperparams: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    kan_output_dir: Path,
    kan_checkpoint_path: Path,
    vae_model_dirs: Dict[str, Path],
    base_dir: Path,
    results_dir: Optional[Path] = None,
    training_time_seconds: Optional[float] = None,
    num_parameters: Optional[int] = None,
    dataset_hash: Optional[str] = None,
    topic_breakdown: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Assembles one experiment record and writes it to results/{run_id}.json.

    The KAN checkpoint (small, ~a few hundred KB) is copied into
    results/checkpoints/ so weight histograms stay reproducible even if a
    later run overwrites kan_output_dir. VAE checkpoints (much larger) are
    NOT copied -- they are only overwritten when a run reuses the exact
    same modality+latent_dim, since each latent_dim gets its own directory
    under models/vae/{modality}/latent{N}/.
    """
    results_dir = results_dir or (base_dir / "results")
    checkpoints_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    excluded_extractors = [m for m in MODALITY_ORDER if m not in active_extractors]

    kan_checkpoint_copy = None
    kan_checkpoint_path = Path(kan_checkpoint_path)
    if kan_checkpoint_path.exists():
        kan_checkpoint_copy = checkpoints_dir / f"{run_id}_best_kan_model.pt"
        shutil.copy2(kan_checkpoint_path, kan_checkpoint_copy)

    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": _get_git_commit_hash(base_dir),
        "active_extractors": active_extractors,
        "excluded_extractors": excluded_extractors,
        "latent_dims": latent_dims,
        "epochs": {
            "vae_epochs_requested": vae_epochs_requested,
            "kan_epochs_requested": kan_epochs_requested,
            "kan_epochs_run": kan_epochs_run,
        },
        "vae_hyperparams": vae_hyperparams,
        "kan_hyperparams": kan_hyperparams,
        "metrics": metrics,
        "compute": {
            "training_time_seconds": training_time_seconds,
            "num_parameters": num_parameters,
        },
        "dataset_hash": dataset_hash,
        "topic_breakdown": topic_breakdown,
        "paths": {
            "kan_output_dir": str(kan_output_dir),
            "kan_checkpoint": str(kan_checkpoint_path),
            "kan_checkpoint_snapshot": str(kan_checkpoint_copy) if kan_checkpoint_copy else None,
            "vae_model_dirs": {k: str(v) for k, v in vae_model_dirs.items()},
        },
    }

    out_path = results_dir / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"Experiment record saved: {out_path}")
    if kan_checkpoint_copy:
        print(f"KAN checkpoint snapshot saved: {kan_checkpoint_copy}")

    return out_path
