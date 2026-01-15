# =====================================================
# FakeNewsStyle - Experiment runner (STEP 1: skeleton)
# =====================================================
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


# =====================================================
# Helpers
# =====================================================
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_config(config_path: str) -> Dict[str, Any]:
    """
    STEP 1: only JSON supported.
    Later you can extend to YAML if needed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_text(path: Path, content: str) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def _now_ts() -> int:
    return int(time.time())


def _resolve_run_dir(out_dir: str, config: Dict[str, Any]) -> Path:
    """
    Use experiment_name from config if present, otherwise 'default'.
    """
    exp_name = str(config.get("experiment_name", "default")).strip() or "default"
    return Path(out_dir) / exp_name


# =====================================================
# Public API used by main.py
# =====================================================
def run_train(config_path: str, out_dir: str = "./runs") -> Optional[str]:
    """
    Mode: train
    STEP 1 behavior:
      - Validates config exists
      - Creates a run directory
      - Creates a placeholder checkpoint file (best.ckpt)
      - Saves train/val placeholder metrics
      - Returns path to best.ckpt
    """
    config = _load_config(config_path)
    run_dir = _resolve_run_dir(out_dir, config)
    _ensure_dir(run_dir)

    # Placeholder: in STEP 2 you will replace this with real training
    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    _write_text(
        ckpt_path,
        content=(
            "FAKENEWSSTYLE PLACEHOLDER CHECKPOINT\n"
            f"created_at_unix={_now_ts()}\n"
            f"config_path={Path(config_path).resolve()}\n"
        ),
    )

    # Placeholder metrics
    train_val_metrics = {
        "status": "placeholder",
        "message": "Training not implemented yet. This is a stub checkpoint.",
        "created_at_unix": _now_ts(),
        "config_path": str(Path(config_path).resolve()),
        "best_ckpt": str(ckpt_path.resolve()),
        "val": {
            "metric": None,
            "note": "No validation executed yet (placeholder).",
        },
    }
    _save_json(run_dir / "metrics" / "train_val_metrics.json", train_val_metrics)
    _save_json(run_dir / "artifacts" / "config_used.json", config)

    return str(ckpt_path.resolve())


def run_test(config_path: str, ckpt_path: str, out_dir: str = "./runs") -> Dict[str, Any]:
    """
    Mode: test
    STEP 1 behavior:
      - Validates config exists
      - Validates ckpt file exists
      - Saves placeholder test metrics
      - Returns test metrics dict
    """
    config = _load_config(config_path)
    run_dir = _resolve_run_dir(out_dir, config)
    _ensure_dir(run_dir)

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    # Placeholder test metrics
    test_metrics = {
        "status": "placeholder",
        "message": "Test not implemented yet. Only validated checkpoint exists.",
        "created_at_unix": _now_ts(),
        "config_path": str(Path(config_path).resolve()),
        "ckpt_path": str(ckpt_file.resolve()),
        "test": {
            "metric": None,
            "note": "No test executed yet (placeholder).",
        },
    }
    _save_json(run_dir / "metrics" / "test_metrics.json", test_metrics)

    return test_metrics


def run_train_test(config_path: str, out_dir: str = "./runs") -> Dict[str, Any]:
    """
    Mode: train_test
    STEP 1 behavior:
      - Calls run_train() to produce a placeholder best.ckpt
      - Calls run_test() using that checkpoint
      - Returns test metrics
    """
    best_ckpt = run_train(config_path=config_path, out_dir=out_dir)
    if not best_ckpt:
        raise RuntimeError("run_train did not return a checkpoint path")
    return run_test(config_path=config_path, ckpt_path=best_ckpt, out_dir=out_dir)
