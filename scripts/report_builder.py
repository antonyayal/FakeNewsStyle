# scripts/report_builder.py
# =====================================================
# Compiles results/*.json (written by src/experiments/run_logger.py)
# into a comparison table, an extractor-combo heatmap, and per-run
# weight histograms (KAN classifier + per-modality VAE encoder/decoder).
#
# Usage:
#   python scripts/report_builder.py
#   python scripts/report_builder.py --split val --skip_weights
# =====================================================

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.experiments.run_logger import MODALITY_ORDER

DEFAULT_METRICS = ["accuracy", "f1", "roc_auc", "log_loss"]


# =====================================================
# Loading results/*.json
# =====================================================
def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    records = []

    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            records.append(json.load(f))

    if not records:
        print(f"[WARNING] No result JSON files found in: {results_dir}")

    return records


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    flat = {
        "run_id": record["run_id"],
        "timestamp": record["timestamp"],
        "git_commit": record.get("git_commit"),
        "active_extractors": "+".join(record["active_extractors"]),
        "n_extractors": len(record["active_extractors"]),
        "vae_epochs_requested": record["epochs"]["vae_epochs_requested"],
        "kan_epochs_requested": record["epochs"]["kan_epochs_requested"],
        "kan_epochs_run": record["epochs"]["kan_epochs_run"],
    }

    for modality in MODALITY_ORDER:
        flat[f"latent_dim_{modality}"] = record["latent_dims"].get(modality)

    for prefix, hyperparams in [
        ("vae", record.get("vae_hyperparams", {})),
        ("kan", record.get("kan_hyperparams", {})),
    ]:
        for k, v in hyperparams.items():
            flat[f"{prefix}_{k}"] = v

    for split, split_metrics in record.get("metrics", {}).items():
        for metric_name, value in split_metrics.items():
            flat[f"{split}_{metric_name}"] = value

    return flat


def build_summary_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [flatten_record(r) for r in records]
    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# =====================================================
# Heatmap: extractor combinations vs metrics
# =====================================================
def plot_extractor_heatmap(
    df: pd.DataFrame,
    split: str,
    metrics: List[str],
    out_path: Path,
) -> None:
    metric_cols = [f"{split}_{m}" for m in metrics if f"{split}_{m}" in df.columns]

    if not metric_cols:
        print(f"[WARNING] No '{split}' metrics found; skipping heatmap.")
        return

    pivot = (
        df.groupby("active_extractors")[metric_cols]
        .mean()
        .rename(columns=lambda c: c.replace(f"{split}_", ""))
    )

    plt.figure(figsize=(max(8, len(metric_cols) * 1.6), max(4, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "value"})
    plt.title(f"Extractor combinations vs. {split} metrics\n(mean across runs)", fontsize=11)
    plt.xlabel("Metric")
    plt.ylabel("Active extractors")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved heatmap: {out_path}")


# =====================================================
# KAN weight histograms (sliced by modality)
# =====================================================
def _modality_slices(active_extractors: List[str], latent_dims: Dict[str, int]):
    slices = {}
    offset = 0

    for modality in MODALITY_ORDER:
        if modality in active_extractors:
            dim = int(latent_dims[modality])
            slices[modality] = slice(offset, offset + dim)
            offset += dim

    return slices


def plot_kan_weight_histograms(record: Dict[str, Any], out_dir: Path) -> None:
    import torch

    checkpoint_path = record["paths"].get("kan_checkpoint_snapshot") or record["paths"].get("kan_checkpoint")
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"[WARNING] KAN checkpoint not found, skipping: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    coeffs = checkpoint["model_state_dict"]["model.0.coeffs"].numpy()  # (in_dim, hidden_dim, num_basis)

    slices = _modality_slices(record["active_extractors"], record["latent_dims"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for modality, sl in slices.items():
        values = coeffs[sl, :, :].flatten()

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=50, color="#4C72B0", alpha=0.85)
        plt.title(f"KAN input-layer weights — {modality}\nrun {record['run_id']}")
        plt.xlabel("weight value")
        plt.ylabel("count")
        plt.tight_layout()

        out_path = out_dir / f"kan_layer1_{modality}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    print(f"Saved KAN weight histograms for run {record['run_id']} in: {out_dir}")


# =====================================================
# VAE weight histograms (per modality, per Dense layer)
# =====================================================
def _warn_if_possibly_stale(model_path: Path, record_timestamp: str, run_id: str) -> None:
    try:
        record_time = datetime.fromisoformat(record_timestamp)
        file_time = datetime.fromtimestamp(model_path.stat().st_mtime).astimezone(record_time.tzinfo)

        if abs((file_time - record_time).total_seconds()) > 3600:
            print(
                f"[WARNING] {model_path} was modified long after run {run_id} finished — "
                "it was likely overwritten by a later run reusing the same modality+latent_dim. "
                "These histograms may not reflect this run's actual weights."
            )
    except Exception:
        pass


def plot_vae_weight_histograms(record: Dict[str, Any], out_dir: Path) -> None:
    from tensorflow import keras

    from src.models.train_vae_from_pkl import Sampling

    out_dir.mkdir(parents=True, exist_ok=True)
    vae_model_dirs = record["paths"].get("vae_model_dirs", {})

    for modality, model_dir in vae_model_dirs.items():
        model_dir = Path(model_dir)

        for part in ["encoder", "decoder"]:
            model_path = model_dir / f"{part}.keras"

            if not model_path.exists():
                print(f"[WARNING] Missing {model_path}, skipping.")
                continue

            _warn_if_possibly_stale(model_path, record["timestamp"], record["run_id"])

            model = keras.models.load_model(
                model_path,
                custom_objects={"Sampling": Sampling},
                compile=False,
            )

            for layer in model.layers:
                weights = layer.get_weights()

                if not weights:
                    continue

                kernel = weights[0]  # Dense kernel

                plt.figure(figsize=(6, 4))
                plt.hist(kernel.flatten(), bins=50, color="#DD8452", alpha=0.85)
                plt.title(f"VAE {part} [{layer.name}] — {modality}\nrun {record['run_id']}")
                plt.xlabel("weight value")
                plt.ylabel("count")
                plt.tight_layout()

                out_path = out_dir / f"vae_{modality}_{part}_{layer.name}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()

    print(f"Saved VAE weight histograms for run {record['run_id']} in: {out_dir}")


# =====================================================
# CLI
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Build experiment comparison report")

    parser.add_argument("--results_dir", type=str, default=str(BASE_DIR / "results"))
    parser.add_argument("--reports_dir", type=str, default=str(BASE_DIR / "reports"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--skip_weights", action="store_true", help="Skip per-run weight histograms (faster)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_results(results_dir)
    if not records:
        return

    df = build_summary_table(records)

    csv_path = reports_dir / "experiments_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table: {csv_path} ({len(df)} runs)")

    plot_extractor_heatmap(
        df,
        split=args.split,
        metrics=args.metrics,
        out_path=reports_dir / f"heatmap_extractor_combos_{args.split}.png",
    )

    if not args.skip_weights:
        for record in records:
            run_weights_dir = reports_dir / "weights" / record["run_id"]
            plot_kan_weight_histograms(record, run_weights_dir)
            plot_vae_weight_histograms(record, run_weights_dir)


if __name__ == "__main__":
    main()
