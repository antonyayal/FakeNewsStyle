# scripts/comparison_report.py
# =====================================================
# Builds a comparison report scoped to one batch produced by
# scripts/run_experiments.py (results/batches/{batch_id}.json).
#
# Reuses scripts/report_builder.py's plotting/flattening functions
# (same heatmap + weight-histogram logic) instead of duplicating them,
# and additionally writes a ranked table and a Markdown findings
# summary scoped to just this batch's runs.
#
# Usage:
#   python scripts/comparison_report.py --batch results/batches/<id>.json
#   python scripts/comparison_report.py                 # uses the most recent batch
# =====================================================

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import report_builder as rb  # reuse flatten_record / heatmap / weight-histogram code

RANK_METRIC = "test_accuracy"
RANK_METRIC_2 = "test_f1"
HEATMAP_METRICS = ["accuracy", "f1", "roc_auc", "log_loss"]


# =====================================================
# Loading
# =====================================================
def latest_batch_path(batches_dir: Path) -> Path:
    candidates = sorted(batches_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No batch manifests found in {batches_dir}")
    return candidates[-1]


def load_batch_records(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    for row in manifest["runs"]:
        if row["status"] != "ok" or not row.get("results_json"):
            continue
        with open(row["results_json"], "r", encoding="utf-8") as f:
            record = json.load(f)
        record["_batch_label"] = row["label"]
        record["_batch_phase"] = row["phase"]
        records.append(record)
    return records


def build_ranked_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        flat = rb.flatten_record(record)
        flat["label"] = record["_batch_label"]
        flat["phase"] = record["_batch_phase"]
        rows.append(flat)

    df = pd.DataFrame(rows)
    lead_cols = ["label", "phase", "active_extractors", RANK_METRIC, RANK_METRIC_2, "test_log_loss"]
    other_cols = [c for c in df.columns if c not in lead_cols]
    df = df[lead_cols + other_cols]
    df = df.sort_values(RANK_METRIC, ascending=False).reset_index(drop=True)
    return df


# =====================================================
# Markdown findings summary
# =====================================================
def _row_by_label(df: pd.DataFrame, label: str):
    match = df[df["label"] == label]
    return match.iloc[0] if len(match) else None


def build_markdown_summary(
    df: pd.DataFrame,
    manifest: Dict[str, Any],
    generated_at: str,
) -> str:
    best = df.iloc[0]
    best_combo = manifest["resolved_best_combo"]
    best_combo_label = "_".join(best_combo) if len(best_combo) <= 2 else "+".join(best_combo)

    lines = [
        f"# Experiment batch comparison — {manifest['batch_id']}",
        "",
        f"Generated: {generated_at}  ",
        f"Runs in this batch: {len(df)}  ",
        f"Selection metric for Phase 2 (best combo): `{manifest['best_combo_metric']}`, tiebreak `{manifest['best_combo_tiebreak']}`",
        "",
        "## Best overall run",
        "",
        f"**`{best['label']}`** ({best['active_extractors']}) — "
        f"test accuracy **{best[RANK_METRIC]:.4f}**, F1 **{best[RANK_METRIC_2]:.4f}**, "
        f"log_loss {best['test_log_loss']:.4f}.",
        "",
        "## Extractor contribution (isolated runs)",
        "",
        "| Extractor | test accuracy | test F1 | test ROC-AUC |",
        "|---|---|---|---|",
    ]

    isolated_labels = ["semantic_only", "emotion_only", "style_only", "context_only"]
    isolated = df[df["label"].isin(isolated_labels)].sort_values(RANK_METRIC, ascending=False)
    for _, r in isolated.iterrows():
        lines.append(f"| {r['active_extractors']} | {r[RANK_METRIC]:.4f} | {r[RANK_METRIC_2]:.4f} | {r['test_roc_auc']:.4f} |")

    if len(isolated):
        top = isolated.iloc[0]
        bottom = isolated.iloc[-1]
        lines += [
            "",
            f"Strongest alone: **{top['active_extractors']}** (accuracy {top[RANK_METRIC]:.4f}). "
            f"Weakest alone: **{bottom['active_extractors']}** (accuracy {bottom[RANK_METRIC]:.4f}).",
        ]

    lines += [
        "",
        "## Combinations (Phase 1)",
        "",
        "| Combination | test accuracy | test F1 | test ROC-AUC |",
        "|---|---|---|---|",
    ]
    phase1 = df[df["phase"].str.startswith("1-")].sort_values(RANK_METRIC, ascending=False)
    for _, r in phase1.iterrows():
        lines.append(f"| {r['active_extractors']} | {r[RANK_METRIC]:.4f} | {r[RANK_METRIC_2]:.4f} | {r['test_roc_auc']:.4f} |")

    lines += [
        "",
        f"**Best combination: `{best_combo_label}`**, automatically selected after Phase 1 "
        f"and used as the fixed basis for the epoch/latent-dim sweeps below.",
        "",
        "## Effect of epochs (combo held fixed)",
        "",
        "| Run | kan_epochs | kan_patience | kan_epochs_run | test accuracy | test F1 |",
        "|---|---|---|---|---|---|",
    ]

    epoch_rows = df[(df["phase"] == "2a-epochs") | (df["label"] == "_".join(best_combo))]
    baseline_label = "_".join(best_combo) if "_".join(best_combo) in df["label"].values else None
    epoch_candidates = ["epochs_short", baseline_label, "epochs_long"] if baseline_label else ["epochs_short", "epochs_long"]
    for label in epoch_candidates:
        if label is None:
            continue
        r = _row_by_label(df, label)
        if r is not None:
            lines.append(
                f"| {label} | {int(r['kan_epochs_requested'])} | {int(r.get('kan_patience', 0)) if 'kan_patience' in r else '-'} "
                f"| {int(r['kan_epochs_run'])} | {r[RANK_METRIC]:.4f} | {r[RANK_METRIC_2]:.4f} |"
            )

    lines += [
        "",
        "## Effect of latent dimension (combo held fixed)",
        "",
        "| Run | latent dims (per active modality) | test accuracy | test F1 |",
        "|---|---|---|---|",
    ]

    latent_candidates = ["latent_small", baseline_label, "latent_large"] if baseline_label else ["latent_small", "latent_large"]
    latent_dim_cols = [c for c in df.columns if c.startswith("latent_dim_")]
    for label in latent_candidates:
        if label is None:
            continue
        r = _row_by_label(df, label)
        if r is not None:
            dims = ", ".join(f"{c.replace('latent_dim_', '')}={int(r[c])}" for c in latent_dim_cols if r[c] == r[c])
            lines.append(f"| {label} | {dims} | {r[RANK_METRIC]:.4f} | {r[RANK_METRIC_2]:.4f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- All 13 runs share the same fixed KAN hyperparameters (hidden_dim=32, num_basis=8, dropout=0.5, "
        "weight_decay=1e-3, batch_size=32); only extractor combination, epochs/patience, and latent_dim vary per run.",
        "- Full per-run metrics (train/val/test) are in `comparison_table.csv` in this same folder.",
        "- See `heatmap_extractor_combos_test.png` for the extractor-combo-vs-metric heatmap and `weights/` for "
        "per-run weight histograms (KAN first layer sliced by modality, plus VAE encoder/decoder layers).",
    ]

    return "\n".join(lines) + "\n"


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Build a comparison report for one run_experiments.py batch")
    parser.add_argument("--batch", type=str, default=None, help="Path to results/batches/{batch_id}.json (default: most recent)")
    parser.add_argument("--skip_weights", action="store_true")
    args = parser.parse_args()

    batches_dir = BASE_DIR / "results" / "batches"
    batch_path = Path(args.batch) if args.batch else latest_batch_path(batches_dir)

    with open(batch_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = load_batch_records(manifest)
    if not records:
        print("No successful runs in this batch.")
        return

    df = build_ranked_table(records)

    out_dir = BASE_DIR / "reports" / f"comparison_{manifest['batch_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_at = rb._format_ts(datetime.now().astimezone().isoformat(timespec="seconds"))

    csv_path = out_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved ranked table: {csv_path}")

    # Only Phase 1 rows are a clean combo-vs-combo comparison (fixed epochs/latent_dim).
    # Phase 2 rows reuse the winning combo while varying epochs/latent_dim, which would
    # otherwise get averaged into that combo's heatmap cell alongside the Phase 1 run.
    phase1_df = df[df["phase"].str.startswith("1-")]

    rb.plot_extractor_heatmap(
        phase1_df,
        split="test",
        metrics=HEATMAP_METRICS,
        out_path=out_dir / "heatmap_extractor_combos_test.png",
        generated_at=generated_at,
    )

    if not args.skip_weights:
        for record in records:
            run_weights_dir = out_dir / "weights" / f"{record['_batch_label']}_{record['run_id']}"
            rb.plot_kan_weight_histograms(record, run_weights_dir)
            rb.plot_vae_weight_histograms(record, run_weights_dir)

    summary_md = build_markdown_summary(df, manifest, generated_at)
    summary_path = out_dir / "SUMMARY.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    print(f"\nComparison report complete: {out_dir}")


if __name__ == "__main__":
    main()
