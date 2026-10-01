# scripts/paper_heatmap.py
# -*- coding: utf-8 -*-
"""
Builds the paper/thesis heatmap of all 23 Phase 2 configurations (standard
split): rows = extractor combination (identity on/off where context is
present), columns = test accuracy, macro-F1, ROC-AUC and log loss, cells =
mean +- std over every run of that configuration (both KAN widths x 3 seeds),
rows ranked by mean test macro-F1. Colour is normalised per column.

Input : results/orchestrator_phase2.jsonl
Output: the English figure for the article and the Spanish one for the thesis.

Usage:
    python scripts/paper_heatmap.py
    python scripts/paper_heatmap.py --en_out path.png --es_out path.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE2_JSONL = BASE_DIR / "results" / "orchestrator_phase2.jsonl"
DEFAULT_EN_OUT = BASE_DIR.parent / "FakeNewsStyle-paper" / "figures" / "Heatmap.png"
DEFAULT_ES_OUT = BASE_DIR.parent / "Tesis_Doctorado" / "figures" / "Heatmap_es.png"

METRICS = ["accuracy", "f1", "roc_auc", "log_loss"]
LABELS = {
    "en": {
        "cols": ["accuracy", "macro-F1", "ROC-AUC", "log loss"],
        "title": "Extractor combinations vs. test metrics (mean ± std over runs, ranked by test macro-F1)",
        "branch": {"semantic": "semantic", "emotion": "emotion", "style": "style", "context": "context"},
        "on": "identity on", "off": "identity off",
    },
    "es": {
        "cols": ["exactitud", "F1 macro", "ROC-AUC", "pérdida log."],
        "title": "Combinaciones de ramas: métricas de prueba (media ± d.e. entre corridas, ordenadas por F1 macro)",
        "branch": {"semantic": "semántica", "emotion": "emoción", "style": "estilo", "context": "contexto"},
        "on": "identidad activada", "off": "identidad desactivada",
    },
}
CMAP = LinearSegmentedColormap.from_list("f1", ["#5aa9dd", "#7d6f8c", "#962f2f"])


def load_runs() -> dict:
    groups = defaultdict(list)
    for line in PHASE2_JSONL.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("status") != "ok":
            continue
        key = (tuple(rec["active_extractors"]), rec.get("context_identity"))
        groups[key].append(rec["test_metrics"])
    return groups


def row_label(key, lang: str) -> str:
    extractors, identity = key
    lab = LABELS[lang]
    name = "+".join(lab["branch"][e] for e in extractors)
    if "context" in extractors and identity in ("on", "off"):
        name += f" ({lab[identity]})"
    return name


def draw(groups: dict, lang: str, out: Path) -> None:
    keys = sorted(groups, key=lambda k: -np.mean([m["f1"] for m in groups[k]]))
    means = np.array([[np.mean([m[c] for m in groups[k]]) for c in METRICS] for k in keys])
    stds = np.array([[np.std([m[c] for m in groups[k]], ddof=1) for c in METRICS] for k in keys])
    norm = (means - means.min(axis=0)) / np.ptp(means, axis=0)

    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=160)
    ax.imshow(norm, aspect="auto", cmap=CMAP, vmin=0, vmax=1)
    for i in range(len(keys)):
        for j in range(len(METRICS)):
            ax.text(j, i, f"{means[i, j]:.3f}±{stds[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(range(len(METRICS)), LABELS[lang]["cols"], fontsize=11)
    ax.set_yticks(range(len(keys)), [row_label(k, lang) for k in keys], fontsize=10)
    ax.set_title(LABELS[lang]["title"], fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--en_out", type=Path, default=DEFAULT_EN_OUT)
    parser.add_argument("--es_out", type=Path, default=DEFAULT_ES_OUT)
    args = parser.parse_args()
    groups = load_runs()
    draw(groups, "en", args.en_out)
    draw(groups, "es", args.es_out)


if __name__ == "__main__":
    main()
