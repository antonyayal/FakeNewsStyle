# scripts/html_report_builder.py
# =====================================================
# Builds a single self-contained HTML comparison report from experiment
# results written by src/experiments/run_logger.py (results/*.json) and,
# optionally, scoped to one batch manifest written by
# scripts/run_experiments.py (results/batches/{batch_id}.json).
#
# results/batches/{batch_id}.json is a flat manifest, not a folder of
# copied run JSONs: each of its "runs" entries points (via results_json)
# back at the same loose files under results/. Loading "all" results
# already includes every batch run once; batch scoping just filters to
# the subset a manifest references and tags each with its label/phase.
#
# Usage:
#   python scripts/html_report_builder.py                # interactive prompt
#   python scripts/html_report_builder.py --all
#   python scripts/html_report_builder.py --batch-id 20260811_235144
# =====================================================

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.experiments.run_logger import MODALITY_ORDER  # noqa: E402

RESULTS_DIR = BASE_DIR / "results"
BATCHES_DIR = RESULTS_DIR / "batches"
REPORTS_DIR = BASE_DIR / "reports"

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"


# =====================================================
# Discovery / loading
# =====================================================
def list_batches() -> List[Path]:
    if not BATCHES_DIR.exists():
        return []
    return sorted(BATCHES_DIR.glob("*.json"))


def load_manifest(batch_path: Path) -> Dict[str, Any]:
    with open(batch_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _batch_tags_by_run_id() -> Dict[str, Tuple[str, str, str]]:
    """Maps run_id -> (batch_id, label, phase) across every manifest in results/batches/."""
    tags: Dict[str, Tuple[str, str, str]] = {}
    for batch_path in list_batches():
        manifest = load_manifest(batch_path)
        for run in manifest.get("runs", []):
            if run.get("run_id"):
                tags[run["run_id"]] = (manifest["batch_id"], run.get("label"), run.get("phase"))
    return tags


def load_loose_results(results_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(results_dir.glob("*.json")):  # top-level only; skips batches/, checkpoints/
        with open(path, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def load_all_scope() -> Tuple[List[Dict[str, Any]], str]:
    records = load_loose_results(RESULTS_DIR)
    tags = _batch_tags_by_run_id()
    for record in records:
        tag = tags.get(record.get("run_id"))
        if tag:
            record["_batch_id"], record["_batch_label"], record["_batch_phase"] = tag
    return records, "Todas las corridas disponibles"


def load_batch_scope(batch_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    manifest = load_manifest(batch_path)
    records = []
    for run in manifest.get("runs", []):
        if run.get("status") != "ok" or not run.get("results_json"):
            continue
        results_path = Path(run["results_json"])
        if not results_path.exists():
            print(f"[WARNING] results_json not found, skipping: {results_path}")
            continue
        with open(results_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        record["_batch_id"] = manifest["batch_id"]
        record["_batch_label"] = run.get("label")
        record["_batch_phase"] = run.get("phase")
        records.append(record)
    scope_desc = f"Lote {manifest['batch_id']} ({len(records)} corridas)"
    return records, scope_desc


# =====================================================
# Interactive / CLI scope selection
# =====================================================
def prompt_scope_interactive() -> Tuple[str, Optional[Path]]:
    batches = list_batches()
    print("\n¿Qué quieres incluir en el reporte?")
    print("  1) Todas las corridas disponibles (results/ + todos los lotes)")
    print("  2) Solo un lote específico de results/batches/")
    choice = input("Elige una opción [1/2]: ").strip()

    if choice != "2":
        return "all", None

    if not batches:
        print("No hay lotes disponibles en results/batches/. Usando 'todas las corridas'.")
        return "all", None

    print("\nLotes disponibles:")
    for i, batch_path in enumerate(batches, start=1):
        manifest = load_manifest(batch_path)
        n_runs = len(manifest.get("runs", []))
        print(f"  {i}) {manifest['batch_id']}  ({n_runs} corridas, creado {manifest.get('created_at', '?')})")

    idx_raw = input(f"Elige un lote [1-{len(batches)}]: ").strip()
    try:
        idx = int(idx_raw)
        if not (1 <= idx <= len(batches)):
            raise ValueError
    except ValueError:
        print(f"Selección inválida: {idx_raw!r}. Cancelando.")
        sys.exit(1)

    return "batch", batches[idx - 1]


def resolve_scope(args: argparse.Namespace) -> Tuple[str, Optional[Path]]:
    if args.all:
        return "all", None
    if args.batch_id:
        batch_path = BATCHES_DIR / f"{args.batch_id}.json"
        if not batch_path.exists():
            available = ", ".join(p.stem for p in list_batches()) or "(ninguno)"
            print(f"[ERROR] No existe results/batches/{args.batch_id}.json. Lotes disponibles: {available}")
            sys.exit(1)
        return "batch", batch_path
    return prompt_scope_interactive()


# =====================================================
# Flattening for the DataFrame / JS payload
# =====================================================
def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metrics = record.get("metrics", {})
    test = metrics.get("test", {})
    train = metrics.get("train", {})
    epochs = record.get("epochs", {})
    kan_hyperparams = record.get("kan_hyperparams", {})
    compute = record.get("compute", {}) or {}

    return {
        "run_id": record["run_id"],
        "timestamp": record["timestamp"],
        "git_commit": record.get("git_commit"),
        "active_extractors": record.get("active_extractors", []),
        "excluded_extractors": record.get("excluded_extractors", []),
        "combo": "+".join(record.get("active_extractors", [])) or "(none)",
        "latent_dims": record.get("latent_dims", {}),
        "epochs": epochs,
        "vae_hyperparams": record.get("vae_hyperparams", {}),
        "kan_hyperparams": kan_hyperparams,
        "metrics": metrics,
        "paths": record.get("paths", {}),
        "batch_id": record.get("_batch_id"),
        "batch_label": record.get("_batch_label"),
        "batch_phase": record.get("_batch_phase"),
        "train_accuracy": train.get("accuracy"),
        "test_accuracy": test.get("accuracy"),
        "test_f1": test.get("f1"),
        "test_roc_auc": test.get("roc_auc"),
        "test_log_loss": test.get("log_loss"),
        "test_ece": test.get("ece"),
        "test_brier_score": test.get("brier_score"),
        # New: reproducibility / cost / traceability (may be absent on older records)
        "seed": kan_hyperparams.get("seed"),
        "training_time_seconds": compute.get("training_time_seconds"),
        "num_parameters": compute.get("num_parameters"),
        "dataset_hash": record.get("dataset_hash"),
        "topic_breakdown": record.get("topic_breakdown"),
    }


def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [flatten_record(r) for r in records]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# =====================================================
# HTML rendering
# =====================================================
CSS = """
:root {
  --bg: #0f1420; --panel: #171d2c; --panel-alt: #1e2537; --border: #2b3348;
  --text: #e6e9f2; --text-dim: #9aa3b8; --accent: #6ea8fe; --good: #3ddc97;
  --warn: #ffb454; --bad: #ff6b6b;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 24px 32px 64px; line-height: 1.5;
}
h1, h2, h3 { font-weight: 600; }
h1 { font-size: 1.6rem; margin-bottom: 4px; }
h2 { font-size: 1.2rem; margin-top: 40px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
header.report-header {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 20px; margin-bottom: 8px;
}
header.report-header .meta { color: var(--text-dim); font-size: 0.92rem; margin-top: 6px; }
header.report-header .meta span { margin-right: 22px; }
.badge {
  display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.76rem;
  font-weight: 600; margin: 1px 3px 1px 0; color: #0b0e16;
}
.badge-semantic { background: #6ea8fe; }
.badge-emotion  { background: #ffb454; }
.badge-style    { background: #3ddc97; }
.badge-context  { background: #d98cff; }
.controls { display: flex; gap: 12px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
.controls input[type="text"] {
  background: var(--panel-alt); border: 1px solid var(--border); color: var(--text);
  padding: 7px 12px; border-radius: 6px; width: 320px; font-size: 0.9rem;
}
.controls label { color: var(--text-dim); font-size: 0.85rem; display: flex; align-items: center; gap: 6px; }
.controls input[type="number"] {
  background: var(--panel-alt); border: 1px solid var(--border); color: var(--text);
  padding: 6px 8px; border-radius: 6px; width: 70px; font-size: 0.85rem;
}
.narrative {
  background: var(--panel); border: 1px solid var(--accent); border-radius: 10px;
  padding: 16px 20px; margin: 16px 0 8px;
}
.narrative h3 { margin: 0 0 10px; font-size: 1rem; color: var(--accent); }
.narrative ul { margin: 0; padding-left: 20px; }
.narrative li { margin-bottom: 6px; font-size: 0.92rem; }
.narrative .warn-item { color: var(--warn); }
table.summary {
  width: 100%; border-collapse: collapse; background: var(--panel); border-radius: 8px; overflow: hidden;
  font-size: 0.88rem;
}
table.summary th, table.summary td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--border); }
table.summary th { background: var(--panel-alt); cursor: pointer; user-select: none; white-space: nowrap; }
table.summary th:hover { color: var(--accent); }
table.summary th .arrow { opacity: 0.5; font-size: 0.75em; margin-left: 4px; }
table.summary tbody tr:hover { background: #1c2333; }
.epoch-early { color: var(--warn); font-weight: 600; }
.epoch-early::after { content: " (early stop)"; font-weight: 400; font-size: 0.85em; opacity: 0.85; }
.gap-warn { color: var(--bad); font-weight: 700; }
.degenerate-tag {
  display: inline-block; background: var(--bad); color: #150707; font-weight: 700;
  font-size: 0.75rem; padding: 2px 8px; border-radius: 6px; margin-left: 4px;
}
tr.degenerate-row { background: rgba(255, 107, 107, 0.08); }
.degenerate-banner {
  background: rgba(255, 107, 107, 0.12); border: 1px solid var(--bad); color: var(--bad);
  border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem; font-weight: 600;
}
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.chart-grid.full { grid-template-columns: 1fr; }
.chart-box { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
@media (max-width: 1000px) { .chart-grid { grid-template-columns: 1fr; } }
details.run-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  margin-bottom: 10px; padding: 10px 16px;
}
details.run-card summary {
  cursor: pointer; font-size: 0.95rem; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
details.run-card summary .run-id { font-family: monospace; color: var(--accent); }
details.run-card summary .metric { color: var(--text-dim); font-size: 0.85rem; }
.card-body { margin-top: 14px; display: grid; grid-template-columns: 260px 1fr; gap: 24px; }
@media (max-width: 800px) { .card-body { grid-template-columns: 1fr; } }
table.confusion { border-collapse: collapse; }
table.confusion td, table.confusion th {
  border: 1px solid var(--border); padding: 10px 16px; text-align: center; font-size: 0.85rem;
}
table.confusion th { background: var(--panel-alt); }
table.confusion td.tp, table.confusion td.tn { color: var(--good); font-weight: 600; }
table.confusion td.fp, table.confusion td.fn { color: var(--bad); font-weight: 600; }
table.mini { border-collapse: collapse; width: 100%; font-size: 0.83rem; }
table.mini th, table.mini td { border: 1px solid var(--border); padding: 6px 10px; text-align: right; }
table.mini th:first-child, table.mini td:first-child { text-align: left; }
table.mini th { background: var(--panel-alt); }
table.mini tbody tr:hover { background: #1c2333; }
.section-block { margin-top: 18px; }
.section-block h4 {
  margin: 0 0 8px; font-size: 0.82rem; color: var(--text-dim); text-transform: uppercase;
}
.seed-stability-table { margin-top: 10px; }
.hparams-cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.hparams-cols h4 { margin: 0 0 6px; font-size: 0.82rem; color: var(--text-dim); text-transform: uppercase; }
.hparams-cols dl { margin: 0; font-size: 0.85rem; }
.hparams-cols dt { color: var(--text-dim); float: left; clear: left; width: 60%; }
.hparams-cols dd { margin: 0 0 3px; text-align: right; }
.empty-msg { color: var(--text-dim); padding: 30px; text-align: center; }
footer { margin-top: 50px; color: var(--text-dim); font-size: 0.8rem; text-align: center; }
"""

JS = """
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const runs = DATA.runs;

const DEGENERATE_EPS = 0.02;

function fmtTs(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString('es', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch (e) { return iso; }
}
function fmtNum(v, digits = 4) {
  return (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(digits);
}
function extractorBadges(list) {
  if (!list || !list.length) return '<span style="color:var(--text-dim)">(none)</span>';
  return list.map(m => `<span class="badge badge-${m}">${m}</span>`).join('');
}

// train_accuracy - test_accuracy; null if either metric is missing (older records).
function overfitGap(r) {
  if (r.train_accuracy === null || r.train_accuracy === undefined ||
      r.test_accuracy === null || r.test_accuracy === undefined) return null;
  return r.train_accuracy - r.test_accuracy;
}

// Flags a run that collapsed to predicting a single class throughout: recall=0/specificity=1
// (always predicts "True"/real) or recall=1/specificity=0 (always predicts "Fake").
// Returns the class it always predicts, or null if the run looks normal / metrics are missing.
function detectDegenerate(r) {
  const t = (r.metrics && r.metrics.test) || {};
  if (t.recall === undefined || t.recall === null || t.specificity === undefined || t.specificity === null) return null;
  if (t.recall <= DEGENERATE_EPS && t.specificity >= 1 - DEGENERATE_EPS) return 'True';
  if (t.recall >= 1 - DEGENERATE_EPS && t.specificity <= DEGENERATE_EPS) return 'Fake';
  return null;
}

// Per-class precision/recall/F1, derived entirely from the confusion matrix
// (tn/fp/fn/tp) already present in every record -- no new pipeline field needed.
function classBreakdown(t) {
  if (t.tp === undefined || t.tp === null) return null;
  const { tp, tn, fp, fn } = t;
  const safeDiv = (a, b) => b > 0 ? a / b : 0;
  const f1 = (p, r) => (p + r) > 0 ? 2 * p * r / (p + r) : 0;
  const fakeP = safeDiv(tp, tp + fp), fakeR = safeDiv(tp, tp + fn);
  const trueP = safeDiv(tn, tn + fn), trueR = safeDiv(tn, tn + fp);
  return {
    Fake: { precision: fakeP, recall: fakeR, f1: f1(fakeP, fakeR), support: tp + fn },
    True: { precision: trueP, recall: trueR, f1: f1(trueP, trueR), support: tn + fp },
  };
}

function mean(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null; }
function std(arr, m) {
  if (arr.length < 2) return 0;
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}

// ---------- Narrative summary ----------
function renderNarrative() {
  const container = document.getElementById('narrative-list');
  if (!runs.length) { container.innerHTML = '<li>No hay corridas para resumir.</li>'; return; }

  const items = [];

  const byF1 = runs.filter(r => r.test_f1 !== null && r.test_f1 !== undefined).sort((a, b) => b.test_f1 - a.test_f1);
  if (byF1.length) {
    const r = byF1[0];
    items.push(`Mejor corrida por F1 de test: <strong>${r.batch_label || r.run_id}</strong> (${r.combo}) — F1=${fmtNum(r.test_f1)}, accuracy=${fmtNum(r.test_accuracy)}.`);
  }

  const byEce = runs.filter(r => r.test_ece !== null && r.test_ece !== undefined).sort((a, b) => a.test_ece - b.test_ece);
  if (byEce.length) {
    const r = byEce[0];
    items.push(`Mejor calibración (ECE más bajo): <strong>${r.batch_label || r.run_id}</strong> (${r.combo}) — ECE=${fmtNum(r.test_ece)}.`);
  }

  const byCombo = {};
  runs.forEach(r => { (byCombo[r.combo] = byCombo[r.combo] || []).push(r); });
  const comboStats = Object.entries(byCombo).map(([combo, rs]) => {
    const accs = rs.map(r => r.test_accuracy).filter(v => v !== null && v !== undefined);
    return { combo, mean: mean(accs), nExtractors: rs[0].active_extractors.length };
  }).filter(c => c.mean !== null);
  if (comboStats.length) {
    const maxAcc = Math.max(...comboStats.map(c => c.mean));
    const withinRange = comboStats.filter(c => maxAcc - c.mean <= 0.02).sort((a, b) => a.nExtractors - b.nExtractors);
    const winner = withinRange[0];
    items.push(`Combinación más eficiente (accuracy media dentro de 2pts del máximo, ${fmtNum(maxAcc)}): <strong>${winner.combo}</strong> (${winner.nExtractors} extractor${winner.nExtractors === 1 ? '' : 'es'}, accuracy media=${fmtNum(winner.mean)}).`);
  }

  const degenerateRuns = runs.map(r => ({ r, tag: detectDegenerate(r) })).filter(x => x.tag);
  if (degenerateRuns.length) {
    const list = degenerateRuns.map(x => `${x.r.batch_label || x.r.run_id} (predice siempre "${x.tag}")`).join(', ');
    items.push(`<span class="warn-item">⚠ ${degenerateRuns.length} corrida(s) colapsada(s) al baseline: ${list}.</span>`);
  } else {
    items.push('Ninguna corrida colapsó a predecir siempre la misma clase.');
  }

  container.innerHTML = items.map(i => `<li>${i}</li>`).join('');
}

// ---------- Summary table ----------
let gapThreshold = 0.15;
const TABLE_COLUMNS = [
  { key: 'run_id', label: 'run_id', sort: (r) => r.run_id },
  { key: 'timestamp', label: 'Fecha/hora', sort: (r) => r.timestamp },
  { key: 'combo', label: 'Extractores activos', sort: (r) => r.combo },
  { key: 'epochs', label: 'Épocas KAN (run/pedidas)', sort: (r) => r.epochs.kan_epochs_run },
  { key: 'test_accuracy', label: 'Test accuracy', sort: (r) => r.test_accuracy },
  { key: 'gap', label: 'Gap (train−test)', sort: (r) => overfitGap(r) },
  { key: 'test_f1', label: 'Test F1', sort: (r) => r.test_f1 },
  { key: 'test_roc_auc', label: 'Test ROC-AUC', sort: (r) => r.test_roc_auc },
  { key: 'seed', label: 'Seed', sort: (r) => r.seed },
  { key: 'training_time', label: 'Tiempo entren. (s)', sort: (r) => r.training_time_seconds },
  { key: 'num_parameters', label: '# Parámetros', sort: (r) => r.num_parameters },
  { key: 'batch', label: 'Lote', sort: (r) => r.batch_label || '' },
];

let sortState = { key: 'timestamp', asc: true };
let filterText = '';

function filteredSortedRuns() {
  let rows = runs;
  if (filterText) {
    const q = filterText.toLowerCase();
    rows = rows.filter(r =>
      r.run_id.toLowerCase().includes(q) ||
      r.combo.toLowerCase().includes(q) ||
      (r.batch_label || '').toLowerCase().includes(q) ||
      (r.batch_phase || '').toLowerCase().includes(q)
    );
  }
  const col = TABLE_COLUMNS.find(c => c.key === sortState.key);
  rows = [...rows].sort((a, b) => {
    let va = col.sort(a), vb = col.sort(b);
    if (va === null || va === undefined) va = -Infinity;
    if (vb === null || vb === undefined) vb = -Infinity;
    if (va < vb) return sortState.asc ? -1 : 1;
    if (va > vb) return sortState.asc ? 1 : -1;
    return 0;
  });
  return rows;
}

function renderTable() {
  const theadRow = document.getElementById('table-head-row');
  theadRow.innerHTML = TABLE_COLUMNS.map(c => {
    const arrow = sortState.key === c.key ? (sortState.asc ? '▲' : '▼') : '';
    return `<th data-key="${c.key}">${c.label}<span class="arrow">${arrow}</span></th>`;
  }).join('');
  theadRow.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      if (sortState.key === key) sortState.asc = !sortState.asc;
      else { sortState.key = key; sortState.asc = true; }
      renderTable();
    });
  });

  const rows = filteredSortedRuns();
  const tbody = document.getElementById('table-body');
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="${TABLE_COLUMNS.length}" class="empty-msg">Sin corridas que coincidan con el filtro.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const early = r.epochs.kan_epochs_run < r.epochs.kan_epochs_requested;
    const epochCell = `<span class="${early ? 'epoch-early' : ''}">${r.epochs.kan_epochs_run} / ${r.epochs.kan_epochs_requested}</span>`;
    const batchCell = r.batch_label ? `${r.batch_label} <span style="color:var(--text-dim)">(${r.batch_phase || ''})</span>` : '—';
    const gap = overfitGap(r);
    const gapCell = gap === null ? '—' : `<span class="${Math.abs(gap) >= gapThreshold ? 'gap-warn' : ''}">${gap.toFixed(4)}</span>`;
    const degenerate = detectDegenerate(r);
    const rowClass = degenerate ? 'degenerate-row' : '';
    const degenerateTag = degenerate ? `<span class="degenerate-tag">⚠ colapsó (predice siempre "${degenerate}")</span>` : '';
    return `<tr class="${rowClass}">
      <td><code>${r.run_id}</code>${degenerateTag}</td>
      <td>${fmtTs(r.timestamp)}</td>
      <td>${extractorBadges(r.active_extractors)}</td>
      <td>${epochCell}</td>
      <td>${fmtNum(r.test_accuracy)}</td>
      <td>${gapCell}</td>
      <td>${fmtNum(r.test_f1)}</td>
      <td>${fmtNum(r.test_roc_auc)}</td>
      <td>${r.seed ?? '—'}</td>
      <td>${r.training_time_seconds !== null && r.training_time_seconds !== undefined ? r.training_time_seconds.toFixed(1) : '—'}</td>
      <td>${r.num_parameters !== null && r.num_parameters !== undefined ? r.num_parameters.toLocaleString() : '—'}</td>
      <td>${batchCell}</td>
    </tr>`;
  }).join('');
}

document.getElementById('table-filter').addEventListener('input', (e) => {
  filterText = e.target.value;
  renderTable();
});
document.getElementById('gap-threshold').addEventListener('input', (e) => {
  const v = parseFloat(e.target.value);
  if (!Number.isNaN(v)) { gapThreshold = v; renderTable(); }
});

// ---------- b) Grouped bar: train/val/test accuracy per run ----------
function renderBarChart() {
  const labels = runs.map(r => r.batch_label ? `${r.batch_label}` : r.run_id.slice(-8));
  const mk = (split) => runs.map(r => (r.metrics[split] || {}).accuracy ?? null);
  const traces = ['train', 'val', 'test'].map(split => ({
    x: labels, y: mk(split), name: split, type: 'bar',
  }));
  Plotly.newPlot('bar-chart', traces, {
    barmode: 'group',
    title: 'Accuracy por split (train / val / test)',
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#e6e9f2' },
    xaxis: { tickangle: -35 },
    margin: { t: 40, b: 90 },
  }, { responsive: true, displaylogo: false });
}

// ---------- c) Heatmap: extractor combo vs test metrics (mean±std when a combo has >1 run) ----------
function renderHeatmap() {
  const metrics = ['test_accuracy', 'test_f1', 'test_roc_auc', 'test_log_loss'];
  const metricLabels = ['accuracy', 'f1', 'roc_auc', 'log_loss'];
  const byCombo = {};
  runs.forEach(r => {
    if (!byCombo[r.combo]) byCombo[r.combo] = [];
    byCombo[r.combo].push(r);
  });
  const combos = Object.keys(byCombo).sort((a, b) => {
    const meanA = byCombo[a].reduce((s, r) => s + (r.test_f1 || 0), 0) / byCombo[a].length;
    const meanB = byCombo[b].reduce((s, r) => s + (r.test_f1 || 0), 0) / byCombo[b].length;
    return meanA - meanB;
  });
  const z = combos.map(combo => metrics.map(m => {
    const vals = byCombo[combo].map(r => r[m]).filter(v => v !== null && v !== undefined);
    return vals.length ? mean(vals) : null;
  }));
  const text = combos.map(combo => metrics.map(m => {
    const vals = byCombo[combo].map(r => r[m]).filter(v => v !== null && v !== undefined);
    if (!vals.length) return '';
    const m_ = mean(vals);
    return vals.length > 1 ? `${m_.toFixed(3)}±${std(vals, m_).toFixed(3)}` : m_.toFixed(3);
  }));

  Plotly.newPlot('heatmap-chart', [{
    z, x: metricLabels, y: combos, type: 'heatmap', colorscale: 'Viridis',
    text, texttemplate: '%{text}', hovertemplate: '%{y} · %{x}: %{z:.4f}<extra></extra>',
  }], {
    title: 'Combinaciones de extractores vs. métricas de test (media±std si hay varias corridas)',
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#e6e9f2' },
    margin: { t: 40, l: 220 },
  }, { responsive: true, displaylogo: false });
}

// ---------- Seed stability: runs sharing combo + hyperparams (excl. seed), differing only by seed ----------
function comboConfigKey(r) {
  const kh = r.kan_hyperparams || {};
  const khNoSeed = Object.fromEntries(Object.entries(kh).filter(([k]) => k !== 'seed'));
  return JSON.stringify({ combo: r.combo, kh: khNoSeed, latent: r.latent_dims, epochsReq: r.epochs.kan_epochs_requested });
}

function renderSeedStability() {
  const groups = {};
  runs.forEach(r => {
    const key = comboConfigKey(r);
    (groups[key] = groups[key] || []).push(r);
  });
  const multiSeedGroups = Object.values(groups).filter(g => new Set(g.map(r => r.seed)).size >= 2);

  const section = document.getElementById('seed-stability-section');
  if (!multiSeedGroups.length) { section.style.display = 'none'; return; }
  section.style.display = '';

  document.getElementById('seed-stability-body').innerHTML = multiSeedGroups.map(g => {
    const accs = g.map(r => r.test_accuracy).filter(v => v !== null && v !== undefined);
    const f1s = g.map(r => r.test_f1).filter(v => v !== null && v !== undefined);
    const accMean = mean(accs), f1Mean = mean(f1s);
    const seeds = g.map(r => r.seed).join(', ');
    return `<tr>
      <td>${g[0].combo}</td>
      <td>${g.length}</td>
      <td>${seeds}</td>
      <td>${accMean.toFixed(4)} ± ${std(accs, accMean).toFixed(4)}</td>
      <td>${f1Mean.toFixed(4)} ± ${std(f1s, f1Mean).toFixed(4)}</td>
    </tr>`;
  }).join('');
}

// ---------- d) Calibration scatter: ECE / Brier vs accuracy ----------
function renderCalibrationScatters() {
  const x = runs.map(r => r.test_accuracy);
  const hover = runs.map(r => `${r.run_id}<br>${r.combo}`);

  Plotly.newPlot('scatter-ece', [{
    x, y: runs.map(r => r.test_ece), text: hover, mode: 'markers',
    marker: { size: 10, color: '#6ea8fe' }, hovertemplate: '%{text}<br>accuracy=%{x:.4f}<br>ece=%{y:.4f}<extra></extra>',
  }], {
    title: 'ECE (test) vs. Accuracy (test)',
    xaxis: { title: 'test accuracy' }, yaxis: { title: 'ECE' },
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#e6e9f2' },
    margin: { t: 40 },
  }, { responsive: true, displaylogo: false });

  Plotly.newPlot('scatter-brier', [{
    x, y: runs.map(r => r.test_brier_score), text: hover, mode: 'markers',
    marker: { size: 10, color: '#ffb454' }, hovertemplate: '%{text}<br>accuracy=%{x:.4f}<br>brier=%{y:.4f}<extra></extra>',
  }], {
    title: 'Brier score (test) vs. Accuracy (test)',
    xaxis: { title: 'test accuracy' }, yaxis: { title: 'Brier score' },
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#e6e9f2' },
    margin: { t: 40 },
  }, { responsive: true, displaylogo: false });
}

// ---------- e) Per-run collapsible cards ----------
function hparamsBlock(title, obj) {
  const entries = Object.entries(obj || {});
  if (!entries.length) return `<div><h4>${title}</h4><p style="color:var(--text-dim)">—</p></div>`;
  const dl = entries.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  return `<div><h4>${title}</h4><dl>${dl}</dl></div>`;
}

function classBreakdownTable(t) {
  const cb = classBreakdown(t);
  if (!cb) return '<p style="color:var(--text-dim)">—</p>';
  const rows = ['Fake', 'True'].map(cls => {
    const m = cb[cls];
    return `<tr><td>${cls}</td><td>${m.precision.toFixed(4)}</td><td>${m.recall.toFixed(4)}</td><td>${m.f1.toFixed(4)}</td><td>${m.support}</td></tr>`;
  }).join('');
  return `<table class="mini"><tr><th>Clase</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>${rows}</table>`;
}

function topicBreakdownTable(tb) {
  if (!tb) return '<p style="color:var(--text-dim)">No disponible para esta corrida (requiere el campo topic_breakdown, agregado en corridas recientes).</p>';
  const rows = Object.entries(tb).sort((a, b) => b[1].n - a[1].n).map(([topic, m]) =>
    `<tr><td>${topic}</td><td>${m.n}</td><td>${m.accuracy.toFixed(4)}</td><td>${m.f1.toFixed(4)}</td></tr>`
  ).join('');
  return `<table class="mini"><tr><th>Topic</th><th>n</th><th>Accuracy</th><th>F1</th></tr>${rows}</table>`;
}

function renderCards() {
  const container = document.getElementById('run-cards');
  container.innerHTML = runs.map(r => {
    const t = r.metrics.test || {};
    const label = r.batch_label ? `${r.batch_label} (${r.batch_phase || ''})` : '';
    const degenerate = detectDegenerate(r);
    const gap = overfitGap(r);
    const banner = degenerate
      ? `<div class="degenerate-banner">⚠ Esta corrida colapsó al baseline: predice siempre "${degenerate}" (recall=${fmtNum(t.recall)}, specificity=${fmtNum(t.specificity)}).</div>`
      : '';
    return `<details class="run-card">
      <summary>
        <span class="run-id">${r.run_id}</span>
        ${extractorBadges(r.active_extractors)}
        <span class="metric">${label}</span>
        <span class="metric">acc=${fmtNum(t.accuracy)} · f1=${fmtNum(t.f1)} · roc_auc=${fmtNum(t.roc_auc)}${gap !== null ? ` · gap=${gap.toFixed(4)}` : ''}</span>
        ${degenerate ? `<span class="degenerate-tag">⚠ colapsó ("${degenerate}")</span>` : ''}
      </summary>
      <div class="card-body">
        <div>
          ${banner}
          <h4 style="color:var(--text-dim); font-size:0.82rem; text-transform:uppercase; margin:0 0 6px;">Matriz de confusión (test)</h4>
          <table class="confusion">
            <tr><th></th><th>Pred: True</th><th>Pred: Fake</th></tr>
            <tr><th>Real: True</th><td class="tn">${t.tn ?? '—'}</td><td class="fp">${t.fp ?? '—'}</td></tr>
            <tr><th>Real: Fake</th><td class="fn">${t.fn ?? '—'}</td><td class="tp">${t.tp ?? '—'}</td></tr>
          </table>
          <p style="color:var(--text-dim); font-size:0.8rem; margin-top:10px;">
            git commit: <code>${(r.git_commit || '').slice(0, 10) || '—'}</code><br>
            dataset hash: <code>${r.dataset_hash ? r.dataset_hash.slice(0, 12) : '—'}</code><br>
            seed: ${r.seed ?? '—'} &nbsp;·&nbsp;
            tiempo entren.: ${r.training_time_seconds !== null && r.training_time_seconds !== undefined ? r.training_time_seconds.toFixed(1) + 's' : '—'} &nbsp;·&nbsp;
            # parámetros: ${r.num_parameters !== null && r.num_parameters !== undefined ? r.num_parameters.toLocaleString() : '—'}<br>
            latent dims: ${Object.entries(r.latent_dims || {}).map(([k, v]) => `${k}=${v}`).join(', ') || '—'}
          </p>
        </div>
        <div class="hparams-cols">
          ${hparamsBlock('Épocas', r.epochs)}
          ${hparamsBlock('VAE hyperparams', r.vae_hyperparams)}
          ${hparamsBlock('KAN hyperparams', r.kan_hyperparams)}
        </div>
      </div>
      <div class="section-block">
        <h4>Desglose por clase (test)</h4>
        ${classBreakdownTable(t)}
      </div>
      <div class="section-block">
        <h4>Accuracy / F1 por Topic (test)</h4>
        ${topicBreakdownTable(r.topic_breakdown)}
      </div>
    </details>`;
  }).join('') || '<p class="empty-msg">No hay corridas para mostrar.</p>';
}

renderNarrative();
renderTable();
if (runs.length) {
  renderBarChart();
  renderHeatmap();
  renderSeedStability();
  renderCalibrationScatters();
}
renderCards();
"""


def render_html(records: List[Dict[str, Any]], scope_kind: str, scope_desc: str, generated_at: str) -> str:
    flat_runs = [flatten_record(r) for r in records]
    flat_runs.sort(key=lambda r: r["timestamp"])

    payload = {"runs": flat_runs, "scope_kind": scope_kind, "scope_desc": scope_desc, "generated_at": generated_at}

    dataset_hashes = {r["dataset_hash"] for r in flat_runs if r.get("dataset_hash")}
    if len(dataset_hashes) == 1:
        dataset_hash_note = f'<span>Dataset hash: <code>{next(iter(dataset_hashes))[:12]}…</code></span>'
    elif len(dataset_hashes) > 1:
        dataset_hash_note = f'<span style="color:var(--warn)">⚠ {len(dataset_hashes)} versiones de dataset distintas en este reporte</span>'
    else:
        dataset_hash_note = ""

    header_html = f"""
    <header class="report-header">
      <h1>Reporte comparativo de experimentos — FakeNewsStyle</h1>
      <div class="meta">
        <span>Generado: {generated_at}</span>
        <span>Corridas incluidas: {len(flat_runs)}</span>
        <span>Alcance: {scope_desc}</span>
        {dataset_hash_note}
      </div>
    </header>
    """

    table_cols_header = """<th>run_id</th><th>Fecha/hora</th><th>Extractores activos</th>
      <th>Épocas KAN (run/pedidas)</th><th>Test accuracy</th><th>Gap (train−test)</th><th>Test F1</th>
      <th>Test ROC-AUC</th><th>Seed</th><th>Tiempo entren. (s)</th><th># Parámetros</th><th>Lote</th>"""

    body = f"""
{header_html}

<section class="narrative">
  <h3>Resumen automático</h3>
  <ul id="narrative-list"></ul>
</section>

<section id="summary">
  <h2>Tabla resumen</h2>
  <div class="controls">
    <input type="text" id="table-filter" placeholder="Filtrar por run_id, extractores o lote...">
    <label>Umbral de alerta de gap (train−test): <input type="number" id="gap-threshold" value="0.15" step="0.01" min="0" max="1"></label>
  </div>
  <div style="overflow-x:auto;">
    <table class="summary">
      <thead><tr id="table-head-row">{table_cols_header}</tr></thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>
</section>

<section id="charts-a">
  <h2>Overfitting: accuracy por split</h2>
  <div class="chart-grid full"><div class="chart-box"><div id="bar-chart" style="height:420px;"></div></div></div>
</section>

<section id="charts-b">
  <h2>Combinaciones de extractores vs. métricas de test</h2>
  <div class="chart-grid full"><div class="chart-box"><div id="heatmap-chart" style="height:460px;"></div></div></div>
  <div id="seed-stability-section" class="section-block seed-stability-table" style="display:none;">
    <h4>Estabilidad entre semillas (mismo combo + hiperparámetros, distinta seed)</h4>
    <table class="mini">
      <thead><tr><th>Combinación</th><th># corridas</th><th>Seeds</th><th>Accuracy (media ± std)</th><th>F1 (media ± std)</th></tr></thead>
      <tbody id="seed-stability-body"></tbody>
    </table>
  </div>
</section>

<section id="charts-c">
  <h2>Calibración: ECE / Brier score vs. accuracy (test)</h2>
  <div class="chart-grid">
    <div class="chart-box"><div id="scatter-ece" style="height:400px;"></div></div>
    <div class="chart-box"><div id="scatter-brier" style="height:400px;"></div></div>
  </div>
</section>

<section id="cards">
  <h2>Detalle por corrida</h2>
  <div id="run-cards"></div>
</section>

<footer>FakeNewsStyle — reporte generado por scripts/html_report_builder.py</footer>

<script id="report-data" type="application/json">{json.dumps(payload, ensure_ascii=False)}</script>
<script src="{PLOTLY_CDN}"></script>
<script>{JS}</script>
"""

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte comparativo — FakeNewsStyle</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


# =====================================================
# CLI
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Build a self-contained HTML comparison report for FakeNewsStyle experiments")
    parser.add_argument("--all", action="store_true", help="Include every run in results/ (loose + all batches)")
    parser.add_argument("--batch-id", type=str, default=None, help="Scope the report to results/batches/<id>.json")
    args = parser.parse_args()

    scope_kind, scope_arg = resolve_scope(args)

    if scope_kind == "all":
        records, scope_desc = load_all_scope()
        batch_id = None
    else:
        records, scope_desc = load_batch_scope(scope_arg)
        batch_id = load_manifest(scope_arg)["batch_id"]

    if not records:
        print("No se encontraron corridas para el alcance elegido.")
        return

    df = build_dataframe(records)
    print(f"\nCargadas {len(df)} corridas. Alcance: {scope_desc}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if batch_id:
        out_path = REPORTS_DIR / f"comparison_report_{batch_id}_{date_str}.html"
    else:
        out_path = REPORTS_DIR / f"comparison_report_{date_str}.html"

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    html = render_html(records, scope_kind, scope_desc, generated_at)
    out_path.write_text(html, encoding="utf-8")

    print(f"Reporte HTML guardado: {out_path}")


if __name__ == "__main__":
    main()
