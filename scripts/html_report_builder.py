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
    epochs = record.get("epochs", {})

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
        "kan_hyperparams": record.get("kan_hyperparams", {}),
        "metrics": metrics,
        "paths": record.get("paths", {}),
        "batch_id": record.get("_batch_id"),
        "batch_label": record.get("_batch_label"),
        "batch_phase": record.get("_batch_phase"),
        "test_accuracy": test.get("accuracy"),
        "test_f1": test.get("f1"),
        "test_roc_auc": test.get("roc_auc"),
        "test_log_loss": test.get("log_loss"),
        "test_ece": test.get("ece"),
        "test_brier_score": test.get("brier_score"),
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
.controls { display: flex; gap: 12px; align-items: center; margin: 12px 0; }
.controls input[type="text"] {
  background: var(--panel-alt); border: 1px solid var(--border); color: var(--text);
  padding: 7px 12px; border-radius: 6px; width: 320px; font-size: 0.9rem;
}
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

// ---------- Summary table ----------
const TABLE_COLUMNS = [
  { key: 'run_id', label: 'run_id', sort: (r) => r.run_id },
  { key: 'timestamp', label: 'Fecha/hora', sort: (r) => r.timestamp },
  { key: 'combo', label: 'Extractores activos', sort: (r) => r.combo },
  { key: 'epochs', label: 'Épocas KAN (run/pedidas)', sort: (r) => r.epochs.kan_epochs_run },
  { key: 'test_accuracy', label: 'Test accuracy', sort: (r) => r.test_accuracy },
  { key: 'test_f1', label: 'Test F1', sort: (r) => r.test_f1 },
  { key: 'test_roc_auc', label: 'Test ROC-AUC', sort: (r) => r.test_roc_auc },
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
    return `<tr>
      <td><code>${r.run_id}</code></td>
      <td>${fmtTs(r.timestamp)}</td>
      <td>${extractorBadges(r.active_extractors)}</td>
      <td>${epochCell}</td>
      <td>${fmtNum(r.test_accuracy)}</td>
      <td>${fmtNum(r.test_f1)}</td>
      <td>${fmtNum(r.test_roc_auc)}</td>
      <td>${batchCell}</td>
    </tr>`;
  }).join('');
}

document.getElementById('table-filter').addEventListener('input', (e) => {
  filterText = e.target.value;
  renderTable();
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

// ---------- c) Heatmap: extractor combo vs test metrics ----------
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
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  }));
  const text = z.map(row => row.map(v => v === null ? '' : v.toFixed(3)));

  Plotly.newPlot('heatmap-chart', [{
    z, x: metricLabels, y: combos, type: 'heatmap', colorscale: 'Viridis',
    text, texttemplate: '%{text}', hovertemplate: '%{y} · %{x}: %{z:.4f}<extra></extra>',
  }], {
    title: 'Combinaciones de extractores vs. métricas de test (media si hay varias corridas)',
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#e6e9f2' },
    margin: { t: 40, l: 220 },
  }, { responsive: true, displaylogo: false });
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

function renderCards() {
  const container = document.getElementById('run-cards');
  container.innerHTML = runs.map(r => {
    const t = r.metrics.test || {};
    const label = r.batch_label ? `${r.batch_label} (${r.batch_phase || ''})` : '';
    return `<details class="run-card">
      <summary>
        <span class="run-id">${r.run_id}</span>
        ${extractorBadges(r.active_extractors)}
        <span class="metric">${label}</span>
        <span class="metric">acc=${fmtNum(t.accuracy)} · f1=${fmtNum(t.f1)} · roc_auc=${fmtNum(t.roc_auc)}</span>
      </summary>
      <div class="card-body">
        <div>
          <h4 style="color:var(--text-dim); font-size:0.82rem; text-transform:uppercase; margin:0 0 6px;">Matriz de confusión (test)</h4>
          <table class="confusion">
            <tr><th></th><th>Pred: True</th><th>Pred: Fake</th></tr>
            <tr><th>Real: True</th><td class="tn">${t.tn ?? '—'}</td><td class="fp">${t.fp ?? '—'}</td></tr>
            <tr><th>Real: Fake</th><td class="fn">${t.fn ?? '—'}</td><td class="tp">${t.tp ?? '—'}</td></tr>
          </table>
          <p style="color:var(--text-dim); font-size:0.8rem; margin-top:10px;">
            git commit: <code>${(r.git_commit || '').slice(0, 10) || '—'}</code><br>
            latent dims: ${Object.entries(r.latent_dims || {}).map(([k, v]) => `${k}=${v}`).join(', ') || '—'}
          </p>
        </div>
        <div class="hparams-cols">
          ${hparamsBlock('Épocas', r.epochs)}
          ${hparamsBlock('VAE hyperparams', r.vae_hyperparams)}
          ${hparamsBlock('KAN hyperparams', r.kan_hyperparams)}
        </div>
      </div>
    </details>`;
  }).join('') || '<p class="empty-msg">No hay corridas para mostrar.</p>';
}

renderTable();
if (runs.length) {
  renderBarChart();
  renderHeatmap();
  renderCalibrationScatters();
}
renderCards();
"""


def render_html(records: List[Dict[str, Any]], scope_kind: str, scope_desc: str, generated_at: str) -> str:
    flat_runs = [flatten_record(r) for r in records]
    flat_runs.sort(key=lambda r: r["timestamp"])

    payload = {"runs": flat_runs, "scope_kind": scope_kind, "scope_desc": scope_desc, "generated_at": generated_at}

    header_html = f"""
    <header class="report-header">
      <h1>Reporte comparativo de experimentos — FakeNewsStyle</h1>
      <div class="meta">
        <span>Generado: {generated_at}</span>
        <span>Corridas incluidas: {len(flat_runs)}</span>
        <span>Alcance: {scope_desc}</span>
      </div>
    </header>
    """

    table_cols_header = """<th>run_id</th><th>Fecha/hora</th><th>Extractores activos</th>
      <th>Épocas KAN (run/pedidas)</th><th>Test accuracy</th><th>Test F1</th>
      <th>Test ROC-AUC</th><th>Lote</th>"""

    body = f"""
{header_html}

<section id="summary">
  <h2>Tabla resumen</h2>
  <div class="controls">
    <input type="text" id="table-filter" placeholder="Filtrar por run_id, extractores o lote...">
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
