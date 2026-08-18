# scripts/orchestrator_phase2.py
# -*- coding: utf-8 -*-
"""
Fase 2: para cada una de las 3 configuraciones ganadoras de la Fase 1, corre
en orden fijo 5 grupos de sweep secuenciales -- cada grupo se agrega con
aggregate_results antes de generar las corridas del siguiente, así que el
plan completo NO se conoce de antemano (se decide en runtime):

  (a)  espacio latente      (--{branch}_latent_dim, solo ramas activas)
  (a2) regularización VAE   (--vae_beta / --vae_dropout -- ver DEFAULT_VAE_REG
                              en experiment_config.py; necesario porque una
                              corrida histórica con beta bajo midió más alto
                              que cualquier combo de la Fase 1)
  (b)  "entradas" del KAN   (--kan_num_basis -- no existe un flag de input_dim
                              separado; el input real del KAN es la suma de
                              las dims latentes activas, ya cubierta por (a))
  (c)  capas/nodos internos (--kan_hidden_dim)
  (d)  parámetros de entrenamiento (--kan_lr / --kan_batch_size / --kan_weight_decay)

Cada valor candidato corre con las 10 semillas fijas de experiment_config.SEEDS.
Checkpointing/resume idéntico a la Fase 1, sobre experiment_config.PHASE2_RESULTS_JSONL.

Nota sobre (a2): --merge_vae_latents de main.py siempre lee de la ruta fija
data/05_vae_latents/{branch}/latent{dim}/ (hardcoded, no respeta
--vae_data_output_dir), así que cualquier candidato con beta/dropout distinto
al default de main.py (ver DEFAULT_VAE_REG) se entrena en un directorio
aislado (PHASE2_VAE_DATA_DIR) y se mergea manualmente en Python (mismo
patrón que scripts/run_full_stack_sweep.py), apuntando --train_kan a esos
PKLs vía --kan_train_pkl/--kan_val_pkl/--kan_test_pkl. Los candidatos que
coinciden con el default siguen reusando el cache compartido normal.

Uso:
    python scripts/orchestrator_phase2.py                       # usa results/phase1_top3.json
    python scripts/orchestrator_phase2.py --winners path.json
    python scripts/orchestrator_phase2.py --configs '[["semantic","emotion"], ["semantic","style","context"]]'
    python scripts/orchestrator_phase2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_results import aggregate_by_config, load_runs, pairwise_wilcoxon, report  # noqa: E402
from experiment_config import (  # noqa: E402
    ALL_MODALITIES,
    BASE_DIR,
    DEFAULT_LATENT_DIM,
    DEFAULT_VAE_REG,
    FIXED_KAN_BASELINE,
    KAN_HIDDEN_DIM_CANDIDATES,
    KAN_NUM_BASIS_CANDIDATES,
    KAN_RUNS_DIR,
    KAN_TRAINING_CANDIDATES,
    LATENT_DIM_CANDIDATES,
    PHASE1_WINNERS_JSON,
    PHASE2_RESULTS_JSONL,
    PHASE2_VAE_DATA_DIR,
    PHASE2_VAE_MERGED_DIR,
    PHASE2_VAE_MODEL_DIR,
    RANKING_METRIC,
    SEEDS,
    VAE_LATENTS_DIR,
    VAE_REG_CANDIDATES,
)
from experiment_runner import (  # noqa: E402
    execute_and_log,
    load_ok_run_keys,
    python_executable,
    run_main_command,
)

DEFAULT_KAN_ARCH = {"kan_num_basis": 16, "kan_hidden_dim": 64}  # main.py's own defaults


def config_label(combo: List[str]) -> str:
    return "_".join(m for m in ALL_MODALITIES if m in combo)


def initial_resolved() -> Dict[str, Any]:
    resolved = {"latent": dict(DEFAULT_LATENT_DIM)}
    resolved.update(DEFAULT_KAN_ARCH)
    resolved.update(FIXED_KAN_BASELINE)
    resolved.update(DEFAULT_VAE_REG)
    return resolved


def is_default_vae_reg(effective: Dict[str, Any]) -> bool:
    return (
        effective["vae_beta"] == DEFAULT_VAE_REG["vae_beta"]
        and effective["vae_dropout"] == DEFAULT_VAE_REG["vae_dropout"]
    )


def ensure_vae_latents(active_extractors: List[str], dims: Dict[str, int], dry_run: bool) -> None:
    """Default-beta/dropout path: reuses the shared cache at
    VAE_LATENTS_DIR, training only what's missing for this latent preset."""
    missing = []
    for branch in active_extractors:
        branch_dir = VAE_LATENTS_DIR / branch / f"latent{dims[branch]}"
        for split in ["train", "val", "test"]:
            if not (branch_dir / f"{split}.pkl").exists():
                missing.append(f"{branch} (latent{dims[branch]}, {split}.pkl)")

    if not missing:
        return

    print(f"    VAE latentes faltantes para este preset: {missing}")
    cmd = [python_executable(), "main.py", "--run_vaes"]
    for branch in ALL_MODALITIES:
        if branch not in active_extractors:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(dims.get(branch, DEFAULT_LATENT_DIM[branch]))]
    print(f"    $ {' '.join(cmd)}")

    if dry_run:
        print("    (dry-run: no se ejecuta)")
        return

    outcome = run_main_command(cmd)
    if outcome["error"] is not None:
        raise RuntimeError(f"Fallo entrenando VAE para {active_extractors} @ {dims}: {outcome['error']}")
    print(f"    OK en {outcome['elapsed_seconds']}s")


def merge_latents_manual(
    active_extractors: List[str], latent_dirs: Dict[str, Path], out_dir: Path
) -> Dict[str, Path]:
    """Mirrors main.py's Step 9 (--merge_vae_latents) column-prefixing logic
    (same {branch}_ prefix convention, same label handling), but reads from
    arbitrary latent_dirs instead of the hardcoded data/05_vae_latents/ path
    -- needed for VAE beta/dropout candidates that main.py can't merge on
    its own. Same pattern as scripts/run_full_stack_sweep.py."""
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
    combo: List[str], config_label_: str, group_name: str, cand_label: str,
    effective: Dict[str, Any], dry_run: bool,
) -> Optional[Dict[str, Path]]:
    """Decides how a candidate's KAN input is produced. Returns None when
    (vae_beta, vae_dropout) match main.py's defaults -- callers should then
    pass --merge_vae_latents and let main.py read the shared cache (after
    ensure_vae_latents makes sure this latent preset exists in it). Returns
    a {split: path} dict when they don't -- an isolated VAE was trained (if
    missing) and manually merged, and callers should point --train_kan at
    those paths directly instead of --merge_vae_latents."""

    if is_default_vae_reg(effective):
        ensure_vae_latents(combo, effective["latent"], dry_run=dry_run)
        return None

    tag = f"beta{effective['vae_beta']}_drop{effective['vae_dropout']}"
    vae_data_dir = PHASE2_VAE_DATA_DIR / config_label_ / tag
    vae_model_dir = PHASE2_VAE_MODEL_DIR / config_label_ / tag
    merged_dir = PHASE2_VAE_MERGED_DIR / config_label_ / f"{group_name}_{cand_label}"

    latent_dirs = {branch: vae_data_dir / branch / f"latent{effective['latent'][branch]}" for branch in combo}
    missing = [
        branch for branch in combo
        if not all((latent_dirs[branch] / f"{s}.pkl").exists() for s in ["train", "val", "test"])
    ]

    if missing:
        print(f"    VAE aislado faltante (beta={effective['vae_beta']}, dropout={effective['vae_dropout']}): {missing}")
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
            print("    (dry-run: no se ejecuta)")
        else:
            outcome = run_main_command(cmd)
            if outcome["error"] is not None:
                raise RuntimeError(
                    f"Fallo entrenando VAE aislado para {combo} "
                    f"@ beta={effective['vae_beta']} dropout={effective['vae_dropout']}: {outcome['error']}"
                )
            print(f"    OK en {outcome['elapsed_seconds']}s")

    if dry_run:
        return {s: merged_dir / f"{s}.pkl" for s in ["train", "val", "test"]}

    return merge_latents_manual(combo, latent_dirs, merged_dir)


def build_kan_command(
    combo: List[str], effective: Dict[str, Any], seed: int, output_dir: Path,
    kan_pkl_paths: Optional[Dict[str, Path]] = None,
) -> List[str]:
    cmd = [python_executable(), "main.py"]
    if kan_pkl_paths is None:
        cmd.append("--merge_vae_latents")
    cmd.append("--train_kan")

    for branch in ALL_MODALITIES:
        if branch not in combo:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(effective["latent"].get(branch, DEFAULT_LATENT_DIM[branch]))]

    if kan_pkl_paths is not None:
        cmd += [
            "--kan_train_pkl", str(kan_pkl_paths["train"]),
            "--kan_val_pkl", str(kan_pkl_paths["val"]),
            "--kan_test_pkl", str(kan_pkl_paths["test"]),
        ]

    cmd += [
        "--kan_num_basis", str(effective["kan_num_basis"]),
        "--kan_hidden_dim", str(effective["kan_hidden_dim"]),
        "--kan_dropout", str(effective["kan_dropout"]),
        "--kan_epochs", str(effective["kan_epochs"]),
        "--kan_patience", str(effective["kan_patience"]),
        "--kan_batch_size", str(effective["kan_batch_size"]),
        "--kan_lr", str(effective["kan_lr"]),
        "--kan_weight_decay", str(effective["kan_weight_decay"]),
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
        # Repetido aquí aunque este subprocess no siempre reentrena el VAE:
        # main.py's Step 10 registra vae_hyperparams a partir de sus propios
        # args sin importar qué lo entrenó de verdad -- sin esto,
        # results/{run_id}.json reportaría mal beta/dropout en corridas que
        # usan un VAE aislado no-default.
        "--vae_beta", str(effective["vae_beta"]),
        "--vae_dropout", str(effective["vae_dropout"]),
    ]
    return cmd


def run_group(
    *,
    combo: List[str],
    label: str,
    group_name: str,
    candidates: Dict[str, Any],
    resolved: Dict[str, Any],
    apply_candidate,
    dry_run: bool,
) -> str:
    """Runs every candidate x SEEDS for one hyperparameter group, then
    aggregates (via aggregate_results) and returns the winning candidate
    label. `apply_candidate(resolved, value) -> effective_dict` builds the
    per-run config from the group's current resolved baseline + candidate.

    Before looping seeds, resolve_kan_input() decides -- from `effective`'s
    (latent, vae_beta, vae_dropout) -- whether this candidate can reuse the
    shared default-VAE cache or needs an isolated VAE + manual merge. This
    runs for every group (not just 'latent'/'vae_reg') because a non-default
    vae_reg winner from an earlier group must keep being honored by later
    groups (num_basis/hidden_dim/training), not silently fall back to the
    default cache."""

    print(f"\n  --- Grupo '{group_name}' ({len(candidates)} candidatos x {len(SEEDS)} semillas) ---")

    ok_keys = load_ok_run_keys(PHASE2_RESULTS_JSONL) if not dry_run else set()

    for cand_label, cand_value in candidates.items():
        effective = apply_candidate(resolved, cand_value)
        kan_pkl_paths = resolve_kan_input(combo, label, group_name, cand_label, effective, dry_run=dry_run)

        for seed in SEEDS:
            key = f"{label}__{group_name}__{cand_label}__seed{seed}"
            output_dir = KAN_RUNS_DIR / "phase2" / label / group_name / cand_label / f"seed{seed}"
            cmd = build_kan_command(combo, effective, seed, output_dir, kan_pkl_paths=kan_pkl_paths)

            if dry_run:
                print(f"    [{key}]\n      $ {' '.join(cmd)}")
                continue

            if key in ok_keys:
                print(f"    [{key}] SKIP (ya completada)")
                continue

            print(f"    [{key}] RUN")
            record = execute_and_log(
                run_key=key,
                cmd=cmd,
                jsonl_path=PHASE2_RESULTS_JSONL,
                meta={
                    "phase": "phase2",
                    "config_label": label,
                    "group": group_name,
                    "candidate_label": cand_label,
                    "active_extractors": combo,
                    "seed": seed,
                    "overrides": {k: v for k, v in effective.items() if k != "latent"},
                },
            )
            if record["status"] == "ok":
                print(f"      OK en {record['elapsed_seconds']}s -- {record['results_json']}")
            else:
                print(f"      FAILED -- {record['error']}")

    if dry_run:
        print(f"    (dry-run: candidato ganador de '{group_name}' no se puede resolver sin datos reales; "
              f"los grupos siguientes asumen el valor por defecto de este grupo.)")
        return None  # signals process_config to keep `resolved` at its current (default) values

    df = load_runs(PHASE2_RESULTS_JSONL)
    df = df[(df.get("config_label") == label) & (df.get("group") == group_name)]
    if df.empty:
        raise RuntimeError(f"Ninguna corrida exitosa en el grupo '{group_name}' de '{label}' -- no se puede continuar.")

    ranking = aggregate_by_config(df, group_by="candidate", metric=RANKING_METRIC)
    wilcoxon_df = pairwise_wilcoxon(df, "candidate", ranking, top_n=min(4, len(ranking)), metric=RANKING_METRIC)
    report(ranking, wilcoxon_df, metric=RANKING_METRIC, top_k=min(3, len(ranking)))

    winner_config_key = ranking.iloc[0]["config"]  # "{group_name}::{cand_label}"
    winner_label = winner_config_key.split("::", 1)[1]
    print(f"  Ganador del grupo '{group_name}': {winner_label}")
    return winner_label


def process_config(combo: List[str], dry_run: bool) -> Dict[str, Any]:
    label = config_label(combo)
    print(f"\n=== Configuración: {label} (extractores: {combo}) ===")

    resolved = initial_resolved()

    # (a) espacio latente
    winner = run_group(
        combo=combo, label=label, group_name="latent",
        candidates=LATENT_DIM_CANDIDATES, resolved=resolved,
        apply_candidate=lambda r, v: {**r, "latent": v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved["latent"] = LATENT_DIM_CANDIDATES[winner]

    # (a2) regularización del VAE (beta / dropout)
    winner = run_group(
        combo=combo, label=label, group_name="vae_reg",
        candidates=VAE_REG_CANDIDATES, resolved=resolved,
        apply_candidate=lambda r, v: {**r, **v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved.update(VAE_REG_CANDIDATES[winner])

    # (b) "entradas" del KAN == num_basis
    winner = run_group(
        combo=combo, label=label, group_name="num_basis",
        candidates={str(v): v for v in KAN_NUM_BASIS_CANDIDATES}, resolved=resolved,
        apply_candidate=lambda r, v: {**r, "kan_num_basis": v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved["kan_num_basis"] = int(winner)

    # (c) capas/nodos intermedios
    winner = run_group(
        combo=combo, label=label, group_name="hidden_dim",
        candidates={str(v): v for v in KAN_HIDDEN_DIM_CANDIDATES}, resolved=resolved,
        apply_candidate=lambda r, v: {**r, "kan_hidden_dim": v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved["kan_hidden_dim"] = int(winner)

    # (d) parámetros de entrenamiento
    winner = run_group(
        combo=combo, label=label, group_name="training",
        candidates=KAN_TRAINING_CANDIDATES, resolved=resolved,
        apply_candidate=lambda r, v: {**r, **v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved.update(KAN_TRAINING_CANDIDATES[winner])

    print(f"\n=== Resuelto para {label}: {resolved} ===")
    return resolved


def load_winning_configs(args) -> List[List[str]]:
    if args.configs:
        return json.loads(args.configs)

    winners_path = Path(args.winners) if args.winners else PHASE1_WINNERS_JSON
    if not winners_path.exists():
        raise FileNotFoundError(
            f"No se encontró {winners_path}. Generarlo con "
            f"'python scripts/aggregate_results.py --input <phase1.jsonl> --group-by extractors --output-winners' "
            f"o pasar --configs '[[...], [...]]' directamente."
        )
    with open(winners_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["winners"]


def main():
    parser = argparse.ArgumentParser(description="Fase 2: sweep secuencial de hiperparámetros sobre el top-3 de Fase 1")
    parser.add_argument("--winners", default=None, help=f"JSON con las configs ganadoras (default: {PHASE1_WINNERS_JSON})")
    parser.add_argument("--configs", default=None, help="Lista de combos inline, p.ej. '[[\"semantic\",\"style\"]]'")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = load_winning_configs(args)
    print(f"Fase 2: {len(configs)} configuraciones ganadoras: {configs}")
    print(f"Resultados: {PHASE2_RESULTS_JSONL}")

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        for combo in configs:
            results[config_label(combo)] = process_config(combo, dry_run=args.dry_run)

    if args.dry_run:
        print("\ndry-run: plan impreso, no se ejecutó nada.")
        return

    print("\n=== Resumen final Fase 2 ===")
    for label, resolved in results.items():
        print(f"  {label}: {resolved}")


if __name__ == "__main__":
    main()
