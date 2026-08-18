# scripts/orchestrator_phase1.py
# -*- coding: utf-8 -*-
"""
Fase 1: sweep completo de combinaciones de expertos (los 15 subconjuntos no
vacíos de {semantic, emotion, style, context}) x las 10 semillas fijas de
experiment_config.SEEDS = 150 corridas.

VAE se entrena UNA sola vez (a las dimensiones latentes por defecto, para
las 4 ramas) antes del sweep -- el entrenamiento de VAE es independiente por
rama en main.py (no depende de qué otras ramas estén activas) y no tiene
semilla configurable, así que reentrenarlo por corrida solo añadiría ruido
no controlado a una comparación que se supone pareada por semilla. Cada una
de las 150 corridas solo ejecuta `--merge_vae_latents --train_kan`, reusando
esos latentes cacheados.

Checkpointing: antes de lanzar una corrida se revisa
experiment_config.PHASE1_RESULTS_JSONL; si ya existe una línea con el mismo
run_key y status "ok", se salta. Reanudar tras una caída es simplemente
volver a correr este mismo script.

Uso:
    python scripts/orchestrator_phase1.py              # corre las 150 (o las que falten)
    python scripts/orchestrator_phase1.py --dry-run     # solo imprime el plan
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import (  # noqa: E402
    ALL_MODALITIES,
    BASE_DIR,
    DEFAULT_LATENT_DIM,
    KAN_RUNS_DIR,
    PHASE1_RESULTS_JSONL,
    SEEDS,
    VAE_LATENTS_DIR,
)
from experiment_runner import (  # noqa: E402
    execute_and_log,
    load_ok_run_keys,
    python_executable,
    run_main_command,
)


def all_nonempty_combos(modalities: List[str]) -> List[List[str]]:
    combos = []
    for r in range(1, len(modalities) + 1):
        combos.extend(list(c) for c in itertools.combinations(modalities, r))
    return combos


COMBOS = all_nonempty_combos(ALL_MODALITIES)


def combo_label(combo: List[str]) -> str:
    return "_".join(combo)


def run_key_for(combo: List[str], seed: int) -> str:
    return f"{combo_label(combo)}__seed{seed}"


def default_vae_latents_missing() -> List[str]:
    missing = []
    for branch in ALL_MODALITIES:
        dim = DEFAULT_LATENT_DIM[branch]
        branch_dir = VAE_LATENTS_DIR / branch / f"latent{dim}"
        for split in ["train", "val", "test"]:
            if not (branch_dir / f"{split}.pkl").exists():
                missing.append(f"{branch} (latent{dim}, {split}.pkl)")
    return missing


def build_vae_prep_command() -> List[str]:
    cmd = [python_executable(), "main.py", "--run_vaes"]
    for branch in ALL_MODALITIES:
        cmd += [f"--{branch}_latent_dim", str(DEFAULT_LATENT_DIM[branch])]
    return cmd


def ensure_default_vae_latents(dry_run: bool) -> None:
    missing = default_vae_latents_missing()
    if not missing:
        print("VAE latentes por defecto ya existen para las 4 ramas -- se reusan.")
        return

    print("VAE latentes por defecto faltantes, se entrenarán una sola vez:")
    for m in missing:
        print(f"  - {m}")

    cmd = build_vae_prep_command()
    print(f"  $ {' '.join(cmd)}")

    if dry_run:
        print("  (dry-run: no se ejecuta)")
        return

    outcome = run_main_command(cmd)
    if outcome["error"] is not None:
        raise RuntimeError(f"Fallo entrenando VAE por defecto: {outcome['error']}")
    print(f"  OK en {outcome['elapsed_seconds']}s")


def build_kan_command(combo: List[str], seed: int) -> List[str]:
    cmd = [python_executable(), "main.py", "--merge_vae_latents", "--train_kan"]

    for modality in ALL_MODALITIES:
        if modality not in combo:
            cmd.append(f"--exclude_{modality}")

    for branch in ALL_MODALITIES:
        cmd += [f"--{branch}_latent_dim", str(DEFAULT_LATENT_DIM[branch])]

    output_dir = KAN_RUNS_DIR / "phase1" / combo_label(combo) / f"seed{seed}"
    cmd += [
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Fase 1: sweep de combos de expertos x semillas")
    parser.add_argument("--dry-run", action="store_true", help="Imprime el plan sin ejecutar nada")
    args = parser.parse_args()

    total = len(COMBOS) * len(SEEDS)
    print(f"Fase 1: {len(COMBOS)} combos x {len(SEEDS)} semillas = {total} corridas")
    print(f"Resultados: {PHASE1_RESULTS_JSONL}")

    ensure_default_vae_latents(dry_run=args.dry_run)

    ok_keys = load_ok_run_keys(PHASE1_RESULTS_JSONL) if not args.dry_run else set()
    if ok_keys:
        print(f"Reanudando: {len(ok_keys)}/{total} corridas ya completadas, se saltan.")

    n_run = 0
    n_skip = 0
    n_failed = 0
    idx = 0

    for combo in COMBOS:
        for seed in SEEDS:
            idx += 1
            key = run_key_for(combo, seed)
            label = f"[{idx:03d}/{total}] {key}"

            if args.dry_run:
                cmd = build_kan_command(combo, seed)
                print(f"{label}\n  $ {' '.join(cmd)}")
                continue

            if key in ok_keys:
                print(f"{label} SKIP (ya completada)")
                n_skip += 1
                continue

            cmd = build_kan_command(combo, seed)
            print(f"{label} RUN\n  $ {' '.join(cmd)}")

            record = execute_and_log(
                run_key=key,
                cmd=cmd,
                jsonl_path=PHASE1_RESULTS_JSONL,
                meta={
                    "phase": "phase1",
                    "group": None,
                    "candidate_label": None,
                    "active_extractors": combo,
                    "seed": seed,
                    "overrides": {},
                    "kan_output_dir": str(cmd[cmd.index("--kan_output_dir") + 1]),
                },
            )

            if record["status"] == "ok":
                n_run += 1
                print(f"  OK en {record['elapsed_seconds']}s -- {record['results_json']}")
            else:
                n_failed += 1
                print(f"  FAILED -- {record['error']}")

    if args.dry_run:
        print(f"\ndry-run: {total} corridas planeadas (no ejecutadas).")
        return

    print(f"\nFase 1 completa (esta invocación): {n_run} corridas nuevas, {n_skip} saltadas, {n_failed} fallidas.")
    print(f"Total acumulado en {PHASE1_RESULTS_JSONL}: {len(load_ok_run_keys(PHASE1_RESULTS_JSONL))}/{total} ok.")


if __name__ == "__main__":
    main()
