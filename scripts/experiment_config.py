# scripts/experiment_config.py
# -*- coding: utf-8 -*-
"""
Módulo compartido para los orquestadores de experimentos (Fase 1 y Fase 2):
semillas fijas, valores candidatos por hiperparámetro, y rutas de resultados.
Editar las constantes de este archivo para ajustar el alcance de los sweeps
sin tocar la lógica de orchestrator_phase1.py / orchestrator_phase2.py.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Semillas fijas -------------------------------------------------------
# Mismo conjunto para todas las configuraciones, para permitir comparaciones
# pareadas (Wilcoxon signed-rank) entre configuraciones.
SEEDS = [7, 42, 123, 777, 2024, 31415, 8675309, 20260817, 99, 1]

# ---- Expertos / modalidades -----------------------------------------------
ALL_MODALITIES = ["semantic", "emotion", "style", "context"]

# Dimensiones latentes por defecto de main.py (deben coincidir con sus
# argparse defaults: --semantic_latent_dim, --emotion_latent_dim, etc.)
DEFAULT_LATENT_DIM = {"semantic": 128, "emotion": 16, "style": 16, "context": 64}

# ---- Fase 2: candidatos por grupo de hiperparámetros -----------------------
# (a) espacio latente -- presets aplicados simultáneamente a todas las ramas
# activas de la configuración bajo prueba.
LATENT_DIM_CANDIDATES = {
    "small": {"semantic": 64, "emotion": 8, "style": 8, "context": 32},
    "default": dict(DEFAULT_LATENT_DIM),
    "large": {"semantic": 256, "emotion": 32, "style": 32, "context": 128},
}

# (a2) regularización del VAE -- beta (peso KL) y dropout, una perilla a la
# vez sobre los defaults de main.py. Se corre después de (a) porque una
# corrida histórica (results/20260812_022815_d3ede7c8.json, vía el antiguo
# scripts/run_full_stack_sweep.py) midió más alto que cualquier combo de la
# Fase 1 solo bajando beta a 0.25 -- ningún otro grupo de la Fase 2 exploraba
# esa dimensión. Valores que coinciden con DEFAULT_VAE_REG (beta=1.0,
# dropout=0.1) reusan el VAE cacheado por defecto; cualquier otro valor
# requiere entrenar VAE en directorios aislados y mergear manualmente (ver
# resolve_kan_input en orchestrator_phase2.py) porque --merge_vae_latents de
# main.py siempre lee del path por defecto data/05_vae_latents/.
DEFAULT_VAE_REG = {"vae_beta": 1.0, "vae_dropout": 0.1}  # defaults de main.py
VAE_REG_CANDIDATES = {
    "default": dict(DEFAULT_VAE_REG),
    "beta_low": {"vae_beta": 0.25},
    "beta_high": {"vae_beta": 4.0},
    "dropout_low": {"vae_dropout": 0.0},
    "dropout_high": {"vae_dropout": 0.3},
}

# (b) "entradas del KAN" == num_basis (número de funciones base RBF por
# KANLayer -- no existe un flag de input_dim separado, ver README_experiments.md)
KAN_NUM_BASIS_CANDIDATES = [4, 8, 16, 32]

# (c) capas/nodos intermedios del KAN
KAN_HIDDEN_DIM_CANDIDATES = [16, 32, 64, 128]

# (d) parámetros de entrenamiento del KAN -- cada entry es un override
# parcial sobre FIXED_KAN_BASELINE (una perilla a la vez).
KAN_TRAINING_CANDIDATES = {
    "lr_low": {"kan_lr": 1e-4},
    "lr_default": {"kan_lr": 1e-3},
    "lr_high": {"kan_lr": 5e-3},
    "batch_16": {"kan_batch_size": 16},
    "batch_64": {"kan_batch_size": 64},
    "wd_low": {"kan_weight_decay": 1e-5},
    "wd_high": {"kan_weight_decay": 1e-3},
}

FIXED_KAN_BASELINE = {
    "kan_dropout": 0.2,
    "kan_epochs": 100,
    "kan_patience": 15,
    "kan_batch_size": 32,
    "kan_lr": 1e-3,
    "kan_weight_decay": 1e-4,
}

# ---- Rutas de resultados ----------------------------------------------------
RESULTS_DIR = BASE_DIR / "results"
PHASE1_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase1.jsonl"
PHASE2_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase2.jsonl"
PHASE1_WINNERS_JSON = RESULTS_DIR / "phase1_top3.json"

KAN_RUNS_DIR = BASE_DIR / "data" / "07_kan_runs"
VAE_LATENTS_DIR = BASE_DIR / "data" / "05_vae_latents"

# Directorios aislados para el grupo (a2) vae_reg de la Fase 2 -- nunca deben
# coincidir con VAE_LATENTS_DIR / "models/vae" (el cache por defecto que usa
# el resto de la Fase 1/2), para no sobreescribirlo al entrenar con beta/
# dropout distintos al default en la misma dimensión latente.
PHASE2_VAE_DATA_DIR = BASE_DIR / "data" / "05_vae_latents_phase2"
PHASE2_VAE_MODEL_DIR = BASE_DIR / "models" / "vae_phase2"
PHASE2_VAE_MERGED_DIR = BASE_DIR / "data" / "06_vae_latents_merged_phase2"

# Métrica usada para rankear y para el test de Wilcoxon.
RANKING_METRIC = "f1"
