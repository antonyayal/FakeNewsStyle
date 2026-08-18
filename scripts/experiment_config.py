# scripts/experiment_config.py
# -*- coding: utf-8 -*-
"""
Shared module for the experiment orchestrators (Phase 1 and Phase 2):
fixed seeds, per-hyperparameter candidate values, and result paths.
Edit this file's constants to adjust the sweep scope without touching
orchestrator_phase1.py / orchestrator_phase2.py's logic.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Fixed seeds ------------------------------------------------------------
# Same set for every configuration, to allow paired comparisons
# (Wilcoxon signed-rank) between configurations.
SEEDS = [7, 42, 123, 777, 2024, 31415, 8675309, 20260817, 99, 1]

# ---- Experts / modalities ---------------------------------------------------
ALL_MODALITIES = ["semantic", "emotion", "style", "context"]

# main.py's default latent dimensions (must match its argparse defaults:
# --semantic_latent_dim, --emotion_latent_dim, etc.)
DEFAULT_LATENT_DIM = {"semantic": 128, "emotion": 16, "style": 16, "context": 64}

# ---- Phase 2: candidates per hyperparameter group ---------------------------
# (a) latent space -- presets applied simultaneously to every active branch
# of the configuration under test.
LATENT_DIM_CANDIDATES = {
    "small": {"semantic": 64, "emotion": 8, "style": 8, "context": 32},
    "default": dict(DEFAULT_LATENT_DIM),
    "large": {"semantic": 256, "emotion": 32, "style": 32, "context": 128},
}

# (a2) VAE regularization -- beta (KL weight) and dropout, one knob at a
# time over main.py's defaults. Runs after (a) because a historical run
# (results/20260812_022815_d3ede7c8.json, via the old
# scripts/run_full_stack_sweep.py) scored higher than any Phase 1 combo
# just by lowering beta to 0.25 -- no other Phase 2 group explored that
# dimension. Values matching DEFAULT_VAE_REG (beta=1.0, dropout=0.1) reuse
# the default cached VAE; any other value requires training a VAE in
# isolated directories and merging manually (see resolve_kan_input in
# orchestrator_phase2.py) because main.py's --merge_vae_latents always
# reads from the default path data/05_vae_latents/.
DEFAULT_VAE_REG = {"vae_beta": 1.0, "vae_dropout": 0.1}  # main.py defaults
VAE_REG_CANDIDATES = {
    "default": dict(DEFAULT_VAE_REG),
    "beta_low": {"vae_beta": 0.25},
    "beta_high": {"vae_beta": 4.0},
    "dropout_low": {"vae_dropout": 0.0},
    "dropout_high": {"vae_dropout": 0.3},
}

# (b) "KAN inputs" == num_basis (number of RBF basis functions per
# KANLayer -- there's no separate input_dim flag, see README_experiments.md)
KAN_NUM_BASIS_CANDIDATES = [4, 8, 16, 32]

# (c) KAN hidden layers/nodes
KAN_HIDDEN_DIM_CANDIDATES = [16, 32, 64, 128]

# (d) KAN training parameters -- each entry is a partial override on top of
# FIXED_KAN_BASELINE (one knob at a time).
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

# ---- Result paths -----------------------------------------------------------
RESULTS_DIR = BASE_DIR / "results"
PHASE1_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase1.jsonl"
PHASE2_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase2.jsonl"
PHASE1_WINNERS_JSON = RESULTS_DIR / "phase1_top3.json"

KAN_RUNS_DIR = BASE_DIR / "data" / "07_kan_runs"
VAE_LATENTS_DIR = BASE_DIR / "data" / "05_vae_latents"
FEATURES_RAW_DIR = BASE_DIR / "data" / "03_features_raw"

# Isolated directories for Phase 2's (a2) vae_reg group -- must never
# collide with VAE_LATENTS_DIR / "models/vae" (the default cache used by
# the rest of Phase 1/2), so training with a beta/dropout other than
# default at the same latent dimension doesn't overwrite it.
PHASE2_VAE_DATA_DIR = BASE_DIR / "data" / "05_vae_latents_phase2"
PHASE2_VAE_MODEL_DIR = BASE_DIR / "models" / "vae_phase2"
PHASE2_VAE_MERGED_DIR = BASE_DIR / "data" / "06_vae_latents_merged_phase2"

# Metric used for ranking and for the Wilcoxon test.
RANKING_METRIC = "f1"
