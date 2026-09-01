# scripts/experiment_config.py
# -*- coding: utf-8 -*-
"""
Shared module for the experiment orchestrators (Phase 1 through Phase 5):
fixed seeds, per-hyperparameter candidate values, and result paths.
Edit this file's constants to adjust the sweep scope without touching
orchestrator_phase{1..5}.py's logic.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Fixed seeds ------------------------------------------------------------
# Same set for every configuration, to allow paired comparisons
# (Wilcoxon signed-rank) between configurations.
SEEDS = [1, 7, 42, 123, 2024]

# ---- Experts / modalities ---------------------------------------------------
ALL_MODALITIES = ["semantic", "emotion", "style", "context"]

# main.py's default latent dimensions (must match its argparse defaults:
# --semantic_latent_dim, --emotion_latent_dim, etc.) -- used as the fallback
# for branches that are excluded from a given run (main.py still wants a
# value for every --{branch}_latent_dim flag even when --exclude_{branch}
# is also passed).
DEFAULT_LATENT_DIM = {"semantic": 128, "emotion": 16, "style": 16, "context": 64}

# ---- Phase 1: latent dimension per branch, in isolation ---------------------
# One extractor active at a time -- capped at each branch's raw feature
# dimension (no point asking a VAE to expand instead of compress):
# semantic 1024 (xlm-roberta-large hidden_size), emotion 23 (7 emo_probs +
# 3 sent_probs + 13 signals), style 35, context 86 (32 source + 32 domain +
# 16 topic + 0 author + 1 age + 5 has_* flags, main.py's defaults).
PHASE1_DIM_CANDIDATES = {
    "semantic": [64, 128, 256, 512, 1024],
    "emotion": [8, 16, 23],
    "style": [8, 16, 32, 35],
    "context": [8, 16, 32, 64, 86],
}
PHASE1_TOP_K = 2  # kept per branch

# ---- Phase 3: VAE-reg + KAN hyperparameters, fused, one knob at a time ------
# main.py's own defaults, used as the shared baseline every candidate below
# overrides exactly one field of.
PHASE3_BASELINE = {
    "vae_beta": 1.0,
    "vae_dropout": 0.1,
    "kan_num_basis": 16,
    "kan_hidden_dim": 64,
    "kan_weight_decay": 1e-4,
}
PHASE3_CANDIDATES = {
    "default": {},
    "beta_01": {"vae_beta": 0.1},
    "beta_025": {"vae_beta": 0.25},
    "beta_05": {"vae_beta": 0.5},
    "beta_4": {"vae_beta": 4.0},
    "dropout_0": {"vae_dropout": 0.0},
    "dropout_02": {"vae_dropout": 0.2},
    "dropout_03": {"vae_dropout": 0.3},
    "basis_4": {"kan_num_basis": 4},
    "basis_8": {"kan_num_basis": 8},
    "basis_32": {"kan_num_basis": 32},
    "hidden_16": {"kan_hidden_dim": 16},
    "hidden_32": {"kan_hidden_dim": 32},
    "hidden_128": {"kan_hidden_dim": 128},
    "wd_low": {"kan_weight_decay": 1e-5},
    "wd_high": {"kan_weight_decay": 1e-3},
}

# kan_lr / kan_batch_size are never swept -- a prior sweep already found the
# default wins for both, so they stay fixed everywhere via FIXED_KAN_BASELINE.
FIXED_KAN_BASELINE = {
    "kan_dropout": 0.2,
    "kan_epochs": 100,
    "kan_patience": 15,
    "kan_batch_size": 32,
    "kan_lr": 1e-3,
}

# ---- Result paths -----------------------------------------------------------
RESULTS_DIR = BASE_DIR / "results"
PHASE1_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase1.jsonl"
PHASE2_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase2.jsonl"
PHASE3_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase3.jsonl"
PHASE4_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase4.jsonl"
PHASE5_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase5.jsonl"

PHASE1_TOP_JSON = RESULTS_DIR / "phase1_top.json"
PHASE2_TOP_JSON = RESULTS_DIR / "phase2_top.json"
PHASE3_TOP_JSON = RESULTS_DIR / "phase3_top.json"
PHASE4_PER_FOLD_JSON = RESULTS_DIR / "phase4_per_fold.json"
PHASE4_TOP_JSON = RESULTS_DIR / "phase4_top.json"
PHASE5_PER_FOLD_JSON = RESULTS_DIR / "phase5_per_fold.json"
PHASE5_TOP_JSON = RESULTS_DIR / "phase5_top.json"

KAN_RUNS_DIR = BASE_DIR / "data" / "07_kan_runs"
VAE_LATENTS_DIR = BASE_DIR / "data" / "05_vae_latents"
FEATURES_RAW_DIR = BASE_DIR / "data" / "03_features_raw"

# Isolated directories for Phase 3's non-default (vae_beta, vae_dropout)
# candidates -- must never collide with VAE_LATENTS_DIR / "models/vae" (the
# default cache used by Phase 1/2/3's default-reg candidates), so training
# with a beta/dropout other than default at the same latent dimension
# doesn't overwrite it.
PHASE3_VAE_DATA_DIR = BASE_DIR / "data" / "05_vae_latents_phase3"
PHASE3_VAE_MODEL_DIR = BASE_DIR / "models" / "vae_phase3"
PHASE3_VAE_MERGED_DIR = BASE_DIR / "data" / "06_vae_latents_merged_phase3"

# Metric used for ranking and for the Wilcoxon test.
RANKING_METRIC = "f1"

# ---- Phase 4: CV packages (kfold, NOT source-disjoint) ----------------------
# Standard stratified k-fold over the pooled corpus (train+development+test).
# PHASE4_N_FOLDS/PHASE4_SPLIT_SEED here must match main.py's --kfold_n/
# --kfold_split_seed defaults so orchestrator_phase4.py and any bare
# `python main.py --corpus_mode kfold ...` invocation address the exact same
# cached folds.
PHASE4_N_FOLDS = 5
PHASE4_SPLIT_SEED = 20260820
PHASE4_KAN_RUNS_DIR = KAN_RUNS_DIR / "phase4"
PHASE4_MERGED_CV_DIR = BASE_DIR / "data" / "06_vae_latents_merged_cv"

# ---- Phase 5: source-disjoint packages ---------------------------------------
# Same protocol as Phase 4 but partitioned so no Source (news outlet) appears
# in more than one of a fold's train/val/test (StratifiedGroupKFold +
# GroupShuffleSplit grouped by Source, see src/data/source_split_corpus.py).
# PHASE5_N_FOLDS/PHASE5_SPLIT_SEED here must match main.py's --source_split_n/
# --source_split_seed defaults.
PHASE5_N_FOLDS = 5
PHASE5_SPLIT_SEED = 20260821
PHASE5_KAN_RUNS_DIR = KAN_RUNS_DIR / "phase5"
PHASE5_MERGED_CV_DIR = BASE_DIR / "data" / "06_vae_latents_merged_source_cv"
