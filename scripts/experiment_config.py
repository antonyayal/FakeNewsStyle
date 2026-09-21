# scripts/experiment_config.py
# -*- coding: utf-8 -*-
"""
Shared config for the 3-phase experiment plan (Option A):

  Phase 1  -- context VAE latent dimension, swept for both identity modes
              (Source/Domain hash embeddings ON vs. OFF). The other three
              branches are NOT swept; they use main.py's default dims, since
              an earlier full per-branch sweep showed the latent size barely
              moves F1 within a branch.
  Phase 2  -- standard-split evaluation. All 15 non-empty extractor combos;
              every combo that includes `context` is run twice (identity ON
              and OFF); every combo-variant at both KAN hidden_dim values;
              x SEEDS. Nothing is filtered.
  Phase 3  -- source-disjoint validation (no news outlet in more than one of
              a fold's train/val/test). Same 23 combo-variants x hidden_dim x
              SEEDS x folds. The contrast Phase 2 vs Phase 3 quantifies how
              much of the F1 was outlet memorization.

VAE and KAN hyperparameters are FIXED (see FINAL_HPARAMS) -- picked from the
old Phase 3 one-knob sweep, where no setting beat main.py's defaults by more
than seed noise (only `vae_beta=4.0` clearly hurt). `kan_hidden_dim` is the
one exception: 32 vs 64 was a coin flip in that data, so both are kept as
the single swept axis.

Edit the constants here to change the sweep scope without touching the
orchestrators.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Fixed seeds ----------------------------------------------------------
# Same set for every configuration, for paired comparisons (Wilcoxon
# signed-rank) between configs. Trimmed from 5 to 3 -- within-config seed
# std on this task is ~0.016 F1, so 3 seeds already pin the mean tightly.
SEEDS = [1, 7, 42]

# ---- Experts / modalities -----------------------------------------------
ALL_MODALITIES = ["semantic", "emotion", "style", "context"]

# main.py's default latent dimensions (must match its argparse defaults).
# The non-context branches use these verbatim in every phase; only `context`
# gets a swept dimension (Phase 1).
DEFAULT_LATENT_DIM = {"semantic": 128, "emotion": 16, "style": 16, "context": 64}

# ---- Fixed VAE + KAN hyperparameters -----------------------------------
# Everything except kan_hidden_dim. Sourced from the old Phase 3 sweep
# (results/orchestrator_phase3.jsonl): marginal effect of each knob was
# below seed noise, so these are main.py's defaults.
FINAL_HPARAMS = {
    "vae_beta": 1.0,
    "vae_dropout": 0.1,
    "kan_num_basis": 16,
    "kan_dropout": 0.2,
    "kan_weight_decay": 1e-4,
    "kan_epochs": 100,
    "kan_patience": 15,
    "kan_batch_size": 32,
    "kan_lr": 1e-3,
}

# The one swept hyperparameter -- 32 vs 64 was within noise in the old
# sweep, so both are carried through every phase.
HIDDEN_DIM_GRID = [32, 64]

# ---- Phase 1: context latent dimension, per identity mode --------------
# identity ON  -> full context raw dim 86 (32 source + 32 domain + 16 topic
#                 + 0 author + 1 age + 5 flags), main.py's context defaults.
# identity OFF -> --context_source_dim 0 --context_domain_dim 0, raw dim
#                 shrinks to 23 (16 topic + 1 age + 6 flags).
PHASE1_CONTEXT_ON_DIMS = [8, 16, 32, 64, 86]
PHASE1_CONTEXT_OFF_DIMS = [4, 8, 16, 23]
PHASE1_TOP_K = 2  # kept per identity mode in phase1_top.json

# Fallbacks if Phase 1 hasn't run: main.py's default for ON, a mid value
# for OFF (capped at the identity-free raw dim of 23).
FALLBACK_CONTEXT_ON_DIM = DEFAULT_LATENT_DIM["context"]
FALLBACK_CONTEXT_OFF_DIM = 16

# ---- Identity-free context (Source/Domain switched off) ---------------
CONTEXT_IDFREE_SOURCE_DIM = 0
CONTEXT_IDFREE_DOMAIN_DIM = 0

# ---- Source-disjoint folds (Phase 3) ---------------------------------
# Must match main.py's --source_split_n / --source_split_seed defaults so a
# bare `python main.py --corpus_mode source_disjoint ...` addresses the same
# cached folds.
SOURCE_SPLIT_N_FOLDS = 5
SOURCE_SPLIT_SEED = 20260821

# ---- Ranking metric --------------------------------------------------
RANKING_METRIC = "f1"

# ---- Result paths --------------------------------------------------
RESULTS_DIR = BASE_DIR / "results"
PHASE1_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase1.jsonl"
PHASE2_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase2.jsonl"
PHASE3_RESULTS_JSONL = RESULTS_DIR / "orchestrator_phase3.jsonl"

PHASE1_TOP_JSON = RESULTS_DIR / "phase1_top.json"
PHASE2_TOP_JSON = RESULTS_DIR / "phase2_top.json"
PHASE3_PER_FOLD_JSON = RESULTS_DIR / "phase3_per_fold.json"
PHASE3_TOP_JSON = RESULTS_DIR / "phase3_top.json"

# ---- Data / model paths --------------------------------------------
KAN_RUNS_DIR = BASE_DIR / "data" / "07_kan_runs"
VAE_LATENTS_DIR = BASE_DIR / "data" / "05_vae_latents"          # shared default cache
FEATURES_RAW_DIR = BASE_DIR / "data" / "03_features_raw"

# Phase 2's manually merged KAN inputs (one dir per prep unit).
PHASE2_MERGED_DIR = BASE_DIR / "data" / "06_vae_latents_merged_optA"

# Isolated identity-free context artifacts, standard split (Phase 1 & 2).
IDFREE_CONTEXT_RAW_DIR = BASE_DIR / "data" / "03_features_raw_idfree" / "context"
IDFREE_CONTEXT_VAE_DATA_DIR = BASE_DIR / "data" / "05_vae_latents_idfree"
IDFREE_CONTEXT_VAE_MODEL_DIR = BASE_DIR / "models" / "vae_idfree"

# Source-disjoint per-fold caches (Phase 3).
#   non-context branches (+ identity-ON context) -> main.py's shared
#   source_cv cache, namespaced by fold by main.py itself:
SOURCE_CV_RAW_DIR = BASE_DIR / "data" / "03_features_raw_source_cv"
SOURCE_CV_VAE_DIR = BASE_DIR / "data" / "05_vae_latents_source_cv"
#   identity-OFF context -> its own isolated per-fold tree:
IDFREE_CONTEXT_FOLD_RAW_DIR = BASE_DIR / "data" / "03_features_raw_source_cv_idfree"
IDFREE_CONTEXT_FOLD_VAE_DATA_DIR = BASE_DIR / "data" / "05_vae_latents_source_cv_idfree"
IDFREE_CONTEXT_FOLD_VAE_MODEL_DIR = BASE_DIR / "models" / "vae_source_cv_idfree"
#   Phase 3's manually merged KAN inputs, namespaced by fold + prep unit:
PHASE3_MERGED_DIR = BASE_DIR / "data" / "06_vae_latents_merged_source_cv_optA"


def source_cv_fold_dir(base: Path, fold_idx: int) -> Path:
    """main.py's fold namespacing for source_disjoint mode
    (src/data/source_split_corpus.py's fold_dir): base/seed{S}_n{N}/fold{k}."""
    return base / f"seed{SOURCE_SPLIT_SEED}_n{SOURCE_SPLIT_N_FOLDS}" / f"fold{fold_idx}"
