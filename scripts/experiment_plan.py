# scripts/experiment_plan.py
# -*- coding: utf-8 -*-
"""
Shared plan primitives for the 3-phase Option A orchestrators
(orchestrator_phase{1,2,3}.py):

  - the 23 combo-variants (15 extractor combos; the 8 that include `context`
    doubled for identity ON/OFF) as "prep units", each carrying its latent
    dimensions;
  - the KAN grid on top of each prep unit (kan_hidden_dim x SEEDS);
  - labels used for run_key / output-dir / merged-latent namespacing;
  - identity-free `context` extraction + VAE, for the standard split and for
    a source-disjoint fold (the only branch that ever needs an isolated
    cache now that vae_beta/vae_dropout are fixed at main.py's defaults).

Inputs:  experiment_config constants, results/phase1_top.json (optional).
Outputs: none (pure helpers).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from experiment_config import (
    ALL_MODALITIES,
    BASE_DIR,
    CONTEXT_IDFREE_DOMAIN_DIM,
    CONTEXT_IDFREE_SOURCE_DIM,
    DEFAULT_LATENT_DIM,
    FALLBACK_CONTEXT_OFF_DIM,
    FALLBACK_CONTEXT_ON_DIM,
    FINAL_HPARAMS,
    HIDDEN_DIM_GRID,
    IDFREE_CONTEXT_FOLD_RAW_DIR,
    IDFREE_CONTEXT_FOLD_VAE_DATA_DIR,
    IDFREE_CONTEXT_FOLD_VAE_MODEL_DIR,
    IDFREE_CONTEXT_RAW_DIR,
    IDFREE_CONTEXT_VAE_DATA_DIR,
    IDFREE_CONTEXT_VAE_MODEL_DIR,
    PHASE1_TOP_JSON,
    SEEDS,
    SOURCE_SPLIT_N_FOLDS,
    SOURCE_SPLIT_SEED,
)
from experiment_runner import (
    latent_cache_is_fresh,
    merge_latents_manual,
    python_executable,
    run_main_command,
)

CONTEXT = "context"


# ---------------------------------------------------------------- labels

def config_label(combo: List[str]) -> str:
    """Canonical combo name, modalities in ALL_MODALITIES order."""
    return "_".join(m for m in ALL_MODALITIES if m in combo)


def prep_label(combo: List[str], context_identity: Optional[str]) -> str:
    """Namespaces everything that depends on (combo, identity) but NOT on
    the KAN hyperparameters: merged latents, fold caches, marker files."""
    base = config_label(combo)
    if CONTEXT in combo:
        return f"{base}__ctx{context_identity}"
    return base


def entry_label(combo: List[str], context_identity: Optional[str], hidden_dim: int) -> str:
    """Full run identity, one KAN training. Used for run_key and kan_output_dir."""
    return f"{prep_label(combo, context_identity)}__hd{hidden_dim}"


# ---------------------------------------------------------------- combos

def all_nonempty_combos() -> List[List[str]]:
    combos: List[List[str]] = []
    for r in range(1, len(ALL_MODALITIES) + 1):
        combos.extend(list(c) for c in itertools.combinations(ALL_MODALITIES, r))
    return combos


def load_context_dims() -> Tuple[int, int]:
    """(identity_on_dim, identity_off_dim) from results/phase1_top.json, or
    the configured fallbacks if Phase 1 hasn't produced it yet."""
    if not PHASE1_TOP_JSON.exists():
        return FALLBACK_CONTEXT_ON_DIM, FALLBACK_CONTEXT_OFF_DIM
    with open(PHASE1_TOP_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    on = data.get("context_on", {}).get("dim", FALLBACK_CONTEXT_ON_DIM)
    off = data.get("context_off", {}).get("dim", FALLBACK_CONTEXT_OFF_DIM)
    return int(on), int(off)


def build_prep_units(context_on_dim: int, context_off_dim: int) -> List[Dict[str, Any]]:
    """The 23 combo-variants. Each is a dict:
        active_extractors : list[str]
        context_identity  : "on" | "off" | None  (None when combo has no context)
        latent_dims       : {branch: dim} for every branch in the combo
        prep_label        : str
    """
    units: List[Dict[str, Any]] = []
    for combo in all_nonempty_combos():
        modes = ["on", "off"] if CONTEXT in combo else [None]
        for mode in modes:
            latent_dims: Dict[str, int] = {}
            for b in combo:
                if b == CONTEXT:
                    latent_dims[b] = context_off_dim if mode == "off" else context_on_dim
                else:
                    latent_dims[b] = DEFAULT_LATENT_DIM[b]
            units.append({
                "active_extractors": combo,
                "context_identity": mode,
                "latent_dims": latent_dims,
                "prep_label": prep_label(combo, mode),
            })
    return units


def iter_kan_entries(prep_units: List[Dict[str, Any]]) -> Iterator[Tuple[Dict[str, Any], int]]:
    """(prep_unit, hidden_dim) for every KAN training in a phase (before the
    seed loop)."""
    for unit in prep_units:
        for hidden_dim in HIDDEN_DIM_GRID:
            yield unit, hidden_dim


def total_kan_runs(prep_units: List[Dict[str, Any]], n_folds: int = 1) -> int:
    return len(prep_units) * len(HIDDEN_DIM_GRID) * len(SEEDS) * n_folds


# ---------------------------------------------------------------- KAN command

def build_kan_cmd(
    *,
    combo: List[str],
    latent_dims: Dict[str, int],
    pkl_paths: Dict[str, Path],
    hidden_dim: int,
    seed: int,
    output_dir: Path,
    extra_flags: Optional[List[str]] = None,
) -> List[str]:
    """`main.py --train_kan` reading pre-merged KAN inputs from pkl_paths,
    with FINAL_HPARAMS + the given hidden_dim. extra_flags carries the
    --corpus_mode block for Phase 3 (harmless for Phase 1/2)."""
    cmd = [python_executable(), "main.py"]
    if extra_flags:
        cmd += extra_flags
    cmd.append("--train_kan")
    for b in ALL_MODALITIES:
        if b not in combo:
            cmd.append(f"--exclude_{b}")
        cmd += [f"--{b}_latent_dim", str(latent_dims.get(b, DEFAULT_LATENT_DIM[b]))]
    cmd += [
        "--kan_train_pkl", str(pkl_paths["train"]),
        "--kan_val_pkl", str(pkl_paths["val"]),
        "--kan_test_pkl", str(pkl_paths["test"]),
        "--kan_hidden_dim", str(hidden_dim),
        "--kan_num_basis", str(FINAL_HPARAMS["kan_num_basis"]),
        "--kan_dropout", str(FINAL_HPARAMS["kan_dropout"]),
        "--kan_weight_decay", str(FINAL_HPARAMS["kan_weight_decay"]),
        "--kan_epochs", str(FINAL_HPARAMS["kan_epochs"]),
        "--kan_patience", str(FINAL_HPARAMS["kan_patience"]),
        "--kan_batch_size", str(FINAL_HPARAMS["kan_batch_size"]),
        "--kan_lr", str(FINAL_HPARAMS["kan_lr"]),
        "--kan_seed", str(seed),
        "--kan_output_dir", str(output_dir.relative_to(BASE_DIR)),
        "--vae_beta", str(FINAL_HPARAMS["vae_beta"]),
        "--vae_dropout", str(FINAL_HPARAMS["vae_dropout"]),
    ]
    return cmd


def merge_kan_inputs(
    combo: List[str], latent_dirs: Dict[str, Path], out_dir: Path, dry_run: bool
) -> Dict[str, Path]:
    """Concat per-branch VAE latents into KAN-ready {split}.pkl under out_dir
    (main.py's --merge_vae_latents logic, but from arbitrary dirs). On a
    dry-run just returns the paths it would write."""
    if dry_run:
        return {s: out_dir / f"{s}.pkl" for s in ("train", "val", "test")}
    return merge_latents_manual(combo, latent_dirs, out_dir)


# ---------------------------------------------- identity-free context

def _corpus_flags_source_disjoint(fold_idx: int) -> List[str]:
    return ["--corpus_mode", "source_disjoint",
            "--source_split_n", str(SOURCE_SPLIT_N_FOLDS),
            "--source_split_index", str(fold_idx),
            "--source_split_seed", str(SOURCE_SPLIT_SEED)]


def _run(cmd: List[str], what: str, dry_run: bool) -> None:
    print(f"    $ {' '.join(cmd)}")
    if dry_run:
        return
    outcome = run_main_command(cmd, require_results_json=False)
    if outcome["error"] is not None:
        raise RuntimeError(f"{what} failed: {outcome['error']}")
    print(f"      OK in {outcome['elapsed_seconds']}s")


def ensure_idfree_context(dim: int, *, fold_idx: Optional[int] = None, dry_run: bool = False) -> Path:
    """Extract `context` with Source/Domain switched off and train its VAE
    into an isolated cache, reusing whatever is already fresh on disk.
    Returns the latent dir holding {train,val,test}.pkl for that dim.

    fold_idx=None  -> standard split, caches under *_idfree/.
    fold_idx=k     -> source-disjoint fold k, caches under *_source_cv_idfree/fold{k}/.
    """
    if fold_idx is None:
        raw_dir = IDFREE_CONTEXT_RAW_DIR
        vae_data_dir = IDFREE_CONTEXT_VAE_DATA_DIR
        vae_model_dir = IDFREE_CONTEXT_VAE_MODEL_DIR
        corpus_flags: List[str] = []
        tag = "standard split"
    else:
        raw_dir = IDFREE_CONTEXT_FOLD_RAW_DIR / f"fold{fold_idx}"
        vae_data_dir = IDFREE_CONTEXT_FOLD_VAE_DATA_DIR / f"fold{fold_idx}"
        vae_model_dir = IDFREE_CONTEXT_FOLD_VAE_MODEL_DIR / f"fold{fold_idx}"
        corpus_flags = _corpus_flags_source_disjoint(fold_idx)
        tag = f"source-disjoint fold {fold_idx}"

    latent_dir = vae_data_dir / "context" / f"latent{dim}"

    if not dry_run and latent_cache_is_fresh("context", dim, vae_data_dir, raw_dir_override=raw_dir):
        print(f"    identity-free context @ dim={dim} ({tag}): cached, skip")
        return latent_dir

    raw_train = raw_dir / "train_context.pkl"
    if dry_run or not raw_train.exists():
        # --preprocess_text is idempotent and guarantees the (fold's) cleaned
        # corpus exists even when this combo-variant has no non-context branch
        # to have built it first.
        _run(
            [python_executable(), "main.py", *corpus_flags,
             "--preprocess_text", "--extract_context",
             "--context_output_dir", str(raw_dir.relative_to(BASE_DIR)),
             "--context_source_dim", str(CONTEXT_IDFREE_SOURCE_DIM),
             "--context_domain_dim", str(CONTEXT_IDFREE_DOMAIN_DIM)],
            f"identity-free context extract ({tag})", dry_run,
        )

    _run(
        [python_executable(), "main.py", "--run_vaes",
         "--exclude_semantic", "--exclude_emotion", "--exclude_style",
         "--context_latent_dim", str(dim),
         "--context_vae_input_dir", str(raw_dir.relative_to(BASE_DIR)),
         "--vae_data_output_dir", str(vae_data_dir.relative_to(BASE_DIR)),
         "--vae_model_output_dir", str(vae_model_dir.relative_to(BASE_DIR))],
        f"identity-free context VAE @ dim={dim} ({tag})", dry_run,
    )
    return latent_dir
