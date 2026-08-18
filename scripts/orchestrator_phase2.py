# scripts/orchestrator_phase2.py
# -*- coding: utf-8 -*-
"""
Phase 2: for each of the 3 winning configurations from Phase 1, runs 5 sweep
groups in a fixed sequential order -- each group is aggregated with
aggregate_results before generating the next one's runs, so the full plan is
NOT known ahead of time (it's decided at runtime):

  (a)  latent space          (--{branch}_latent_dim, active branches only)
  (a2) VAE regularization    (--vae_beta / --vae_dropout -- see DEFAULT_VAE_REG
                               in experiment_config.py; needed because a
                               historical run with low beta scored higher
                               than any Phase 1 combo)
  (b)  KAN "inputs"          (--kan_num_basis -- there's no separate
                               input_dim flag; the KAN's real input is the
                               sum of the active latent dims, already
                               covered by (a))
  (c)  internal layers/nodes (--kan_hidden_dim)
  (d)  training parameters   (--kan_lr / --kan_batch_size / --kan_weight_decay)

Each candidate value runs with the 10 fixed seeds from experiment_config.SEEDS.
Checkpointing/resume identical to Phase 1, over experiment_config.PHASE2_RESULTS_JSONL.

Note on (a2): main.py's --merge_vae_latents always reads from the fixed path
data/05_vae_latents/{branch}/latent{dim}/ (hardcoded, ignores
--vae_data_output_dir), so any candidate with a beta/dropout other than
main.py's default (see DEFAULT_VAE_REG) is trained in an isolated directory
(PHASE2_VAE_DATA_DIR) and merged manually in Python (same pattern as
scripts/run_full_stack_sweep.py), pointing --train_kan at those PKLs via
--kan_train_pkl/--kan_val_pkl/--kan_test_pkl. Candidates matching the default
keep reusing the normal shared cache.

Usage:
    python scripts/orchestrator_phase2.py                       # uses results/phase1_top3.json
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
    latent_cache_is_fresh,
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
    missing = [
        f"{branch} (latent{dims[branch]}, missing or stale vs. current corpus)"
        for branch in active_extractors
        if not latent_cache_is_fresh(branch, dims[branch], VAE_LATENTS_DIR)
    ]

    if not missing:
        return

    print(f"    Missing VAE latents for this preset: {missing}")
    cmd = [python_executable(), "main.py", "--run_vaes"]
    for branch in ALL_MODALITIES:
        if branch not in active_extractors:
            cmd.append(f"--exclude_{branch}")
        cmd += [f"--{branch}_latent_dim", str(dims.get(branch, DEFAULT_LATENT_DIM[branch]))]
    print(f"    $ {' '.join(cmd)}")

    if dry_run:
        print("    (dry-run: not executing)")
        return

    outcome = run_main_command(cmd, require_results_json=False)
    if outcome["error"] is not None:
        raise RuntimeError(f"Failed training VAE for {active_extractors} @ {dims}: {outcome['error']}")
    print(f"    OK in {outcome['elapsed_seconds']}s")


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
        if not latent_cache_is_fresh(branch, effective["latent"][branch], vae_data_dir)
    ]

    if missing:
        print(f"    Missing isolated VAE (beta={effective['vae_beta']}, dropout={effective['vae_dropout']}): {missing}")
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
            print("    (dry-run: not executing)")
        else:
            outcome = run_main_command(cmd, require_results_json=False)
            if outcome["error"] is not None:
                raise RuntimeError(
                    f"Failed training isolated VAE for {combo} "
                    f"@ beta={effective['vae_beta']} dropout={effective['vae_dropout']}: {outcome['error']}"
                )
            print(f"    OK in {outcome['elapsed_seconds']}s")

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
        # Repeated here even though this subprocess doesn't always retrain
        # the VAE: main.py's Step 10 logs vae_hyperparams from its own args
        # regardless of what actually trained it -- without this,
        # results/{run_id}.json would misreport beta/dropout on runs that
        # use a non-default isolated VAE.
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

    print(f"\n  --- Group '{group_name}' ({len(candidates)} candidates x {len(SEEDS)} seeds) ---")

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
                print(f"    [{key}] SKIP (already completed)")
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
                print(f"      OK in {record['elapsed_seconds']}s -- {record['results_json']}")
            else:
                print(f"      FAILED -- {record['error']}")

    if dry_run:
        print(f"    (dry-run: '{group_name}''s winning candidate can't be resolved without real data; "
              f"later groups assume this group's default value.)")
        return None  # signals process_config to keep `resolved` at its current (default) values

    df = load_runs(PHASE2_RESULTS_JSONL)
    df = df[(df.get("config_label") == label) & (df.get("group") == group_name)]
    if df.empty:
        raise RuntimeError(f"No successful runs in group '{group_name}' of '{label}' -- cannot continue.")

    ranking = aggregate_by_config(df, group_by="candidate", metric=RANKING_METRIC)
    wilcoxon_df = pairwise_wilcoxon(df, "candidate", ranking, top_n=min(4, len(ranking)), metric=RANKING_METRIC)
    report(ranking, wilcoxon_df, metric=RANKING_METRIC, top_k=min(3, len(ranking)))

    winner_config_key = ranking.iloc[0]["config"]  # "{group_name}::{cand_label}"
    winner_label = winner_config_key.split("::", 1)[1]
    print(f"  Winner of group '{group_name}': {winner_label}")
    return winner_label


def process_config(combo: List[str], dry_run: bool) -> Dict[str, Any]:
    label = config_label(combo)
    print(f"\n=== Configuration: {label} (extractors: {combo}) ===")

    resolved = initial_resolved()

    # (a) latent space
    winner = run_group(
        combo=combo, label=label, group_name="latent",
        candidates=LATENT_DIM_CANDIDATES, resolved=resolved,
        apply_candidate=lambda r, v: {**r, "latent": v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved["latent"] = LATENT_DIM_CANDIDATES[winner]

    # (a2) VAE regularization (beta / dropout)
    winner = run_group(
        combo=combo, label=label, group_name="vae_reg",
        candidates=VAE_REG_CANDIDATES, resolved=resolved,
        apply_candidate=lambda r, v: {**r, **v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved.update(VAE_REG_CANDIDATES[winner])

    # (b) KAN "inputs" == num_basis
    winner = run_group(
        combo=combo, label=label, group_name="num_basis",
        candidates={str(v): v for v in KAN_NUM_BASIS_CANDIDATES}, resolved=resolved,
        apply_candidate=lambda r, v: {**r, "kan_num_basis": v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved["kan_num_basis"] = int(winner)

    # (c) internal layers/nodes
    winner = run_group(
        combo=combo, label=label, group_name="hidden_dim",
        candidates={str(v): v for v in KAN_HIDDEN_DIM_CANDIDATES}, resolved=resolved,
        apply_candidate=lambda r, v: {**r, "kan_hidden_dim": v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved["kan_hidden_dim"] = int(winner)

    # (d) training parameters
    winner = run_group(
        combo=combo, label=label, group_name="training",
        candidates=KAN_TRAINING_CANDIDATES, resolved=resolved,
        apply_candidate=lambda r, v: {**r, **v},
        dry_run=dry_run,
    )
    if winner is not None:
        resolved.update(KAN_TRAINING_CANDIDATES[winner])

    print(f"\n=== Resolved for {label}: {resolved} ===")
    return resolved


def load_winning_configs(args) -> List[List[str]]:
    if args.configs:
        return json.loads(args.configs)

    winners_path = Path(args.winners) if args.winners else PHASE1_WINNERS_JSON
    if not winners_path.exists():
        raise FileNotFoundError(
            f"{winners_path} not found. Generate it with "
            f"'python scripts/aggregate_results.py --input <phase1.jsonl> --group-by extractors --output-winners' "
            f"or pass --configs '[[...], [...]]' directly."
        )
    with open(winners_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["winners"]


def main():
    parser = argparse.ArgumentParser(description="Phase 2: sequential hyperparameter sweep over Phase 1's top-3")
    parser.add_argument("--winners", default=None, help=f"JSON with the winning configs (default: {PHASE1_WINNERS_JSON})")
    parser.add_argument("--configs", default=None, help="Inline list of combos, e.g. '[[\"semantic\",\"style\"]]'")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = load_winning_configs(args)
    print(f"Phase 2: {len(configs)} winning configurations: {configs}")
    print(f"Results: {PHASE2_RESULTS_JSONL}")

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        for combo in configs:
            results[config_label(combo)] = process_config(combo, dry_run=args.dry_run)

    if args.dry_run:
        print("\ndry-run: plan printed, nothing executed.")
        return

    print("\n=== Phase 2 final summary ===")
    for label, resolved in results.items():
        print(f"  {label}: {resolved}")


if __name__ == "__main__":
    main()
