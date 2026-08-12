# scripts/run_homologated_experiment.py
# =====================================================
# One-off comparison: does mixing news sources across train/val/test
# (data/raw/{train,development,test}_homologated.xlsx) fix the train->test
# accuracy gap found in results/batches/20260811_235144.json?
#
# That diagnosis (see reports/comparison_20260811_235144/SUMMARY.md and the
# domain-shift analysis done in-session) found: 87% of test Sources never
# appear in train, and within train, Source almost perfectly predicts the
# label (satire sites are ~100% Fake, mainstream outlets ~0% Fake) -- so the
# original split lets a KAN "cheat" by fingerprinting known sources, a
# shortcut that can't transfer to test's disjoint source roster.
# *_homologated.xlsx are a re-shuffled version of the same 1548 articles
# across splits (same Ids overall, different train/val/test assignment)
# meant to mix sources across splits and remove that shortcut.
#
# This script runs the full corpus->features->VAE->KAN chain against the
# homologated xlsx, entirely isolated from the default pipeline's
# directories (data/02_corpus_clean, data/03_features_raw, models/vae/ --
# git-tracked or otherwise reused by other experiments) so it never
# overwrites anything from the existing baseline. Two KAN configs are
# trained for direct comparison against known numbers from the old split:
#   - semantic+style, winning batch-1 hyperparams -> compare vs 0.6923/0.6868
#   - all_extractors, main.py defaults           -> compare vs 0.6573/0.5882
#
# Usage:
#   python scripts/run_homologated_experiment.py
# =====================================================

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.text.preprocess_text import preprocess_pkl_for_model, get_logger as get_text_logger
from src.features.semantic_extractor import extract_semantic_features_for_splits
from src.features.emotion_extractor import extract_emotion_features_for_splits
from src.features.style_extractor import StyleExtractor, StyleExtractorConfig
from src.features.context_extractor import ContextExtractor, ContextExtractorConfig
from src.models.train_vae_from_pkl import train_vae_from_paths
from src.models.kan import train_kan_from_pkls
from src.experiments.run_logger import log_experiment_result, hash_files, MODALITY_ORDER
from src.evaluation.metrics import evaluate_binary_classifier, save_metrics, compute_topic_breakdown

RAW_DIR = BASE_DIR / "data" / "raw"
SCRATCH_DIR = Path("/tmp/claude-1001/-home-antonio-Projects-FakeNewsStyle/fbd1d9ea-f7ac-4ea0-bcda-f29bdae1f4ee/scratchpad/homologated_raw_pkl")

CLEAN_DIR = BASE_DIR / "data" / "02_corpus_clean_homologated"
RAW_FEATURES_DIR = BASE_DIR / "data" / "03_features_raw_homologated"
VAE_DATA_DIR = BASE_DIR / "data" / "05_vae_latents_homologated"
VAE_MODEL_DIR = BASE_DIR / "models" / "vae_homologated"
MERGED_DIR = BASE_DIR / "data" / "06_vae_latents_merged_homologated"

LATENT_DIMS = {"semantic": 128, "emotion": 16, "style": 16, "context": 64}
HIDDEN_DIMS = {"semantic": [512, 256], "emotion": [128, 64], "style": [128, 64], "context": [256, 128]}

XLSX_FILES = {
    "train": "train_homologated.xlsx",
    "val": "development_homologated.xlsx",
    "test": "test_homologated.xlsx",
}


def _load_pkl_any(path: Path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_texts_and_ids_from_obj(obj, text_col: str, id_col: str | None = None):
    texts = obj[text_col].astype(str).tolist()
    ids = obj[id_col].tolist() if (id_col and id_col in obj.columns) else None
    return texts, ids


def _extract_labels_from_obj(obj, label_col: str = "label"):
    if label_col in obj.columns:
        return obj[label_col].tolist()
    return None


# =====================================================
# Step A: xlsx -> homologated-label corpus_clean pkls
# =====================================================
def build_corpus_clean() -> None:
    print("=" * 80)
    print("Step A: building homologated corpus_clean (label homologation applied)")
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = BASE_DIR / "logs" / "preprocess_homologated"

    for split, fname in XLSX_FILES.items():
        xlsx_path = RAW_DIR / fname
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        tmp_pkl = SCRATCH_DIR / f"{split}_raw.pkl"
        df.to_pickle(tmp_pkl)

        out_path = CLEAN_DIR / f"{split}.pkl"
        preprocess_pkl_for_model(input_pkl=tmp_pkl, output_pkl=out_path, logger=get_text_logger(log_dir))
        print(f"  {fname} -> {out_path} ({len(df)} rows)")


# =====================================================
# Step B: feature extraction (isolated output dirs)
# =====================================================
def extract_features() -> None:
    print("=" * 80)
    print("Step B: extracting semantic/emotion/style/context features")

    for branch in ["semantic", "emotion", "style", "context"]:
        (RAW_FEATURES_DIR / branch).mkdir(parents=True, exist_ok=True)

    extract_semantic_features_for_splits(
        input_dir=CLEAN_DIR,
        output_dir=RAW_FEATURES_DIR / "semantic",
        log_dir=BASE_DIR / "logs" / "features_homologated" / "semantic",
        pooling="mean",
        device="cpu",
        batch_size=8,
        max_len=256,
    )
    print("  semantic done")

    extract_emotion_features_for_splits(
        input_dir=CLEAN_DIR,
        output_dir=RAW_FEATURES_DIR / "emotion",
        log_dir=BASE_DIR / "logs" / "features_homologated" / "emotion",
        text_col="text_xlmr",
        batch_size=32,
        device="cpu",
        use_preprocess_tweet=False,
        normalize_signals_by="chars",
        extra_signals=True,
    )
    print("  emotion done")

    style_extractor = StyleExtractor(StyleExtractorConfig(spacy_model="es_core_news_sm"))
    ctx_extractor = ContextExtractor(ContextExtractorConfig())

    for split_name in ["train", "val", "test"]:
        in_path = CLEAN_DIR / f"{split_name}.pkl"
        obj = _load_pkl_any(in_path)

        texts, ids = _extract_texts_and_ids_from_obj(obj, "Text", "Id")
        labels = _extract_labels_from_obj(obj)
        style_extractor.save_features_pkl(
            texts=texts, ids=ids, labels=labels,
            output_path=RAW_FEATURES_DIR / "style" / f"{split_name}_style.pkl",
            batch_size=64,
            metadata={"dataset": "FakeNewsCorpusSpanish_homologated", "split": split_name, "source_pkl": str(in_path)},
            log_dir=BASE_DIR / "logs" / "features_homologated" / "style",
            log_name=f"style_{split_name}.log",
        )

        df = pd.read_pickle(in_path)
        ids2 = df["Id"].tolist()
        labels2 = df["label"].tolist()
        rows = df.to_dict(orient="records")
        ctx_extractor.save_features_pkl(
            rows=rows, ids=ids2, labels=labels2,
            output_path=RAW_FEATURES_DIR / "context" / f"{split_name}_context.pkl",
            metadata={"dataset": "FakeNewsCorpusSpanish_homologated", "split": split_name, "source_pkl": str(in_path)},
            log_dir=BASE_DIR / "logs" / "features_homologated" / "context",
            log_name=f"context_{split_name}.log",
        )
    print("  style + context done")


# =====================================================
# Step C: VAE training (isolated output dirs, models/vae/ untouched)
# =====================================================
def train_vaes() -> None:
    print("=" * 80)
    print("Step C: training per-branch VAEs on homologated data")

    for branch, latent_dim in LATENT_DIMS.items():
        feature_cols = ["emo_probs", "sent_probs", "signals"] if branch == "emotion" else None
        train_pkl = RAW_FEATURES_DIR / branch / f"train_{branch}.pkl"
        val_pkl = RAW_FEATURES_DIR / branch / f"val_{branch}.pkl"
        test_pkl = RAW_FEATURES_DIR / branch / f"test_{branch}.pkl"

        train_vae_from_paths(
            train_pkl=train_pkl, val_pkl=val_pkl, test_pkl=test_pkl,
            feature_column=feature_cols, label_column=None,
            latent_dim=latent_dim, hidden_dims=HIDDEN_DIMS[branch],
            dropout=0.1, beta=1.0, epochs=100, batch_size=32, learning_rate=1e-3,
            output_data_dir=VAE_DATA_DIR / branch / f"latent{latent_dim}",
            output_model_dir=VAE_MODEL_DIR / branch / f"latent{latent_dim}",
            feature_name=branch,
        )
        print(f"  VAE {branch} (latent{latent_dim}) done")


# =====================================================
# Step D: merge VAE latents (mirrors main.py Step 9, isolated paths)
# =====================================================
def merge_latents(active_extractors: list[str]) -> dict[str, Path]:
    latent_dirs = {name: VAE_DATA_DIR / name / f"latent{LATENT_DIMS[name]}" for name in active_extractors}
    out_paths = {}

    for split in ["train", "val", "test"]:
        dfs = []
        labels = None
        for feature_name, feature_dir in latent_dirs.items():
            df = pd.read_pickle(feature_dir / f"{split}.pkl")
            if "label" in df.columns:
                current_labels = df["label"].reset_index(drop=True)
                if labels is None:
                    labels = current_labels
                df = df.drop(columns=["label"])
            df = df.reset_index(drop=True)
            df.columns = [c if str(c).startswith(f"{feature_name}_") else f"{feature_name}_{c}" for c in df.columns]
            dfs.append(df)

        merged_df = pd.concat(dfs, axis=1)
        if labels is not None:
            merged_df["label"] = labels.values

        combo_tag = "_".join(active_extractors)
        out_dir = MERGED_DIR / combo_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.pkl"
        merged_df.to_pickle(out_path)
        out_paths[split] = out_path

    return out_paths


# =====================================================
# Step E: train + log a KAN run
# =====================================================
def run_kan(label: str, active_extractors: list[str], kan_kwargs: dict) -> None:
    print("=" * 80)
    print(f"Step E: training KAN [{label}] extractors={active_extractors} kwargs={kan_kwargs}")

    paths = merge_latents(active_extractors)
    output_dir = BASE_DIR / "data" / "07_kan_runs" / f"homologated_{label}"

    kan_result = train_kan_from_pkls(
        train_pkl=str(paths["train"]), val_pkl=str(paths["val"]), test_pkl=str(paths["test"]),
        output_dir=str(output_dir), **kan_kwargs,
    )

    metrics = {}
    for split_name, split_preds in kan_result["predictions"].items():
        m = evaluate_binary_classifier(split_preds["y_true"], split_preds["y_prob"])
        metrics[split_name] = m
        save_metrics(m, output_dir, prefix=f"{split_name}_metrics")

    dataset_hash = hash_files([paths["train"], paths["val"], paths["test"]])

    # Positional join against the homologated test corpus (same assumption as
    # main.py's Step 10: row order is preserved end-to-end, never reshuffled).
    test_corpus_df = pd.read_pickle(CLEAN_DIR / "test.pkl")
    topic_breakdown = compute_topic_breakdown(
        y_true=kan_result["predictions"]["test"]["y_true"],
        y_prob=kan_result["predictions"]["test"]["y_prob"],
        topics=test_corpus_df["Topic"].tolist(),
    ) if "Topic" in test_corpus_df.columns else None

    record_path = log_experiment_result(
        active_extractors=active_extractors,
        latent_dims=LATENT_DIMS,
        vae_epochs_requested=100,
        kan_epochs_requested=kan_kwargs["epochs"],
        kan_epochs_run=kan_result.get("epochs_run", kan_kwargs["epochs"]),
        vae_hyperparams={"batch_size": 32, "learning_rate": 1e-3, "beta": 1.0, "dropout": 0.1},
        kan_hyperparams={
            "hidden_dim": kan_kwargs["hidden_dim"], "num_basis": kan_kwargs["num_basis"],
            "dropout": kan_kwargs["dropout"], "batch_size": kan_kwargs["batch_size"],
            "lr": kan_kwargs["lr"], "weight_decay": kan_kwargs["weight_decay"],
            "patience": kan_kwargs["patience"], "seed": kan_kwargs["seed"],
        },
        metrics=metrics,
        kan_output_dir=output_dir,
        kan_checkpoint_path=Path(kan_result["best_model_path"]),
        vae_model_dirs={m: VAE_MODEL_DIR / m / f"latent{LATENT_DIMS[m]}" for m in active_extractors},
        base_dir=BASE_DIR,
        training_time_seconds=kan_result.get("training_time_seconds"),
        num_parameters=kan_result.get("num_parameters"),
        dataset_hash=dataset_hash,
        topic_breakdown=topic_breakdown,
    )

    print(f"  test accuracy={metrics['test']['accuracy']:.4f} f1={metrics['test']['f1']:.4f} "
          f"roc_auc={metrics['test']['roc_auc']:.4f}")
    print(f"  record: {record_path}")


def main():
    build_corpus_clean()
    extract_features()
    train_vaes()

    # Config 1: semantic+style, exact winning hyperparams from batch 20260811_235144
    run_kan(
        label="semantic_style",
        active_extractors=["semantic", "style"],
        kan_kwargs=dict(hidden_dim=32, num_basis=8, dropout=0.5, epochs=50, batch_size=32,
                         lr=1e-3, weight_decay=1e-3, patience=5, seed=42),
    )

    # Config 2: all_extractors, main.py CLI defaults
    run_kan(
        label="all_extractors",
        active_extractors=MODALITY_ORDER,
        kan_kwargs=dict(hidden_dim=64, num_basis=16, dropout=0.2, epochs=100, batch_size=32,
                         lr=1e-3, weight_decay=1e-4, patience=15, seed=42),
    )

    print("=" * 80)
    print("Done. Compare against results/batches/20260811_235144.json (semantic_style: 0.6923/0.6868)")
    print("and the all-extractors default sanity run for the old split (0.6573/0.5882).")


if __name__ == "__main__":
    main()
