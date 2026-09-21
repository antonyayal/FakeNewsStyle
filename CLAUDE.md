# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FakeNewsStyle — Spanish fake news detection research (PhD project), emphasizing stylistic features alongside semantic, emotional, and contextual signals. Prioritize clarity and reproducibility over premature optimization; this is a research codebase, not production software.

## Environment

- Runs on a university SSH server. A venv already exists at `venv/` (Python 3.12.3) — activate with `source venv/bin/activate` before running anything.
- Key pinned deps: `torch` 2.11, `tensorflow`/`keras` 2.21/3.14 (VAEs are Keras, KAN classifier is PyTorch — both are used, not redundant), `transformers` 5.6, `spacy` 3.8 (`es_core_news_sm` model), `pysentimiento` 0.7.3, `scikit-learn` 1.8.
- Install: `pip install -r requirements.txt` (note: the file lists `torch` twice — a plain entry and a `--index-url .../cu121` entry for CUDA wheels).

## Running the pipeline

Everything is driven by `main.py` via boolean flags (`action="store_true"` — pass the flag to enable a step, omit it to skip); flags are independent `if` blocks executed top-to-bottom in one process, so they can be combined in a single invocation. **`README.md`'s "Execution Guide" is out of date** — it references `--merge_features`, which doesn't exist. The actual end-to-end sequence (folder names match the `# Step N` comments in `main.py`):

```bash
python main.py --prepare_corpus                 # data/raw/*.xlsx -> data/01_corpus_pkl/*.pkl
python main.py --preprocess_text                 # -> data/02_corpus_clean/*.pkl (adds text_xlmr, label)

python main.py --extract_semantic --extract_emotion --extract_style --extract_context
                                                  # -> data/03_features_raw/{semantic,emotion,style,context}/{split}_*.pkl

python main.py --merge_raw_features               # -> data/04_features_merged/{split}.pkl (raw baseline, pre-VAE)

python main.py --run_vaes                         # trains one VAE per branch -> models/vae/{branch}/latent{dim}/
                                                  #   + data/05_vae_latents/{branch}/latent{dim}/{split}.pkl

python main.py --merge_vae_latents                 # -> data/06_vae_latents_merged/{split}.pkl (KAN-ready)

python main.py --train_kan                        # trains KAN classifier + evaluates -> data/07_kan_runs/merged/
                                                  #   + logs a run record to results/{run_id}.json
```

Per-branch VAE latent dims and KAN hyperparameters are configurable via flags (`--semantic_latent_dim`, `--emotion_latent_dim`, `--style_latent_dim`, `--context_latent_dim`, `--kan_hidden_dim`, `--kan_num_basis`, `--vae_beta`, `--kan_epochs`, etc.) — run `python main.py --help` for the full list. To exclude a modality from VAE training and the KAN input entirely (not just skip re-extracting it), pass `--exclude_semantic`/`--exclude_emotion`/`--exclude_style`/`--exclude_context`.

Each `--train_kan` run writes a JSON record to `results/` (active extractors, latent dims, epochs, hyperparams, full metrics, git commit, a snapshot of the KAN checkpoint) via `src/experiments/run_logger.py`. Run `python scripts/report_builder.py` to compile all `results/*.json` into `reports/experiments_summary.csv`, an extractor-combo heatmap, and per-run weight histograms.

The full evaluation is driven by a 3-phase plan of orchestrators over `main.py` (`scripts/orchestrator_phase{1,2,3}.py`, shared config in `scripts/experiment_config.py` / `scripts/experiment_plan.py`, mechanics in `scripts/experiment_runner.py`): **Phase 1** sweeps only the `context` VAE latent dim, per identity mode (Source/Domain ON vs. OFF); **Phase 2** evaluates all 15 extractor combos (the 8 with `context` doubled for identity ON/OFF) × `kan_hidden_dim` ∈ {32,64} × 3 seeds on the standard split; **Phase 3** repeats those 23 combo-variants across 5 source-disjoint folds. All other VAE/KAN hyperparameters are fixed in `FINAL_HPARAMS` (an old 1350-run sweep found nothing beating `main.py`'s defaults by more than seed noise). 27 + 138 + 690 = 855 runs. Full guide: `scripts/README_experiments.md`. The previous 6-phase plan and its ~2480 run records are in `results_old_3/` (gitignored).

Standalone inspection utilities live in `scripts/` (e.g. `inspect_pkl.py` for deep PKL inspection, `pca_latent_dim_suggester.py` for choosing VAE latent dims from explained variance, `peek_pkl_row.py --pkl <path> --row <i>` for a quick single-row peek).

## Architecture

Four independent feature branches, each compressed by its own VAE into a shared latent space, then fused for classification:

```
raw xlsx -> 01_corpus_pkl -> 02_corpus_clean (text_xlmr)
    -> {semantic, emotion, style, context} extractors  (03_features_raw)
        -> per-branch VAE (β-VAE, Keras)                (05_vae_latents)
            -> latent merge (concat, prefixed h_{branch}_*)  (06_vae_latents_merged)
                -> KAN classifier (PyTorch)               (07_kan_runs)
```

- **Semantic** (`src/features/semantic_extractor.py`): `FacebookAI/xlm-roberta-large-finetuned-conll02-spanish`, mean/CLS/attention pooling (`AttentionPooling` is untrained/random-init unless the whole model is fine-tuned — don't treat it as meaningful unless that changes), optional L2-normalize. Output: DataFrame with `sem_emb` (list column).
- **Emotion** (`src/features/emotion_extractor.py`): pysentimiento (`emotion`+`sentiment`, Spanish) probability vectors plus hand-built lexical signal ratios (exclamation/uppercase/emoji/etc.). Output: DataFrame with `emo_probs`/`sent_probs`/`signals` list columns.
- **Style** (`src/features/style_extractor.py`): spaCy/textstat/wordfreq stylometry (readability, formality, syntactic complexity, lexical diversity, POS ratios, OOV rate, punctuation/burstiness signals). Output is a **dict payload** (`{X, feature_names, ids, meta}`), not a DataFrame — different schema from semantic/emotion. `--style_feature_version` (default `1` = the original 35 features, unchanged; `2` = +12 extra lexical/surface features → 47: expressive-punctuation-per-word, interrobang, ALL-CAPS words, degree adverbs, superlatives, hyperbolic quantifiers, urgency/alarm cues, rhetorical-question ratio, speech verbs, connector density + narrative-vs-argumentative contrast, intrigue verbs; verb groups use spaCy lemmas when available; `3` = the `future_steps/style_extractor_revision.md` rework: prune v1 to its 16 keepers + `sig_excl_char_ratio` + Credibility (`cred_*`), Interactivity (`inter_*`), Drama (`intr_*`), Readability-dispersion (`read_*`) scalar facets → 36 features, **not** a superset of v1/v2, and loads spaCy with NER enabled). `meta["feature_version"]` records which was used; v2/v3 each need their own re-extract + style VAE retrain (isolate with `--style_output_dir` / `--style_vae_input_dir` / `--results_dir`).
- **Context** (`src/features/context_extractor.py`): deterministic hash embeddings (no training) for Source/Domain/Topic/Author + article age. Also a **dict payload**, same schema as style.
- **VAE** (`src/models/train_vae_from_pkl.py`): one β-VAE per branch (`reconstruction_loss + beta * kl_loss`), Dense+Dropout encoder/decoder mirrored around `hidden_dims`. Saves `encoder.keras`, `decoder.keras`, `vae_final.weights.h5`, `scaler.joblib` per branch/latent-dim, plus latent PKLs with columns `{branch}_latent_{i}`.
- **KAN** (`src/models/kan.py`): `KANLayer` uses a fixed RBF basis (`linspace(-3,3,num_basis)`, shared learnable width) rather than splines — not a literal Kolmogorov-Arnold spline network. `KANClassifier` = `KANLayer -> LayerNorm -> SiLU -> Dropout` ×2 `-> Linear(1)`, trained with `BCEWithLogitsLoss`/AdamW/early stopping via `train_kan_from_pkls`.
- **Label convention**: `1 = Fake`, `0 = True/Real` — consistent across `kan.py`, `merge_raw_features_for_kan.py`, and `src/evaluation/metrics.py`.
- **Two merge paths exist for raw features** — `src/features/feature_merger.py` (Id-aligned, dict-payload output, `sem_/emo_/sty_/ctx_` prefixes) is **not called by `main.py`**; the wired-in one is `src/features/merge_raw_features_for_kan.py` (row-order concat, one-hot expansion for categoricals, column-aligned across splits). Don't assume `feature_merger.py` is live without checking `main.py` first.
- **Metrics** (`src/evaluation/metrics.py`): `evaluate_binary_classifier` (accuracy, ROC/PR-AUC, ECE, Brier, entropy, confusion matrix, assumes `y_prob` = P(Fake)); `save_metrics` writes both `{prefix}.json` and `{prefix}.csv` — this is the source of the `train/val/test_metrics.{json,csv}` files under `data/07_kan_runs/`.
- **Experiment logging** (`src/experiments/run_logger.py`): writes one `results/{run_id}.json` per `--train_kan` run plus a checkpoint snapshot in `results/checkpoints/` (so it survives a later run reusing the same `--kan_output_dir`). `scripts/report_builder.py` reads all of `results/*.json` to build comparison tables/heatmaps/weight histograms in `reports/` (gitignored). Unrelated to the old `run_experiment.py` skeleton, which has been removed.

## Conventions

- English for comments/docs and code/identifiers. Every `.py` file should open with a module-level docstring summarizing what it does (goal, inputs/outputs, usage) -- see `src/features/semantic_extractor.py` for the template.
- `pathlib.Path` everywhere; scripts resolve `BASE_DIR = Path(__file__).resolve().parent` and `sys.path.insert(0, str(BASE_DIR))`.
- Extractor logging: `get_logger(log_dir, name)` writes console + a timestamped file `{log_dir}/{name}_{YYYY-MM-DD_HH-MM-SS}.log` (see `logs/features/*`, `logs/prepare_corpus/*`). Follow this pattern for any new extractor/stage.
- Feature PKL naming: `data/03_features_raw/{branch}/{split}_{branch}.pkl`; VAE latent PKLs at `data/05_vae_latents/{branch}/latent{dim}/{split}.pkl`; VAE model checkpoints at `models/vae/{branch}/latent{dim}/` (only the dim matching `main.py`'s current default is kept — `semantic=128, emotion=16, style=16, context=64` — stale sweep dirs get pruned).
- `.gitignore` excludes `*.pkl`, `*.npy`, `*.pt`, and (typo) `*.hi5` — this does **not** match `*.h5`, so VAE `.weights.h5` files and `.keras`/xlsx model artifacts end up tracked in git (visible as modified in `git status` after every VAE/KAN run). Be aware of this when committing — don't assume model binaries are gitignored.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
