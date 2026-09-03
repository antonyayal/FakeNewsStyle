# Style extractor — revision plan

**Date:** 2026-09-01
**Status:** proposed, not yet implemented
**Inputs:** `style_importance_results/{permutation_importance,redundant_pairs,correlation_matrix}.csv`
(RandomForest on style-only features, permutation importance = ROC-AUC drop, 30 repeats,
75/25 split) + Yang et al., IJCAI 2019, *How to Write High-quality News on Social Network*.

## Context / motivation

- Style-only pipeline (VAE → KAN) currently tops out at **F1 ≈ 0.73–0.74**, **ROC-AUC ≈ 0.72**.
- Corpus is balanced (~50/50; 681 train / 295 val / 572 test), so F1 is essentially
  AUC-limited → the lever is **better features**, not threshold tuning.
- Permutation importances are tiny (max 0.013, `std` > `mean` for many features), so the
  **"importance ≤ 0" cuts must be confirmed with a KAN ablation**, not the isolated RF alone.
  The **redundancy** cuts (corr ≥ 0.89) are solid on their own.

## Notes on the Yang et al. paper

- **Different task**: predicts news *quality* (= popularity/engagement), not *veracity*.
  Use its **feature definitions**, never its weights or conclusions.
- **Effect direction flips**: their "good" (high Sensation, exclamation marks, rhetoric)
  ≈ our **Fake**. Do not import the idea that "more `!` = better".
- Their 99.6% classification accuracy = **publisher-style leakage** (VERY GOOD = People's Daily
  + CCTV vs TYPICAL = Xinhua ×2 — really outlet identification). Same Source→label problem
  already flagged for this project. Their honest number is the Intra-User analysis
  (per-feature SRC ≤ 0.35; final model SRC 0.606).
- Chinese / HanLP-specific — needs Spanish reimplementation (spaCy `es_core_news_sm` + lexicons).
- Their high-level features are **crude unnormalized sums**
  (`Formality = Noun + Adj + Prep − Pron − Verb − Adv − Sentence broken`). Keep the individual
  features, normalized per token/sentence, and let the KAN weight them.

---

## A. REMOVE

### A1. Redundant — safe cut (corr ≥ 0.89)

| Remove | Reason | Keep |
|---|---|---|
| `ttr` | corr −1.0 with `redundancy`, 0.92 with `herdans_c` | `root_ttr` |
| `redundancy` | literally `1 − ttr` | `root_ttr` |
| `herdans_c` | same construct (lexical diversity) | `root_ttr` |
| `sig_long_word_ratio` | corr 0.89 with `sig_avg_word_len` **and** importance −0.003 | none (see A2) |
| `pos_verb_ratio` | corr −0.89 with `formality_f`, lower importance | `formality_f` |

### A2. Importance ≤ 0 in isolation — cut after confirming with a KAN ablation

`sig_avg_word_len`, `pos_adj_ratio`, `pos_pron_ratio`, `len_sent`,
`error_rate` (OOV proxy is mis-calibrated — **fix or cut**), `avg_dep_depth`,
`sig_ellipsis_per_sent`, `sig_repeated_char_ratio`, `sig_percent_ratio`,
`sig_quotes_ratio`, `sig_intensifier_ratio`.

### A3. Replace (good concept, poor implementation)

- `sig_proper_like_ratio` (PROPN ratio) → replace with the named-entity features in C1.
- `sig_q_per_sent` / `sig_excl_per_sent` (importance ~0 here, but `!` is a classic fake-news
  signal) → collapse into a single `sig_excl_char_ratio` (`!` per character); move `?` to the
  Interactivity facet (C2).

**Net A: ~35 features → ~16 scalars.**

---

## B. KEEP (features with some signal)

`root_ttr`, `sig_burstiness`, `verbs_per_sent`, `formality_f`, `sconj_per_sent`,
`pos_noun_ratio`, `pos_adv_ratio`, `pos_det_ratio`, `pos_adp_ratio`,
`sig_stopword_ratio`, `sig_uppercase_ratio`, `sig_digit_ratio`, `ifsz`,
`sig_hedge_ratio`, `sig_se_per_sent`, `sig_punct_ratio`.

---

## C. IMPLEMENT

### C1. Credibility facet — top priority (not currently present; best path to higher F1)

All normalized per token or per sentence:

- `cred_numeral_ratio` — numeric tokens (digits + spelled-out numbers) / tokens
- `cred_entity_ratio` — `len(doc.ents)` / tokens
- `cred_when_ratio` — `DATE` / `TIME` entities / sentences
- `cred_where_ratio` — `LOC` / `GPE` entities / sentences
- `cred_who_ratio` — `PER` / `ORG` entities / sentences
- `cred_quote_density` — paired quotes + reported-speech verbs
  (`dijo, afirmó, aseguró, declaró, señaló, informó`) / sentences
- `cred_attribution_ratio` — source markers
  (`según, de acuerdo con, fuentes, el comunicado, informó`) / sentences
- `cred_uncertainty_ratio` — strong hedging + conditional
  (`presuntamente, supuestamente, al parecer, habría, podría`) / tokens —
  **separate** from `sig_hedge_ratio`
- `cred_url_present` — 0/1 if the raw text contains `http` / a link

### C2. Interactivity facet (new)

- `inter_first_person_ratio` — 1st-person pronouns + verb morphology `-mos` / tokens
- `inter_second_person_ratio` — `tú, usted(es), vosotros` + `-s/-áis` / tokens
- `inter_interrogative_ratio` — `qué, cómo, por qué, cuándo, dónde, quién` as interrogatives / sentences
- `inter_question_ratio` — `?` / sentences (moved from A3)

### C3. Interestingness / drama facet (new — watch overlap with the `emotion` branch)

- `intr_adversative_ratio` — `pero, sin embargo, no obstante, aunque` / sentences
- `intr_scare_quote_ratio` — short quoted phrases / sentences
- **Do NOT** duplicate emoji / sentiment — already in `emotion`.

### C4. Readability — dispersion (cheap, improves what's already measured)

- `read_sent_len_cv` — coefficient of variation of sentence length (better than raw `sig_burstiness`)
- `read_pct_short_sent` — % sentences < 8 words
- `read_pct_long_sent` — % sentences > 30 words
- `read_complex_word_ratio` — words ≥ 3 syllables / words

### C5. Vector blocks — the big AUC lever (separate sub-blocks, not individual features)

- **Function-word profile**: relative frequency of ~150 closed-class words
  (prepositions, conjunctions, determiners, pronouns, high-frequency adverbs) →
  `TruncatedSVD` to ~32 dims. Classic topic-independent stylometry.
- **POS n-grams**: bi/tri-grams over the spaCy POS sequence, TF-IDF top ~300 → SVD ~24 dims.
- **Char n-grams (3–5)**: TF-IDF `min_df=5, max_features≈3000` → SVD ~64 dims.
  ⚠️ starts mixing style with topic — decide whether that is acceptable for the thesis framing.

---

## D. Structural change (required if C5 is added)

The style vector goes from ~16 to ~150+ dims. Then either:

- raise `--style_latent_dim` (try 32–64), **or**
- concat `[C1–C4 scalars | function-word SVD | POS SVD | char SVD]` into one vector and let the
  VAE compress it, **or**
- bypass the VAE for style and use a GBM / linear model directly.

## E. Validation after changes

1. Re-run `scripts/style_feature_importance.py --pkl data/03_features_raw/style/train_style.pkl`.
2. KAN ablation (with / without each new block) on **both** the normal split **and** the
   source-disjoint Phase 4 split — separates real style signal from source memorization.
3. Report style-only F1 mean±std over the 5 seeds on both splits.

## Expected outcome

Char n-grams + linear/GBM should push style-only to **AUC ~0.78–0.82, F1 ~0.77–0.80**.
Beating `context` / `emotion` (F1 ~0.87) with style alone is unlikely on this corpus.
