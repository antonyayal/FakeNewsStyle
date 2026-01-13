# Architecture — FakeNewsStyle

## 1) Objetivo

Este proyecto implementa una arquitectura modular para **detección de noticias falsas en español**, integrando múltiples familias de características:

- **Semánticas** (embeddings tipo BERT/RoBERTa-es/BETO)
- **Estilo** (rasgos lingüísticos/estilométricos, n-grams, POS patterns, puntuación, etc.)
- **Emoción/Sentimiento** (lexicons o modelos)
- **Legibilidad** (readability, complejidad, longitud, etc.)
- **Dominio** (memoria por dominio/tema/fuente si aplica)

La arquitectura está diseñada para ser **reproducible y extensible**.

---

## 2) Diagrama lógico (alto nivel)

```mermaid
flowchart TD
  A[Input: text and optional metadata] --> B[Preprocessing]
  B --> B1[Cleaning and normalization]
  B --> B2[Tokenization]
  B --> B3[Train-val-test split]

  B --> C[Feature extraction]
  C --> C1[Semantic features: BETO or RoBERTa-es]
  C --> C2[Style features: stylometry and syntax]
  C --> C3[Emotion features: lexicon or model]
  C --> C4[Readability features: indices and stats]
  C --> C5[Domain features: memory or clustering optional]

  C1 --> D[Fusion and representation]
  C2 --> D
  C3 --> D
  C4 --> D
  C5 --> D

  D --> D1[Concatenation plus MLP]
  D --> D2[Attention or gating optional]
  D --> D3[Memory augmented fusion optional]

  D1 --> E[Classifier head]
  D2 --> E
  D3 --> E

  E --> F[Prediction: fake or real plus score]
  E --> G[Evaluation]
  G --> G1[Metrics: Acc Prec Rec F1 AUC]
  G --> G2[Error analysis]
  G --> G3[Ablation study]

  F --> H[Inference and explainability]
  H --> H1[SHAP or LIME optional]
  H --> H2[Feature importance for baselines]
```

# Arquitectura
![Architecture Diagram](architecture_overview.svg)