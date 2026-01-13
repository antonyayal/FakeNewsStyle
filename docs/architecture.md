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
  A[Input: texto + metadatos opcionales] --> B[Preprocessing]
  B --> B1[Limpieza/Normalización]
  B --> B2[Tokenización]
  B --> B3[Split: train/val/test]
  
  B --> C[Feature Extraction Layer]
  C --> C1[Semantic Features<br/>BERT/RoBERTa-es/BETO]
  C --> C2[Style Features<br/>Estilometría / POS / n-grams / puntuación]
  C --> C3[Emotion Features<br/>Lexicons/Modelos]
  C --> C4[Readability Features<br/>Índices + stats]
  C --> C5[Domain Features (opcional)<br/>Memoria/Clusters/Topic]

  C1 --> D[Fusion / Representation]
  C2 --> D
  C3 --> D
  C4 --> D
  C5 --> D

  D --> D1[Concatenation / MLP]
  D --> D2[Attention / Gating (opcional)]
  D --> D3[M3FEND Fusion + Memory (opcional)]

  D1 --> E[Classifier Head]
  D2 --> E
  D3 --> E

  E --> F[Predicción: Fake / Real<br/>+ Scores]
  E --> G[Evaluación]
  G --> G1[Métricas: Acc, Prec, Rec, F1, AUC]
  G --> G2[Error Analysis]
  G --> G3[Ablation Study]
  
  F --> H[Inference / Explainability]
  H --> H1[SHAP/LIME (opcional)]
  H --> H2[Feature Importance (baselines)]
