Perfecto — mantengo tu estructura original pero la extiendo con todo lo nuevo (extractores, PCA, VAE, ejecución, etc.) sin romper el estilo académico 👇

---

````md
# FakeNewsStyle  
**Style-aware Architecture for Fake News Detection in Spanish**

> 🚧 **Work in Progress**  
> This repository is currently **under active development** as part of a **PhD research project**.  
> The architecture, experiments, and results are subject to change as the doctoral study progresses.

---

## 📌 Overview

**FakeNewsStyle** is a research-oriented framework for **fake news detection in Spanish**, designed with a strong emphasis on **stylistic features** and their interaction with semantic, emotional, readability, and domain-based representations.

The system follows a **multi-feature modular pipeline**, where different aspects of the text are modeled independently and later integrated into a unified representation.

This project is being developed as part of a **doctoral research study**, and therefore represents an **evolving research artifact** rather than a finalized production system.

---

## 🎯 Research Motivation

Most fake news detection approaches rely primarily on **semantic representations** derived from large pre-trained language models. However, deceptive content often exhibits **distinctive writing styles**, emotional patterns, and structural cues that are **complementary to semantics**.

This research explores:

- The contribution of **stylistic features** to fake news detection  
- The effectiveness of **multi-feature modular architectures**  
- The role of **emotional and contextual signals**  
- The interaction between semantic and non-semantic representations  
- The use of **latent compression (VAE)** for heterogeneous feature spaces  

---

## 🧠 Proposed Architecture

The architecture follows a **multi-branch pipeline**:

### 🔹 Stage 1: Feature Extraction

Each input sample is processed through independent extractors:

- **Semantic extractor**
  - XLM-RoBERTa embeddings (≈1024 dims)
- **Emotion extractor**
  - Emotion probabilities  
  - Sentiment probabilities  
  - Emotional signals  
- **Style extractor**
  - Readability, formality, syntactic complexity  
  - Lexical diversity  
  - POS distributions  
  - Stylometric signals  
- **Context extractor**
  - Source, domain, topic embeddings  
  - Metadata features (age, presence flags)

---

### 🔹 Stage 2: Latent Compression (VAE per extractor)

Each feature space is compressed independently:

| Extractor | Input Dim | Latent Dim |
|----------|----------|-----------|
| Semantic | 1024     | 96        |
| Emotion  | ~23      | 12        |
| Style    | ~35      | 16        |
| Context  | 103      | 32        |

---

### 🔹 Stage 3: Shared Latent Projection

Each latent vector is projected into a common space:

```text
z_sem (96) → 16
z_emo (12) → 16
z_sty (16) → 16
z_ctx (32) → 16
````

---

### 🔹 Stage 4: Fusion and Classification

```text
[h_sem | h_emo | h_sty | h_ctx] → 64 dims
```

Final representation is passed to:

* KAN-based classifier (planned)
* or baseline classifier

---

### 📊 Architecture Diagram

See:

```
docs/architecture.md
```

---

## 📂 Project Structure

```
FakeNewsStyle/
├── configs/
├── data/
│   ├── raw/
│   ├── processed_to_PKL/
│   ├── processed_by_model/
│   ├── features/
│   │   ├── semantic/
│   │   ├── emotion/
│   │   ├── style/
│   │   ├── context/
│   │   └── merged/
│   └── latent/              # (planned) VAE outputs
├── docs/
├── logs/
├── reports/
├── runs/
├── scripts/
│   ├── inspect_pkl.py
│   ├── inspect_features.py
│   └── pca_latent_dim_suggester.py
├── src/
│   ├── data/
│   ├── text/
│   ├── features/
│   └── models/              # (planned) VAE / KAN
├── main.py
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/antonyayal/FakeNewsStyle.git
cd FakeNewsStyle

python -m venv venv
source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
pip install matplotlib scikit-learn
```

---

## ▶️ Execution Guide

### 🔹 Full pipeline (recommended)

```bash
python main.py --prepare_corpus 1
python main.py --preprocess_text 1

python main.py --extract_semantic 1
python main.py --extract_emotion 1
python main.py --extract_style 1
python main.py --extract_context 1

python main.py --merge_features 1
```

---

### 🔹 Individual extractors

#### Semantic

```bash
python main.py --extract_semantic 1
```

#### Emotion

```bash
python main.py --extract_emotion 1
```

#### Style

```bash
python main.py --extract_style 1
```

#### Context

```bash
python main.py --extract_context 1
```

---

### 🔹 Inspect generated features

```bash
python scripts/inspect_features.py \
  --pkl data/features/semantic/FakeNewsCorpusSpanish/train_semantic.pkl
```

---

### 🔹 PCA analysis for latent dimension

```bash
python scripts/pca_latent_dim_suggester.py \
  --pkl \
    data/features/semantic/FakeNewsCorpusSpanish/train_semantic.pkl \
    data/features/emotion/FakeNewsCorpusSpanish/train_emotion.pkl \
    data/features/style/FakeNewsCorpusSpanish/train_style.pkl \
    data/features/context/FakeNewsCorpusSpanish/train_context.pkl
```

Outputs:

* optimal latent dimensions
* variance explained
* PCA plots (`reports/pca_latent_dims/`)

---

## 🧪 Experimental Pipeline

Current experimental flow:

```text
raw data
   ↓
feature extraction
   ↓
PCA analysis
   ↓
VAE compression (planned)
   ↓
projection to shared space
   ↓
fusion
   ↓
classification
```

---

## 📊 Feature Normalization

* **Style features**

  * normalized per type (ratios, log scaling, tanh)
* **Context features**

  * scaled to [-1,1]
* **Semantic**

  * optionally L2 normalized
* **Emotion**

  * probabilistic outputs already bounded

---

## 🔬 Research Contributions (in progress)

* Multi-feature fake news detection in Spanish
* Style-aware representation learning
* Modular latent compression using VAE
* Shared latent space alignment
* Future: evidential and KAN-based fusion

---

## 🚧 Project Status

This repository is **under construction** and part of an ongoing **PhD dissertation**.

* Some modules are experimental or incomplete
* VAE and fusion modules are in development
* APIs and internal design may change
* Results are preliminary

---

## 📜 License

Apache 2.0 License

---

## ✉️ Contact

**Jose Antonio Ayala-Barbosa**
PhD Student – Computer Science
UNAM / IIMAS

