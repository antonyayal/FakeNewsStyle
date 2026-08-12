Input text + metadata
        │
        ▼
┌───────────────────────────────────────────────────┐
│                 Corpus preparation                 │
│  data/raw/*.xlsx → 01_corpus_pkl → 02_corpus_clean  │
│           (text preprocessing for XLM-R)            │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────┬───────────────┬───────────────┬───────────────┐
│ Semantic      │ Emotion       │ Style         │ Context       │
│ Extractor     │ Extractor     │ Extractor     │ Extractor     │
└──────┬────────┴──────┬────────┴──────┬────────┴──────┬────────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
   sem_emb        emo/sent_probs    style dict      context dict
   (~1024)          + signals        payload          payload
                      (~23)            (~35)           (~103)
       │               │               │               │
       ▼               ▼               ▼               ▼
   VAE_sem          VAE_emo         VAE_sty         VAE_ctx
  z=128 (default)  z=16 (default)  z=16 (default)  z=64 (default)
       │               │               │               │
       └───────────────┴───────┬───────┴───────────────┘
                                ▼
                  Direct concatenation (no shared
                   projection stage — raw VAE latents,
                    prefixed {branch}_latent_i)
                     224 dims total (with defaults)
                                │
                                ▼
                    KAN classifier (PyTorch)
                  RBF-basis KANLayer ×2 → Linear(1)
                                │
                                ▼
                   Fake (1) / True-Real (0) prediction
                                │
                                ▼
                results/{run_id}.json experiment record
