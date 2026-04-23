Input text + metadata
        │
        ▼
┌────────────────────────────────────┐
│         Preprocessing stage        │
│  XLSX → PKL → text preprocessing   │
└────────────────────────────────────┘
        │
        ▼
┌───────────────┬───────────────┬───────────────┬───────────────┐
│ Semantic      │ Emotion       │ Style         │ Context       │
│ Extractor     │ Extractor     │ Extractor     │ Extractor     │
└──────┬────────┴──────┬────────┴──────┬────────┴──────┬────────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
   sem_emb         emotion vec      style vec      context vec
   (1024)             (~23)           (~35)          (103)
       │               │               │               │
       ▼               ▼               ▼               ▼
   VAE_sem          VAE_emo         VAE_sty         VAE_ctx
    z=96             z=12            z=16            z=32
       │               │               │               │
       ▼               ▼               ▼               ▼
 h_sem (16)       h_emo (16)      h_sty (16)      h_ctx (16)
       └───────────────┬───────────────┬───────────────┘
                       ▼
             Concatenated shared latent
                       (64 dims)
                       │
                       ▼
              KAN / downstream classifier
                       │
                       ▼
                 Fake / Real prediction