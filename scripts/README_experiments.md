# Orquestación de experimentos (Fase 1 a Fase 5)

Sistema de 5 fases sobre `main.py`, con checkpointing/resume y logging
centralizado en JSON-lines. Ver `scripts/experiment_config.py` para editar
semillas y valores candidatos.

Archivos:
- `experiment_config.py` — semillas fijas (`SEEDS`), candidatos por
  hiperparámetro, rutas de resultados.
- `experiment_runner.py` — mecánica compartida: lanzar `main.py`, capturar
  su `results/{run_id}.json`, apendear una línea al JSON-lines de la fase,
  y entrenar/mergear las VAEs que cada fase necesita (`ensure_vae_latents`
  para el cache compartido, `resolve_kan_input`/`merge_latents_manual`
  para VAEs aisladas cuando `vae_beta`/`vae_dropout` no son el default).
- `aggregate_results.py` — agregación (media ± std) + Wilcoxon pareado por
  semilla; reusable como CLI o importado desde `orchestrator_phase{1,2,3}.py`.
- `orchestrator_phase1.py` — dimensión latente por extractor, en solitario.
- `orchestrator_phase2.py` — combinaciones de extractores.
- `orchestrator_phase3.py` — VAE-reg + hiperparámetros KAN, fusionados.
- `fold_validation.py` — motor compartido de validación por folds (usado por
  `orchestrator_phase4.py` y `orchestrator_phase5.py`, que solo difieren en
  `--corpus_mode`).
- `orchestrator_phase4.py` — robustez de partición (k-fold normal).
- `orchestrator_phase5.py` — robustez a leakage por medio (folds source-disjoint).

Las 5 fases usan el mismo CLI: `--run` (ejecuta, resumible), `--dry-run`
(combinado con `--run`, solo imprime los comandos), `--summary` (reagrega el
JSONL existente sin relanzar nada).

## Fase 1 — dimensión latente por extractor

```bash
source venv/bin/activate
python scripts/orchestrator_phase1.py --run --dry-run   # revisar el plan (85 corridas)
python scripts/orchestrator_phase1.py --run              # ejecutar de verdad
```

Un solo extractor activo a la vez (`--exclude_*` en los otros 3), variando
su dimensión latente sobre `PHASE1_DIM_CANDIDATES` (17 combinaciones:
5 semantic + 3 emotion + 4 style + 5 context) x 5 seeds = 85 corridas. Cada
par (branch, dim) entrena su VAE una sola vez vía
`experiment_runner.ensure_vae_latents` y la reusa en sus 5 seeds.

No compara extractores entre sí (un modelo de un solo extractor rinde peor
que cualquier combo — eso lo decide la Fase 2) — solo rankea *dimensiones
dentro del mismo extractor*, para fijar la mejor dim de cada branch antes de
probar combinaciones.

Resultados: `results/orchestrator_phase1.jsonl` (una línea por corrida) y,
al terminar (o con `--summary`), `results/phase1_top.json` con el top 2 de
dimensión por cada uno de los 4 branches.

## Fase 2 — combinaciones de extractores

```bash
python scripts/orchestrator_phase2.py --run --dry-run   # revisar el plan (75 corridas)
python scripts/orchestrator_phase2.py --run
```

Requiere `results/phase1_top.json`. Los 15 combos on/off de {semantic,
emotion, style, context}, cada branch activo usando su dimensión rank-1 de
la Fase 1 (no se cruzan las 2 dimensiones candidatas) x 5 seeds = 75
corridas.

El ranking final no compite solo entre estos 15 combos: se suman las 8
entradas de la Fase 1 (top-2 x 4 branches, ya evaluadas, no se re-corren) al
mismo pool, y se toma el top 5 global. Salida: `results/phase2_top.json`.

## Fase 3 — VAE-reg + hiperparámetros KAN, fusionados

```bash
python scripts/orchestrator_phase3.py --run --dry-run   # revisar el plan (450 corridas)
python scripts/orchestrator_phase3.py --run
```

Requiere `results/phase2_top.json`. Para cada uno de los 5 configs
ganadores, corre las 18 variantes de `PHASE3_CANDIDATES` (un baseline —
los defaults de `main.py` — más una perilla movida a la vez: 4 `vae_beta`,
3 `vae_dropout`, 3 `kan_num_basis`, 4 `kan_hidden_dim` (8/16/32/128, más
64 como baseline — cubre todo el rango 8–128), 2 `kan_weight_decay`; más
una variante combinada, `combo_basis8_hidden16_wdhigh`, que mueve
`kan_num_basis`, `kan_hidden_dim` y `kan_weight_decay` a la vez) x 5 seeds
= 450 corridas. `kan_lr`/`kan_batch_size` no se sweepean — un sweep
anterior ya encontró que ganan en su default.

**VAE compartida vs. aislada**: `--merge_vae_latents` de `main.py` siempre
lee de la ruta fija `data/05_vae_latents/{branch}/latent{dim}/`. Las
variantes con `vae_beta=1.0, vae_dropout=0.1` (el default) reusan ese cache
compartido. Las que cambian beta/dropout entrenan una VAE aislada bajo
`data/05_vae_latents_phase3/` / `models/vae_phase3/` y la mergean
manualmente (`experiment_runner.resolve_kan_input` /
`merge_latents_manual`), apuntando `--train_kan` a esos PKLs vía
`--kan_train_pkl`/`--kan_val_pkl`/`--kan_test_pkl`.

Ranking: 5 configs x 18 variantes = 90 combinaciones → top 5 global,
totalmente resuelto (extractores + dims + los 5 hiperparámetros). Salida:
`results/phase3_top.json`.

## Fase 4 y Fase 5 — validación por folds

```bash
python scripts/orchestrator_phase4.py --run --dry-run   # k-fold normal (125 corridas)
python scripts/orchestrator_phase4.py --run

python scripts/orchestrator_phase5.py --run --dry-run   # folds source-disjoint (125 corridas)
python scripts/orchestrator_phase5.py --run
```

Ambas requieren `results/phase3_top.json` y son **ramas independientes**
(ninguna depende de la otra, las dos parten del mismo top 5 de la Fase 3).
No exploran nada nuevo: repiten cada uno de los 5 configs ganadores across
5 folds x 5 seeds = 125 corridas cada una, vía `--corpus_mode kfold`
(Fase 4) o `--corpus_mode source_disjoint` (Fase 5) — ver el docstring de
`main.py` y `src/data/kfold_corpus.py` / `src/data/source_split_corpus.py`.

La Fase 5 es el test definitivo del leakage por `Source` documentado en el
README principal: si el F1 colapsa hacia el techo de la ablación
identity-free incluso con las 4 modalidades activas, confirma que el número
alto de F1 era en buena parte memorización del medio, no señal genuina de
estilo/semántica.

Cada fase produce dos archivos: `phase{4,5}_per_fold.json` (detalle
completo por config x fold, media/std/min/max sobre las 5 seeds) y
`phase{4,5}_top.json` (un solo ganador global: el config con mejor F1
promedio combinando los 5 folds x 5 seeds, no un ganador por fold).

## Manejo de errores

Si una corrida de `main.py` falla (código de salida != 0, o no imprime la
línea `Experiment record saved: ...`, o el JSON no se puede leer), se
escribe una línea con `"status": "failed"` y `"error": "<detalle>"` en el
JSONL, y el batch **continúa** con la siguiente corrida — nunca se detiene
por un solo fallo. Al reanudar (correr el mismo comando `--run` otra vez),
las corridas fallidas o interrumpidas a mitad (que nunca llegaron a escribir
línea) se reintentan automáticamente — no hay flag especial de resume.

## Dónde queda todo

- `results/orchestrator_phase{1..5}.jsonl` — logs centralizados, uno por fase.
- `results/phase{1,2,3}_top.json`, `results/phase{4,5}_per_fold.json` +
  `results/phase{4,5}_top.json` — salidas agregadas de cada fase.
- `results/{run_id}.json` — un registro por corrida individual (los escribe
  `main.py` vía `src/experiments/run_logger.py`, no se toca).
- `data/07_kan_runs/phase{1..5}/...` — checkpoints/métricas por corrida.
- `data/05_vae_latents_phase3/`, `models/vae_phase3/`,
  `data/06_vae_latents_merged_phase3/` — VAEs aisladas y sus latentes
  mergeados manualmente, solo para variantes de Fase 3 con
  `vae_beta`/`vae_dropout` distinto al default. Se pueden borrar sin afectar
  el cache compartido de `data/05_vae_latents/` / `models/vae/`.
- `data/06_vae_latents_merged_cv/` (Fase 4) y
  `data/06_vae_latents_merged_source_cv/` (Fase 5) — latentes mergeados por
  fold, namespaced por `{entry_label}` ya que 5 configs distintos se validan
  sobre los mismos folds.

## Costo esperado

Fase 1: 85 corridas. Fase 2: 75. Fase 3: 450. Fase 4: 125. Fase 5: 125.
Total: 860 corridas KAN-only (más el entrenamiento puntual de las VAEs que
falten en cada cache). Orden de segundos-minutos por corrida KAN-only en
GPU/CPU del servidor; el arranque de cada subproceso `python main.py ...`
importa torch/tensorflow/transformers/spaCy sin importar qué flag se use
(~7s medidos), así que el overhead de arranque, no el entrenamiento, domina
el tiempo total en fases con muchas corridas cortas (Fase 3 sobre todo).
