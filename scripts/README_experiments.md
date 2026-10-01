# Orquestación de experimentos (Opción A — 3 fases)

Plan de 3 fases sobre `main.py`, con checkpointing/resume y logging
centralizado en JSON-lines. Editar `scripts/experiment_config.py` para
cambiar el alcance (semillas, dims candidatas de `context`, grid de
`hidden_dim`, hiperparámetros fijos) sin tocar la lógica de los
orquestadores.

## Idea

Las fases 3–6 del plan anterior (sweep de hiperparámetros de VAE/KAN,
k-fold normal, control identity-free de `context`) se colapsaron:

- **Hiperparámetros de VAE/KAN fijos.** El sweep viejo (1 350 corridas,
  `results_old_3/orchestrator_phase3.jsonl`) mostró que ninguna perilla
  supera a los defaults de `main.py` por más que el ruido de semilla; solo
  `vae_beta=4.0` claramente empeoraba. Quedan fijos en
  `experiment_config.FINAL_HPARAMS`. La única excepción es `kan_hidden_dim`
  (32 vs 64 fue un volado), que se mantiene como único eje de barrido
  (`HIDDEN_DIM_GRID`).
- **K-fold normal eliminado.** Coincidía con el split estándar (no ve el
  leakage de `Source`), así que no aporta sobre la Fase 2.
- **`context` identity-free integrado.** Ya no es una fase aparte: cada
  combo que incluye `context` se corre dos veces, con `Source Name`/`Source Link`
  encendidos y apagados, en las Fases 2 y 3.

Archivos:
- `experiment_config.py` — constantes (`SEEDS`, `FINAL_HPARAMS`,
  `HIDDEN_DIM_GRID`, `PHASE1_CONTEXT_ON_DIMS`/`_OFF_DIMS`, rutas).
- `experiment_plan.py` — primitivas compartidas: los 23 combo-variants
  (`build_prep_units`), el grid KAN (`iter_kan_entries`), labels para
  `run_key`/dirs, el comando KAN (`build_kan_cmd`), y la extracción + VAE
  de `context` identity-free (`ensure_idfree_context`, la única rama que
  aún necesita cache aislada ahora que `vae_beta`/`vae_dropout` son fijos).
- `experiment_runner.py` — mecánica compartida: lanzar `main.py`, capturar
  su `results/{run_id}.json`, apendear una línea al JSONL de la fase,
  entrenar VAEs faltantes (`ensure_vae_latents`) y mergear latentes
  (`merge_latents_manual`).
- `aggregate_results.py` — agregación (media ± std) + Wilcoxon pareado por
  semilla; CLI y usado por la Fase 1.
- `orchestrator_phase{1,2,3}.py` — las 3 fases.

Las 3 fases usan el mismo CLI: `--run` (ejecuta, resumible), `--dry-run`
(combinado con `--run`, solo imprime los comandos), `--summary` (reagrega
el JSONL existente sin relanzar nada).

## Fase 1 — dimensión latente de `context`, por modo de identidad

```bash
source venv/bin/activate
python scripts/orchestrator_phase1.py --run --dry-run   # revisar el plan (27 corridas)
python scripts/orchestrator_phase1.py --run
```

Solo se barre `context` (los otros 3 branches usan los defaults de
`main.py`: `semantic=128, emotion=16, style=16` — el sweep viejo mostró que
la dim dentro de un branch mueve el F1 menos que el ruido de semilla).

Dos sub-sweeps de `context` solo (`--exclude_*` en los otros 3):
- **identity ON** — `Source Name`/`Source Link` encendidos (defaults de `main.py`),
  dims `PHASE1_CONTEXT_ON_DIMS` = `[8, 16, 32, 64, 86]`.
- **identity OFF** — `--context_source_name_dim 0 --context_source_link_dim 0`, dims
  `PHASE1_CONTEXT_OFF_DIMS` = `[4, 8, 16, 23]` (capado en la dim cruda
  identity-free de 23).

(5 + 4) dims × 3 semillas = **27 corridas**. Cada par (modo, dim) entrena
un VAE y lo reusa en sus 3 semillas. El ranking elige la mejor dim *dentro
de cada modo* — los dos modos nunca se comparan aquí.

Salida: `results/orchestrator_phase1.jsonl` y `results/phase1_top.json`
(`{context_on: {dim, ...}, context_off: {dim, ...}}` + top-2 por modo). Las
Fases 2 y 3 lo leen; si no existe, usan los fallbacks de
`experiment_config` (`FALLBACK_CONTEXT_ON_DIM=64`, `_OFF_DIM=16`).

## Fase 2 — evaluación en split estándar

```bash
python scripts/orchestrator_phase2.py --run --dry-run   # 138 corridas
python scripts/orchestrator_phase2.py --run
```

El resultado in-distribution. Todas las combinaciones de extractores, sin
filtrar:
- las 15 combinaciones no vacías de {semantic, emotion, style, context};
- cada combo con `context` se corre 2 veces (identidad ON / OFF);
- cada uno de esos **23 combo-variants** a los 2 valores de `kan_hidden_dim`;
- × 3 semillas.

= 23 × 2 × 3 = **138 corridas KAN** sobre el split fijo train/val/test.

Por combo-variant se preparan los latentes una sola vez (branches no-context
e identidad-ON desde el cache compartido `data/05_vae_latents/`; `context`
identity-OFF desde su cache aislado `data/05_vae_latents_idfree/`), se
concatenan en PKLs KAN-ready bajo `data/06_vae_latents_merged_optA/{label}/`,
y encima corre el grid `hidden_dim × semilla` como puro entrenamiento KAN.

Salida: `results/orchestrator_phase2.jsonl` y `results/phase2_top.json` (los
23 combo-variants ranqueados por F1 de validación, con métricas de test al
lado).

## Fase 3 — validación source-disjoint

```bash
python scripts/orchestrator_phase3.py --run --dry-run   # 690 corridas
python scripts/orchestrator_phase3.py --run
```

El resultado con control de leakage. Mismos 23 combo-variants × 2
`hidden_dim` × 3 semillas × **5 folds source-disjoint** = **690 corridas** —
particionando con `--corpus_mode source_disjoint` para que ningún medio
(`Source`) aparezca en más de un train/val/test de un fold
(`StratifiedGroupKFold` + `GroupShuffleSplit` agrupado por `Source`, ver
`src/data/source_split_corpus.py`).

El contraste **Fase 2 vs Fase 3** cuantifica cuánto del F1 del split
estándar era memorización de medio y no señal genuina de estilo/semántica
(ver "Known Limitations & Caveats" en el README principal).

Por (combo-variant, fold): una pasada de preparación construye el corpus /
features / VAEs de ese fold y los concatena en PKLs KAN-ready
(branches no-context + `context` identidad-ON en el cache compartido
`data/05_vae_latents_source_cv/` que namespacea `main.py` por fold;
`context` identidad-OFF en su cache aislado por fold
`data/05_vae_latents_source_cv_idfree/fold{k}/`); luego el grid
`hidden_dim × semilla` como puro KAN. Un `_marker.json` por (fold, combo-
variant) permite que un `--run` reanudado salte la preparación ya hecha.

Salida: `results/orchestrator_phase3.jsonl`, `results/phase3_per_fold.json`
(detalle por combo-variant × fold) y `results/phase3_top.json` (los 23
ranqueados, media/std sobre folds × semillas — sin colapsar a un solo
ganador).

## Correr las 3 fases

```bash
source venv/bin/activate
nohup bash -c '
  python scripts/orchestrator_phase1.py --run &&
  python scripts/orchestrator_phase2.py --run &&
  python scripts/orchestrator_phase3.py --run
' > logs/overnight_run.out 2>&1 &
disown
```

`&&`-encadenado para que un fallo detenga el resto; cada fase es resumible
por su cuenta (re-lanzar el mismo `--run` salta lo que ya terminó ok).
`nohup ... & disown` sobrevive al cierre de la sesión SSH — seguir con
`tail -f logs/overnight_run.out`.

**Total: 27 + 138 + 690 = 855 corridas KAN**, más el entrenamiento puntual
de VAEs faltantes (los branches no-context ya están en el cache compartido;
lo nuevo es `context` identity-free, ~9 VAEs para el split estándar y ~5 por
fold). Orden de segundos-minutos por corrida KAN; el arranque de cada
subproceso `python main.py ...` (~7 s importando torch/tf/transformers/
spaCy) domina el total.

## Manejo de errores

Si una corrida de `main.py` falla (código != 0, o no imprime
`Experiment record saved: ...`, o el JSON no se puede leer) se escribe una
línea `"status": "failed"` en el JSONL y el batch **continúa**. Al reanudar
(mismo comando `--run`), las corridas fallidas o interrumpidas a mitad se
reintentan automáticamente — no hay flag especial de resume.

Los `run_key` codifican todo lo que varía dentro de una fase
(`{combo}__ctx{on|off}__hd{hd}__seed{s}`, más `__fold{k}` en la Fase 3), así
que reanudar nunca salta una corrida que debería ser distinta. Si cambias
`FINAL_HPARAMS` o `HIDDEN_DIM_GRID`, empieza con un JSONL nuevo (los
`run_key` no guardan valores absolutos de hiperparámetro).

## Dónde queda todo

- `results/orchestrator_phase{1,2,3}.jsonl` — logs centralizados, uno por fase.
- `results/phase1_top.json`, `results/phase2_top.json`,
  `results/phase3_per_fold.json` + `results/phase3_top.json` — salidas agregadas.
- `results/{run_id}.json` — un registro por corrida (lo escribe `main.py`).
- `data/07_kan_runs/phase{1,2,3}/...` — checkpoints/métricas por corrida.
- `data/06_vae_latents_merged_optA/` (Fase 2) y
  `data/06_vae_latents_merged_source_cv_optA/` (Fase 3) — latentes mergeados
  manualmente, KAN-ready.
- `data/05_vae_latents_idfree/`, `models/vae_idfree/` (split estándar) y
  `data/05_vae_latents_source_cv_idfree/`, `models/vae_source_cv_idfree/`
  (por fold) — VAEs de `context` identity-free. Se pueden borrar sin tocar
  el cache compartido.
- `results_old_3/` — el plan viejo (6 fases, 5 semillas) y sus ~2 480
  registros de corrida, movidos aquí al migrar. `.gitignore` lo cubre.
