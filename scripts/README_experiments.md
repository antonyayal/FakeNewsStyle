# Orquestación de experimentos (Fase 1 + Fase 2)

Sistema de dos fases sobre `main.py`, con checkpointing/resume y logging
centralizado en JSON-lines. Ver `scripts/experiment_config.py` para editar
semillas y valores candidatos.

Archivos:
- `experiment_config.py` — semillas fijas, candidatos por hiperparámetro, rutas.
- `experiment_runner.py` — mecánica compartida: lanzar `main.py`, capturar
  su `results/{run_id}.json`, apendear una línea al JSON-lines de la fase.
- `orchestrator_phase1.py` — 15 combos de expertos x 10 semillas = 150 corridas.
- `aggregate_results.py` — agregación (media ± std) + Wilcoxon pareado por
  semilla; reusable como CLI o importado desde `orchestrator_phase2.py`.
- `orchestrator_phase2.py` — sweep secuencial de 4 grupos de hiperparámetros
  sobre las 3 configuraciones ganadoras de la Fase 1.

## Fase 1

```bash
source venv/bin/activate
python scripts/orchestrator_phase1.py --dry-run     # revisar el plan primero (150 corridas)
python scripts/orchestrator_phase1.py                # ejecutar de verdad
```

Antes del sweep, entrena el VAE por defecto **una sola vez** (si no existe
ya en `data/05_vae_latents/`) y lo reusa en las 150 corridas — el VAE de
cada rama es independiente del combo de expertos activos y no tiene semilla
configurable, así que reentrenarlo por corrida solo metería ruido no
controlado. Cada corrida individual solo hace
`--merge_vae_latents --train_kan` con `--exclude_*` y `--kan_seed`.

Resultados: `results/orchestrator_phase1.jsonl` (una línea por corrida, con
`active_extractors`, `seed`, y todos los campos de `metrics.test` que ya
escribe `src/experiments/run_logger.py`, incluyendo `mcc`, `n_params`,
`train_time_sec`).

**Reanudar tras una caída**: correr exactamente el mismo comando
(`python scripts/orchestrator_phase1.py`). Al arrancar, el script lee el
JSONL existente y salta cualquier `run_key` (combo+semilla) que ya tenga
`status: "ok"`. No hay flag especial de resume — es el comportamiento por
defecto.

## Agregar resultados de la Fase 1 y elegir el top 3

```bash
python scripts/aggregate_results.py \
    --input results/orchestrator_phase1.jsonl \
    --group-by extractors \
    --output-winners            # sin valor -> escribe results/phase1_top3.json
```

Imprime la tabla de ranking (ordenada por F1 medio ± std sobre las 10
semillas), el test de Wilcoxon signed-rank pareado por semilla entre las
configuraciones consecutivas más cercanas al top, y señala el top 3. Con
`--output-winners` también escribe `results/phase1_top3.json`, el archivo
que `orchestrator_phase2.py` espera por defecto.

## Fase 2

```bash
python scripts/orchestrator_phase2.py --dry-run      # revisar el plan del grupo (a); (b)(c)(d) son provisionales
python scripts/orchestrator_phase2.py                 # usa results/phase1_top3.json por defecto
# o, sin pasar por Fase 1 / aggregate_results.py:
python scripts/orchestrator_phase2.py --configs '[["semantic","emotion"], ["semantic","style","context"]]'
```

Para cada configuración ganadora, corre en orden fijo y **decide en runtime**
qué lanzar después de cada grupo (llama a `aggregate_results.aggregate_by_config`
para elegir el candidato con mejor F1 medio antes de generar el siguiente):

1. `latent` — dimensión del espacio latente (presets `small`/`default`/`large`,
   solo sobre las ramas activas de la config; entrena VAE si falta).
2. `vae_reg` — regularización del VAE, `--vae_beta` / `--vae_dropout` (candidatos
   en `VAE_REG_CANDIDATES`). Se agregó porque una corrida histórica
   (`results/20260812_022815_d3ede7c8.json`, vía el antiguo
   `scripts/run_full_stack_sweep.py`) midió más alto que cualquier combo de la
   Fase 1 solo bajando beta a 0.25 — ninguna otra parte del sweep exploraba esa
   dimensión.
3. `num_basis` — "entradas" del KAN, `--kan_num_basis` (no existe un flag de
   input_dim separado; el input real ya está cubierto por el grupo `latent`).
4. `hidden_dim` — nodos intermedios del KAN, `--kan_hidden_dim`.
5. `training` — `--kan_lr` / `--kan_batch_size` / `--kan_weight_decay`, una
   perilla a la vez sobre el baseline.

**Sobre `vae_reg` y el cache compartido de VAE**: `--merge_vae_latents` de
`main.py` siempre lee de la ruta fija `data/05_vae_latents/{branch}/latent{dim}/`
— la misma que usa la Fase 1 y el resto de la Fase 2 — así que entrenar ahí
con un beta/dropout distinto al default la corrompería para todo lo demás.
Por eso, cualquier candidato de `vae_reg` que **no** coincida con
`DEFAULT_VAE_REG` (`beta=1.0`, `dropout=0.1`, los defaults de `main.py`) se
entrena en un directorio aislado (`data/05_vae_latents_phase2/`,
`models/vae_phase2/`) y se mergea manualmente en Python (mismo patrón que
`scripts/run_full_stack_sweep.py`), apuntando `--train_kan` a esos PKLs vía
`--kan_train_pkl`/`--kan_val_pkl`/`--kan_test_pkl` en vez de
`--merge_vae_latents`. El candidato `"default"` de `vae_reg`, y cualquier
grupo posterior mientras `vae_reg` siga resuelto en su default, siguen
reusando el cache compartido normal.

Resultados: `results/orchestrator_phase2.jsonl`, mismo esquema y mismo
mecanismo de resume que la Fase 1 (`run_key` incluye config+grupo+candidato+
semilla, así que reanudar es volver a correr el mismo comando).

En `--dry-run`, solo el grupo `latent` refleja candidatos reales; los grupos
`vae_reg`/`num_basis`/`hidden_dim`/`training` se imprimen asumiendo los
valores por defecto de `main.py` para los grupos previos (no se pueden
resolver sin datos reales).

## Manejo de errores

Si una corrida de `main.py` falla (código de salida != 0, o no imprime la
línea `Experiment record saved: ...`, o el JSON no se puede leer), se
escribe una línea con `"status": "failed"` y `"error": "<detalle>"` en el
JSONL, y el batch **continúa** con la siguiente corrida — nunca se detiene
por un solo fallo. Al reanudar, las corridas fallidas (o interrumpidas a
mitad, que nunca llegaron a escribir línea) se reintentan automáticamente.

## Dónde queda todo

- `results/orchestrator_phase1.jsonl`, `results/orchestrator_phase2.jsonl` — logs centralizados.
- `results/phase1_top3.json` — top 3 combos de la Fase 1 (entrada de la Fase 2).
- `results/{run_id}.json` — un registro por corrida individual (los escribe `main.py`, no se toca).
- `data/07_kan_runs/phase1/{combo}/seed{N}/`, `data/07_kan_runs/phase2/{combo}/{grupo}/{candidato}/seed{N}/` — checkpoints/métricas por corrida.
- `data/05_vae_latents_phase2/`, `models/vae_phase2/`, `data/06_vae_latents_merged_phase2/` — VAEs aislados y sus latentes mergeados manualmente, solo para candidatos de `vae_reg` con beta/dropout distinto al default (ver más arriba). Se pueden borrar sin afectar el cache compartido de `data/05_vae_latents/` / `models/vae/`.

## Costo esperado

Fase 1: 150 corridas KAN-only (tras el prep de VAE, único). Fase 2, peor
caso: 3 configs x (3 + 5 + 4 + 4 + 7 candidatos) x 10 semillas = 690 corridas,
la mayoría KAN-only salvo los grupos `latent` y `vae_reg` (que sí pueden
reentrenar VAE por rama/preset — `vae_reg` una vez por cada uno de sus 4
candidatos no-default, ya que `"default"` reusa lo que dejó `latent`). Orden
de segundos-minutos por corrida KAN-only en GPU/CPU del servidor.
