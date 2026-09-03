# Guía del proyecto FakeNewsStyle (para alguien que nunca usó NLP)

> Este documento asume que sabes programar (Python, estructuras de datos, algo de
> ML clásico) pero **nunca trabajaste con NLP** (procesamiento de lenguaje
> natural), embeddings, VAEs ni clasificadores tipo red neuronal. Está escrito
> para que puedas entender de punta a punta qué hace este proyecto, por qué,
> y cómo se configura cada pieza.

---

## 1. ¿Qué problema resuelve este proyecto?

Es un proyecto de doctorado (UNAM/IIMAS) sobre **detección de noticias falsas
en español**. La pregunta de investigación central no es solo "¿es fake o
no?", sino: **¿cuánto ayuda el *estilo de escritura* (aparte del contenido
semántico) a detectar noticias falsas?**

Por eso el sistema no usa un solo modelo de texto, sino **4 "ramas" o
modalidades** independientes, cada una capturando un aspecto distinto del
artículo:

| Rama | Qué mide | Ejemplo de señal |
|---|---|---|
| **Semantic** (semántica) | De qué habla el texto | Embeddings de un modelo de lenguaje grande (XLM-RoBERTa) |
| **Emotion** (emoción) | Tono emocional | Probabilidades de alegría/enojo/miedo, uso de mayúsculas, signos de exclamación |
| **Style** (estilo) | Cómo está escrito, no qué dice | Legibilidad, diversidad léxica, ratios gramaticales |
| **Context** (contexto) | Metadatos del artículo | Medio (`Source`), dominio web, tema (`Topic`), antigüedad |

Cada rama se procesa por separado, se comprime a un espacio "latente" más
pequeño (con un VAE), y luego todas las ramas comprimidas se concatenan y se
pasan a un clasificador final (KAN) que decide `Fake` (1) o `Real` (0).

---

## 2. Conceptos NLP mínimos para entender el resto

Si ya conoces estos términos, salta a la sección 3.

- **Embedding**: convertir texto (o una categoría, como el nombre de un
  medio) en un vector de números. La idea es que textos "parecidos"
  terminen con vectores parecidos. En este proyecto hay dos tipos de
  embeddings muy distintos:
  - **Embeddings aprendidos por un modelo de lenguaje** (rama *semantic*):
    salen de una red neuronal (XLM-RoBERTa) entrenada previamente por
    terceros con muchísimo texto. Capturan significado.
  - **Embeddings por hashing** (rama *context*): no hay ningún entrenamiento
    involucrado. Se toma un string (ej. el nombre de un medio) y se le
    aplica una función hash determinística para mapearlo a un vector de
    tamaño fijo. Es un truco barato para convertir categorías con muchos
    valores posibles (cientos de medios distintos) en vectores sin tener
    que aprender un vocabulario. Ver sección 4.4.

- **Pooling**: un modelo de lenguaje como XLM-RoBERTa te da un vector *por
  cada palabra/token* del texto (una matriz `[tokens, dims]`). Pero
  necesitamos **un solo vector por artículo**. "Pooling" es la operación que
  reduce esa matriz a un vector: promediar todos los tokens (`mean`), tomar
  solo el primer token especial (`cls`), o aprender pesos de importancia por
  token (`attention`).

- **Espacio latente / latent space**: un espacio vectorial de menor
  dimensión que "resume" la información original. Si el embedding semántico
  original tiene 1024 dimensiones, un espacio latente de 128 dimensiones
  intenta conservar lo esencial con 8 veces menos números.

- **VAE (Variational Autoencoder / autoencoder variacional)**: una red
  neuronal que aprende a comprimir un vector de entrada a un vector latente
  pequeño (*encoder*) y luego reconstruirlo de vuelta (*decoder*). Se
  entrena para que la reconstrucción se parezca al original. El truco
  "variacional" es que en vez de aprender un punto fijo en el espacio
  latente, aprende una distribución (media + varianza) — esto regulariza el
  espacio latente para que sea más suave y útil. Ver sección 5.

- **Clasificador**: al final, un modelo que toma el vector combinado
  (semántico+emocional+estilístico+contextual, ya comprimidos) y produce una
  probabilidad de que el artículo sea falso. Aquí se usa un "KAN" (ver
  sección 6).

- **One-hot encoding**: convertir una categoría (ej. "Deportes", "Política")
  en un vector donde solo una posición vale 1 y el resto 0. Se usa en el
  merge de features crudas (Paso 7), no en las ramas VAE.

- **Split train/val/test**: los datos se dividen en tres conjuntos —
  *train* (para entrenar), *val*/*development* (para ajustar
  hiperparámetros y decidir cuándo detener el entrenamiento) y *test* (para
  medir el desempeño final, nunca usado en entrenamiento).

---

## 3. El pipeline completo, de punta a punta

Todo se ejecuta desde `main.py`, con **flags booleanos independientes**
(`--flag` activa ese paso, omitirlo lo salta). Los pasos son bloques `if`
secuenciales, así que se pueden combinar en una sola invocación.

```
data/raw/*.xlsx                          (Excel original)
    │  --prepare_corpus
    ▼
data/01_corpus_pkl/*.pkl                 (convertido a pickle)
    │  --preprocess_text
    ▼
data/02_corpus_clean/*.pkl               (agrega columna text_xlmr + label)
    │  --extract_semantic / --extract_emotion / --extract_style / --extract_context
    ▼
data/03_features_raw/{branch}/{split}_{branch}.pkl   (features crudas por rama)
    │  --merge_raw_features   (opcional, baseline pre-VAE, NO usado por el resto del pipeline)
    ▼
data/04_features_merged/{split}.pkl
    │
    │  --run_vaes  (entrena 1 VAE por rama sobre 03_features_raw)
    ▼
models/vae/{branch}/latent{dim}/          (encoder, decoder, pesos, scaler)
data/05_vae_latents/{branch}/latent{dim}/{split}.pkl   (vectores latentes)
    │  --merge_vae_latents
    ▼
data/06_vae_latents_merged/{split}.pkl    (todas las ramas concatenadas, listo para KAN)
    │  --train_kan
    ▼
data/07_kan_runs/{output_dir}/            (checkpoint + métricas train/val/test)
results/{run_id}.json                     (registro completo del experimento)
```

Comando típico de punta a punta:

```bash
source venv/bin/activate

python main.py --prepare_corpus
python main.py --preprocess_text
python main.py --extract_semantic --extract_emotion --extract_style --extract_context
python main.py --merge_raw_features
python main.py --run_vaes
python main.py --merge_vae_latents
python main.py --train_kan
```

O todo junto en una sola llamada (cada flag es independiente):

```bash
python main.py --extract_semantic --extract_emotion --extract_style --extract_context \
  --run_vaes --merge_vae_latents --train_kan
```

**Nota importante:** el `README.md` del repo tiene una "Execution Guide"
desactualizada (menciona `--merge_features`, que no existe). La secuencia de
arriba, tomada directo de `main.py`, es la correcta.

### 3.1 Modos de corpus (`--corpus_mode`)

Además del split fijo original, el pipeline soporta repartir los datos de
otras formas — esto es clave para las Fases 3 y 4 explicadas en la sección 8:

- `original` (default): el split fijo train/development/test tal como viene
  el corpus.
- `kfold`: junta train+development+test y los reparte en `--kfold_n` folds
  estratificados por label (NO por medio/`Source`).
- `source_disjoint`: igual que kfold, pero garantiza que ningún medio
  (`Source`) aparezca en más de un split del mismo fold — esto es la
  corrección directa al problema de fuga de datos que se explica en la
  sección 9.

Cambiar `--corpus_mode` redirige automáticamente todos los directorios de
salida a un árbol paralelo (`*_cv` o `*_source_cv`), así que nunca se pisan
los artefactos de un modo con otro.

---

## 4. Los 4 extractores de features

### 4.1 Semantic (`src/features/semantic_extractor.py`)

- Modelo: `FacebookAI/xlm-roberta-large-finetuned-conll02-spanish` (un
  modelo de lenguaje grande, multilingüe, afinado para español).
- Para cada texto, saca el vector de cada token (`last_hidden_state`) y
  aplica **pooling** (`--semantic_pooling`, default `mean`) para obtener un
  solo vector de ~1024 dimensiones por artículo.
- Opción `attention` pooling: usa una capa de atención aprendida — **pero
  ojo**, esa capa nunca se entrena (no hay fine-tuning end-to-end en este
  proyecto), así que sus pesos son aleatorios. No tiene sentido usarla como
  si fuera "mejor" salvo que en el futuro se entrene el modelo completo.
- Salida: DataFrame con columna `sem_emb` (lista de floats), más metadatos
  (`pooling`, `model_name`, `max_len`).

### 4.2 Emotion (`src/features/emotion_extractor.py`)

- Usa `pysentimiento` (librería preentrenada en español) para dos tareas:
  `emotion` (alegría, enojo, miedo, etc.) y `sentiment` (positivo/negativo/
  neutro) — ambas devuelven vectores de probabilidad.
- Además calcula **señales léxicas hechas a mano** (no aprendidas): ratio de
  signos de exclamación/interrogación, ratio de mayúsculas, ratio de
  emojis, uso de palabras "intensificadoras" (ej. "alarmante", "shock"),
  ratio de puntuación repetida, etc.
- Salida: DataFrame con columnas `emo_probs`, `sent_probs`, `signals` (listas).

### 4.3 Style (`src/features/style_extractor.py`)

- No usa ningún modelo de lenguaje. Usa **spaCy** (parsing gramatical),
  **textstat** (legibilidad) y **wordfreq** (frecuencia de palabras en
  español) para construir ~35 features artesanales:
  - Legibilidad (fórmula IFSZ, similar a Flesch para español).
  - Formalidad (ratio de sustantivos/adjetivos/preposiciones vs.
    pronombres/verbos/interjecciones — fórmula de Heylighen & Dewaele).
  - Complejidad sintáctica (longitud de oración, profundidad del árbol de
    dependencias, subordinadas por oración).
  - Diversidad léxica (Type-Token Ratio, Herdan's C, Root TTR).
  - Ratios de categorías gramaticales (POS: sustantivo, verbo, adjetivo...).
  - Tasa de palabras "raras" (proxy de errores ortográficos, vía
    `wordfreq`).
  - Señales de puntuación/estilo: exclamaciones, mayúsculas, comillas,
    caracteres repetidos, "burstiness" (variabilidad en longitud de
    oraciones).
- Si spaCy no está disponible, cae a un modo *fallback* solo con regex (pierde
  las features basadas en POS/dependencias, no falla el pipeline).
- Cada feature se normaliza según su tipo (ratios → clip [0,1], scores tipo
  formalidad/legibilidad → `tanh`, conteos → `log1p`) para que todas entren
  al VAE en escalas comparables.
- Salida: **no es un DataFrame**, es un **dict payload**:
  `{X: [N,D], feature_names: [...], ids, y, meta}` — mismo esquema que
  *context*.

### 4.4 Context (`src/features/context_extractor.py`)

- No usa NLP en absoluto ni entrena nada. Toma metadatos estructurados:
  `Source` (medio), `Domain` (extraído de la URL en `Link`), `Topic`,
  opcionalmente `Author`/fecha.
- Cada categoría se convierte a un vector de tamaño fijo con **feature
  hashing** determinístico (`_hash_embed`): se aplica MD5 al string y se usa
  para elegir un índice y un signo dentro de un vector de tamaño `dim`. Esto
  evita tener que "aprender" un vocabulario de cientos de medios distintos.
- También agrega la antigüedad del artículo en días (si hay columna de
  fecha) y flags booleanos (`ctx_has_source`, `ctx_has_topic`, etc.).
- Salida: dict payload igual que *style*.
- **⚠️ Este es el extractor central del problema de fuga de datos — ver
  sección 9.**

---

## 5. Las VAEs (compresión latente)

Código: `src/models/train_vae_from_pkl.py`. Se entrena **un VAE por rama**
(4 VAEs independientes), cada uno con su propia arquitectura de capas
ocultas fija:

| Rama | Capas ocultas (encoder, espejo en decoder) | Latent dim default |
|---|---|---|
| semantic | `[512, 256]` | 128 |
| emotion | `[128, 64]` | 16 |
| style | `[128, 64]` | 16 |
| context | `[256, 128]` | 64 |

Arquitectura: `Dense + Dropout` apiladas según `hidden_dims`, hasta llegar a
dos cabezas `z_mean` y `z_log_var` (la distribución latente), de las que se
"muestrea" `z` con el truco de reparametrización (`Sampling` layer). El
decoder es el espejo simétrico.

**Función de pérdida (β-VAE):**

```
loss = reconstruction_loss + beta * kl_loss
```

- `reconstruction_loss`: qué tan bien el decoder reconstruye el vector
  original a partir de `z`.
- `kl_loss`: qué tan cerca está la distribución latente aprendida de una
  distribución normal estándar (regularización).
- `beta` controla el balance: más beta = espacio latente más "ordenado" pero
  reconstrucción peor; menos beta = mejor reconstrucción pero espacio menos
  regularizado.

Antes de entrenar, cada rama se estandariza con `StandardScaler` (se ajusta
solo en train, se reutiliza igual en val/test).

### 5.1 Hiperparámetros de VAE que se mueven en los experimentos

Todos vía flags de `main.py`, aplicados **a las 4 ramas simultáneamente**
cuando `--run_vaes` está activo:

| Flag | Default en `main.py` | Qué controla |
|---|---|---|
| `--semantic_latent_dim` | 128 | Dimensión del espacio latente de la rama semantic |
| `--emotion_latent_dim` | 16 | ídem emotion |
| `--style_latent_dim` | 16 | ídem style |
| `--context_latent_dim` | 64 | ídem context |
| `--vae_epochs` | 100 | Épocas máximas (con early stopping, patience=10 fijo en el código) |
| `--vae_batch_size` | 32 | Tamaño de batch |
| `--vae_learning_rate` | 1e-3 | Learning rate del optimizador Adam |
| `--vae_beta` | 1.0 | Peso de la pérdida KL (β del β-VAE) |
| `--vae_dropout` | 0.1 | Dropout en encoder/decoder |
| `--exclude_semantic/emotion/style/context` | (desactivado) | Excluye por completo esa rama del entrenamiento VAE y del merge posterior |

En los barridos de experimentos (`scripts/experiment_config.py`), estos son
los valores que efectivamente se prueban:

- **Latent dims** (aplicados a las 4 ramas a la vez, como "preset"):
  - `small`: `sem=64, emo=8, sty=8, ctx=32`
  - `default`: `sem=128, emo=16, sty=16, ctx=64`
  - `large`: `sem=256, emo=32, sty=32, ctx=128`
- **Regularización VAE** (`beta`/`dropout`, un knob a la vez sobre el default):
  - `beta_low = 0.25`, `beta_high = 4.0`
  - `dropout_low = 0.0`, `dropout_high = 0.3`
  - default: `beta=1.0, dropout=0.1`

---

## 6. El clasificador KAN

Código: `src/models/kan.py`, entrenado en PyTorch sobre
`data/06_vae_latents_merged/{split}.pkl` (los 4 latentes ya concatenados,
con prefijo `{branch}_` en cada columna).

**Aclaración importante del propio código:** el nombre "KAN" viene de
*Kolmogorov-Arnold Network*, pero esta implementación **no usa splines**
(que es lo que define a un KAN "de verdad" en la literatura). Usa una
**base RBF fija** (`KANLayer`): centros equiespaciados en `linspace(-3, 3,
num_basis)`, con un ancho (`gamma`) aprendible compartido, y coeficientes
aprendidos por combinación de entrada/salida/base. Trátalo como "una capa no
lineal con forma libre aproximada por funciones de base radial", no como una
red de splines de Kolmogorov-Arnold literal.

**Arquitectura (`KANClassifier`):**

```
KANLayer(in_dim → hidden_dim, num_basis)
  → LayerNorm → SiLU → Dropout
KANLayer(hidden_dim → hidden_dim/2, num_basis)
  → LayerNorm → SiLU → Dropout
Linear(hidden_dim/2 → 1)
```

Entrenado con `BCEWithLogitsLoss` (pérdida binaria estándar) + `AdamW`, con
**early stopping** sobre la pérdida de validación (se guarda el mejor
checkpoint, `best_kan_model.pt`).

**Convención de etiquetas (importante, es consistente en todo el proyecto):
`1 = Fake`, `0 = True/Real`.** `y_prob` siempre es P(Fake).

### 6.1 Hiperparámetros de KAN que se mueven en los experimentos

| Flag | Default en `main.py` | Qué controla |
|---|---|---|
| `--kan_num_basis` | 16 | Número de funciones de base RBF por `KANLayer` |
| `--kan_hidden_dim` | 64 | Neuronas en la primera capa oculta (la segunda es `hidden_dim/2`) |
| `--kan_dropout` | 0.2 | Dropout entre capas |
| `--kan_epochs` | 100 | Épocas máximas |
| `--kan_batch_size` | 32 | Tamaño de batch |
| `--kan_lr` | 1e-3 | Learning rate (AdamW) |
| `--kan_weight_decay` | 1e-4 | Regularización L2 (AdamW) |
| `--kan_patience` | 15 | Épocas sin mejora en val_loss antes de early stopping |
| `--kan_seed` | 42 | Semilla (afecta init de pesos, dropout, shuffle del DataLoader) |

Valores explorados en los barridos (`scripts/experiment_config.py`):

- `num_basis`: `[4, 8, 16, 32]`
- `hidden_dim`: `[16, 32, 64, 128]`
- Entrenamiento (un knob a la vez sobre el baseline fijo):
  - `lr`: `1e-4 (low)`, `1e-3 (default)`, `5e-3 (high)`
  - `batch_size`: `16`, `64` (además del default 32)
  - `weight_decay`: `1e-5 (low)`, `1e-3 (high)` (además del default 1e-4)
- Baseline fijo mientras se explora lo anterior: `dropout=0.2, epochs=100,
  patience=15, batch_size=32, lr=1e-3, weight_decay=1e-4`.
- **10 semillas fijas** para cada configuración candidata, para poder hacer
  comparaciones pareadas (test de Wilcoxon):
  `[7, 42, 123, 777, 2024, 31415, 8675309, 20260817, 99, 1]`.

---

## 7. Métricas y evaluación

Código: `src/evaluation/metrics.py`. Después de cada `--train_kan`, se
evalúan train/val/test con `evaluate_binary_classifier`, que calcula:

- **Básicas**: accuracy, balanced accuracy, precision, recall, specificity, F1.
- **De ranking/probabilidad**: ROC-AUC, PR-AUC, log loss, Brier score.
- **De calibración/incertidumbre**: ECE (*expected calibration error*, qué
  tan bien las probabilidades predichas reflejan la frecuencia real de
  aciertos), entropía media/std de las predicciones.
- **Matriz de confusión**: tn/fp/fn/tp, false positive rate, false negative rate.
- **MCC** (Matthews correlation coefficient) — útil cuando hay desbalance de clases.

`save_metrics()` escribe tanto `{prefix}.json` como `{prefix}.csv` — de ahí
salen los archivos `train/val/test_metrics.{json,csv}` en `data/07_kan_runs/`.

También hay un **desglose por Topic** (`compute_topic_breakdown`) para ver
si el modelo depende de temas sobre-representados en train — solo para el
split de test, y solo si hay una columna `Topic` alineable posicionalmente.

### 7.1 Registro de experimentos

Cada corrida de `--train_kan` escribe un JSON en `results/{run_id}.json`
(vía `src/experiments/run_logger.py`) con: extractores activos, latent
dims, todos los hiperparámetros de VAE y KAN usados, métricas completas de
las 3 splits, tiempo de entrenamiento, número de parámetros, hash del
dataset usado, commit de git, y una copia del checkpoint del KAN.

Para comparar experimentos:

```bash
python scripts/report_builder.py     # -> reports/experiments_summary.csv + heatmap + histogramas
```

O abrir `viewer/index.html` directamente en el navegador (sin servidor) y
arrastrar uno o varios `results/*.json` — da una tabla ordenable/filtrable,
resumen narrativo automático, gráficas de overfitting/calibración y
estabilidad entre semillas.

---

## 8. Las 4 fases experimentales

El proyecto avanzó en fases, cada una construida sobre la anterior
(`scripts/orchestrator_phase{1,2,3,4}.py`, config compartida en
`scripts/experiment_config.py`):

### Fase 1 — Selección de combinación de modalidades
Prueba qué subconjunto de las 4 ramas (`semantic`, `emotion`, `style`,
`context`) da mejor F1, con `10` semillas fijas por combinación, sobre el
split original fijo. Resultado: `results/phase1_top3.json` — el top-3 pasa a Fase 2.

### Fase 2 — Barrido de hiperparámetros
Sobre las combinaciones ganadoras de Fase 1, explora latent dims (small/
default/large), regularización VAE (beta/dropout), `num_basis`, `hidden_dim`
y parámetros de entrenamiento del KAN (uno a la vez sobre el baseline).
**Ganador de Fase 2 (confirmado):** las 4 modalidades activas
(`semantic+emotion+style+context`), split original fijo,
**F1 medio = 0.8678** (std=0.0047, n=10 semillas). Guardado en
`results/phase2_final_top.json`.

### Fase 3 — Robustez ante particiones (k-fold estratificado, NO por medio)
Toma el único config ganador de Fase 2 y lo repite sobre 5 folds
estratificados por label (`--corpus_mode kfold`), NO por `Source`. Confirma
que el resultado no es un accidente de qué filas cayeron en test:
**F1 medio = 0.8624** (std=0.0215, min=0.8123, max=0.9032, 50 corridas = 5
folds × 10 semillas). Pero **esto no detecta fuga de datos por medio**,
porque un mismo `Source` puede seguir apareciendo tanto en train como en
test dentro de un fold — ver sección 9.

### Fase 4 — Test definitivo: splits sin fuga por medio (`source_disjoint`)
Repite el mismo config ganador, pero con `--corpus_mode source_disjoint`:
ahora ningún medio (`Source`) puede aparecer en más de un split del mismo
fold (`StratifiedGroupKFold` + `GroupShuffleSplit` agrupado por `Source`,
`src/data/source_split_corpus.py`). Este es el test que separa "señal
estilística real" de "memorización del medio".

**Estado (2026-08-24):** Fase 4 empezó a correr hoy — hay resultados
parciales en `results/orchestrator_phase4.jsonl` (10 semillas del fold 0,
mismos hiperparámetros ganadores de Fase 2: `latent sem=64/emo=8/sty=8/
ctx=32`, `num_basis=8`, `hidden_dim=16`, `dropout=0.2`, `lr=0.001`,
`weight_decay=0.001`, `vae_beta=1.0`, `vae_dropout=0.3`). Los números del
fold 0 muestran **F1 ≈ 0.20–0.52** (muy por debajo tanto del ~0.86 de la
Fase 2/3 como del ~0.75 estimado por la ablación de la sección 9) — pero es
**solo un fold, aún en curso**; no se puede concluir nada definitivo hasta
tener las 5 folds × 10 semillas completas y compararlas contra
`results/phase4_final_top.json` (que todavía no existe). Vale la pena
revisar el estado real de `results/orchestrator_phase4.jsonl` antes de citar
estos números como finales.

---

## 9. ⚠️ El hallazgo más importante: fuga de datos por medio (`Source`)

Este es el límite metodológico más relevante del proyecto, documentado en
la sección "Known Limitations & Caveats" del `README.md`:

- En el split original, **`Source` (el medio) casi determina perfectamente
  la etiqueta**: solo 5 de 197 medios en train publican tanto `Fake` como
  `True`. Lo mismo pasa con `Domain` (derivado de la URL): solo 7 de 196.
- **43% de los medios/dominios del test también aparecen en train.**
- El extractor `context` hashea `Source` y `Domain` directamente en su
  embedding — así que parte de lo que el modelo "aprende" es simplemente
  "este medio siempre publica Fake", no un patrón de estilo genuino.

**Ablación controlada** (mismas 10 semillas, misma arquitectura, quitando
`Source`/`Domain` del context extractor con `--context_source_dim 0
--context_domain_dim 0`, dejando solo Topic+edad):

| Variante | F1 medio | Rango |
|---|---|---|
| Context completo (Source+Domain+Topic) | 0.8604 | 0.8396 – 0.8956 |
| Context sin Source/Domain (solo Topic+edad) | 0.7449 | 0.6973 – 0.7829 |
| Sin rama context | 0.7559 | 0.7294 – 0.7858 |

Cada semilla con context completo superó a **cada** semilla sin identidad de
medio — los rangos no se traslapan. Conclusión: casi el 100% de la
contribución de `context` es memorización de medio, no señal contextual
genuina. El **techo honesto** de este pipeline sin depender de la identidad
del medio ronda **F1 ≈ 0.75–0.76**, no el ~0.86–0.90 reportado con context
completo.

**Por qué importa para una tesis centrada en estilo:** si la Fase 4
confirma que el F1 colapsa hacia ~0.75 en splits sin fuga, eso valida que el
~0.86 de Fases 2/3 estaba inflado por memorización de medio — y que el
verdadero techo del estilo/semántica (sin trampa) es más bajo pero
metodológicamente honesto.

---

## 10. Cómo correr las cosas (chuleta rápida)

```bash
source venv/bin/activate

# pipeline completo, split original
python main.py --prepare_corpus --preprocess_text
python main.py --extract_semantic --extract_emotion --extract_style --extract_context
python main.py --run_vaes --merge_vae_latents --train_kan

# solo re-entrenar KAN con otros hiperparámetros (reusa VAEs/latentes ya generados)
python main.py --train_kan --kan_num_basis 8 --kan_hidden_dim 16 --kan_lr 0.001 --kan_seed 7

# correr un fold source-disjoint (Fase 4) individual
python main.py --corpus_mode source_disjoint --source_split_n 5 --source_split_index 0 \
  --preprocess_text --extract_semantic --extract_emotion --extract_style --extract_context \
  --run_vaes --merge_vae_latents --train_kan --kan_seed 42

# excluir una modalidad (ej. quitar context para chequear fuga)
python main.py --run_vaes --merge_vae_latents --train_kan --exclude_context

# inspeccionar un pkl generado
python scripts/inspect_pkl.py --pkl data/03_features_raw/semantic/train_semantic.pkl
python scripts/peek_pkl_row.py --pkl data/03_features_raw/semantic/train_semantic.pkl --row 0

# comparar todos los experimentos corridos
python scripts/report_builder.py
```

Utilidad adicional: `scripts/pca_latent_dim_suggester.py` sugiere dimensiones
latentes razonables por rama a partir de varianza explicada (PCA), antes de
elegir `--*_latent_dim`.

---

## 11. Glosario rápido

- **XLM-RoBERTa**: modelo de lenguaje multilingüe preentrenado (familia
  RoBERTa/BERT), usado aquí solo para extraer embeddings, no se reentrena.
- **pysentimiento**: librería en español para análisis de sentimiento/emoción.
- **spaCy**: librería de NLP clásico (tokenización, POS tagging, parsing de
  dependencias).
- **textstat / wordfreq**: librerías auxiliares para legibilidad y
  frecuencia de palabras.
- **feature hashing**: truco para convertir categorías (sin vocabulario
  fijo) en vectores de tamaño fijo vía función hash, sin entrenar nada.
- **β-VAE**: variante del VAE con un peso `beta` en la pérdida KL para
  controlar qué tan regularizado queda el espacio latente.
- **RBF (radial basis function)**: función que depende de la distancia a un
  "centro" — aquí se usa como base no lineal dentro de cada `KANLayer`.
- **Early stopping**: detener el entrenamiento cuando la pérdida de
  validación deja de mejorar por `patience` épocas, para evitar overfitting.
- **AdamW**: variante de Adam (optimizador) con weight decay desacoplado.
- **ECE (expected calibration error)**: qué tan bien calibradas están las
  probabilidades predichas (si el modelo dice 80% de confianza, ¿acierta
  ~80% de las veces?).
- **Source-disjoint split**: partición de datos donde ninguna categoría de
  agrupación (aquí, el medio/`Source`) aparece en más de un conjunto
  (train/val/test) — evita que el modelo "memorice" la categoría en vez de
  aprender el patrón real.
