# Tabla comparativa de modelos — ReWear AI

> Generado por `scripts/generate_comparison_table.py` a partir de archivos del repositorio. No editar a mano: volver a correr el script. Todas las columnas están en `reports/tabla_comparativa.csv`.

## Cómo leer esta tabla

- Cada fila es un modelo o variante evaluado en un split. **Sólo se comparan filas del mismo experimento y split** (columna `grupo_comparacion` del CSV).
- mAP (cajas, E1), AUROC por patch (E2) y AUROC por imagen o píxel (E3) responden preguntas distintas: un número mayor en un experimento no indica un método mejor que otro de otro experimento.
- Supervisado vs. anomalías sólo se compara directamente en **E3 (FabriD)**, con las mismas imágenes de test.
- Valid se usó para elegir variantes, umbrales y épocas; test se evaluó una sola vez con lo elegido en valid.
- IC 95 % por bootstrap: E1 remuestrea capturas (recall y F1), E2 agrupa por imagen madre (AUROC por patch) y E3 está pareado por `source_id` (AUROC por imagen, sólo test). Si los IC individuales se superponen no se puede afirmar que un modelo sea mejor; la diferencia pareada, cuando existe, es la evidencia más directa.
- En negrita, el mejor valor de cada columna dentro de su tabla. Los tiempos dependen del hardware y de la unidad de cada experimento: no se comparan entre experimentos.
- `pendiente`: resultado que todavía no existe; no se completa con estimaciones.

## Resumen de test

- **E1-H — CISUTAC Holes.** Pendiente, test de Holes todavía no evaluado: A0 YOLO11n (Foto completa 640); A3 YOLO11n o YOLO26n (Elegida en el issue #7, con selección de checkpoint según el diagnóstico del issue #5).
- **E1-S — CISUTAC Spots.** mAP@0.5 (métrica principal, sin IC): mayor en A3-abl (Tiles 640, sin negativos) 0.097, frente a A0 (Foto completa 640) 0.002. Recall de cajas: mayor en A3-abl (Tiles 640, sin negativos) 0.177 (0.125–0.238), frente a A0 (Foto completa 640) 0.010 (0.000–0.023): los IC no se superponen.
- **E2 — RAW-FABRID.** AUROC por patch (criterio de selección): mayor en PatchCore (base con patches reducidos a 128 px) 0.902 (0.873–0.930), frente a PaDiM (ResNet18, 100 canales) 0.864 (0.834–0.896): los IC se superponen; frente a Estadística local (clásico, media y desvío en ventana 15×15) 0.596 (0.563–0.626): los IC no se superponen. Diferencia pareada B2 − B1 (PatchCore elegida − PaDiM) en AUROC por patch: +0.038 (IC 95 %: +0.003 a +0.073; 143 madres, 2000 remuestreos, semilla 20261121). Tomada del texto de conclusiones del notebook 03, no de una salida de celda.
- **E3 — FabriD v3.** AUPRO@0.3 (criterio de selección en valid, sin IC): mayor en YOLO11n-seg (imgsz 640, hasta 100 épocas, paciencia 30) 0.831, frente a PatchCore (ResNet18, 256 px, coreset 1 %) 0.331. AUROC por imagen: mayor en YOLO11n-seg (imgsz 640, hasta 100 épocas, paciencia 30) 0.874 (0.824–0.918), frente a PatchCore (ResNet18, 256 px, coreset 1 %) 0.643 (0.577–0.711): los IC no se superponen. Diferencia pareada YOLO11n-seg − PatchCore en AUROC por imagen: +0.231 (IC 95 %: +0.148 a +0.308; bootstrap pareado por `source_id`, 2000 réplicas).
- **E4 — Transferencia FabriD → CISUTAC sin ajuste.** Pendiente, falta el best.pt del E3: YOLO11n-seg de E3 (Transferencia sin ajuste, máscaras convertidas a cajas: hole → agujero, stain → mancha).

> **Nota sobre E1.** La A1 (foto completa a 1280) quedó cerca de la A3 en valid pero no fue a test, así que todavía no se puede afirmar que los tiles le ganen a subir la resolución. mAP@0.5 en valid — Holes: A1 0.105 frente a 0.131 de A3 (Tiles 640, 1 negativo por positivo); Spots: A1 0.031 frente a 0.039 de A3-abl (Tiles 640, sin negativos).

## E1-H — CISUTAC Holes: detección de agujeros con cajas (supervisado)

YOLO11n preentrenado en COCO. Compara reducir la foto completa (A0, A1) con recorrerla en tiles (A2, A3).

### Valid (43 capturas originales, 71 cajas reales)

| ID | Variante | Modelo | ¿Fue a test? | mAP@0.5 | mAP@0.5:0.95 | AP_small | Precisión | Recall (IC 95 %) | F1 (IC 95 %) | Umbral conf. | Tiles/captura | s/captura |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | Foto completa 640 | YOLO11n | pendiente | 0.042 | 0.011 | 0.000 | 0.133 | 0.085 (0.029–0.167) | 0.103 (0.035–0.191) | 0.05 | 1.0 | 0.038 |
| A1 | Foto completa 1280 | YOLO11n | no | 0.105 | 0.034 | 0.033 | **0.311** | 0.197 (0.108–0.315) | 0.241 (0.139–0.358) | 0.14 | 1.0 | 0.049 |
| A2 | Modelo A0 + tiles 640 en inferencia | YOLO11n | no | 0.005 | 0.001 | 0.002 | 0.039 | 0.028 (0.000–0.079) | 0.033 (0.000–0.101) | 0.22 | 32.0 | 0.597 |
| A2 | Modelo A0 + tiles 640 y foto completa en inferencia | YOLO11n | no | 0.010 | 0.002 | 0.002 | 0.075 | 0.056 (0.013–0.129) | 0.065 (0.014–0.167) | 0.22 | 33.0 | 0.651 |
| A3 | Tiles 640, 1 negativo por positivo | YOLO11n | a definir (issue #7) | **0.131** | 0.039 | 0.077 | 0.276 | **0.225** (0.143–0.319) | **0.248** (0.162–0.340) | 0.18 | 32.0 | 0.454 |
| A3 | Tiles 640 neg1 + foto completa en inferencia | YOLO11n | a definir (issue #7) | 0.119 | 0.035 | 0.077 | 0.250 | **0.225** (0.143–0.319) | 0.237 (0.154–0.326) | 0.18 | 33.0 | 0.537 |
| A3 | Tiles 640, 1 negativo por positivo | YOLO26n | a definir (issue #7) | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| A3-abl | Tiles 1024, 1 negativo por positivo | YOLO11n | a definir (issue #7) | 0.084 | 0.023 | 0.050 | 0.238 | 0.141 (0.063–0.225) | 0.177 (0.085–0.264) | 0.15 | 15.0 | 0.374 |
| A3-abl | Tiles 640, sin negativos | YOLO11n | a definir (issue #7) | 0.121 | 0.040 | 0.069 | 0.267 | **0.225** (0.127–0.356) | 0.244 (0.143–0.353) | 0.33 | 32.0 | 0.485 |
| A3-abl | Tiles 640, 3 negativos por positivo | YOLO11n | a definir (issue #7) | 0.109 | **0.043** | **0.084** | 0.161 | 0.211 (0.111–0.319) | 0.183 (0.103–0.257) | 0.08 | 32.0 | 0.447 |
| A3-abl | Tiles 640 neg1, presupuesto extendido a 100 épocas | YOLO11n | a definir (issue #7) | 0.107 | **0.043** | 0.031 | 0.241 | 0.197 (0.110–0.294) | 0.217 | 0.38 | 32.0 | 0.441 |

- A3 YOLO26n (Tiles 640, 1 negativo por positivo): en curso en el issue #6.
- A3-abl YOLO11n (Tiles 640 neg1, presupuesto extendido a 100 épocas): sin metricas.json en el repo; valores de la salida del notebook 02b, con 3 decimales y sin IC de F1.

### Test

| ID | Variante | Modelo | mAP@0.5 | mAP@0.5:0.95 | AP_small | Precisión | Recall (IC 95 %) | F1 (IC 95 %) | Umbral conf. | Tiles/captura | s/captura |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | Foto completa 640 | YOLO11n | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |
| A3 | Elegida en el issue #7, con selección de checkpoint según el diagnóstico del issue #5 | YOLO11n o YOLO26n | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente | pendiente |

- Test de Holes todavía no evaluado.

## E1-S — CISUTAC Spots: detección de manchas con cajas (supervisado)

Mismo diseño que E1-H, con manchas.

### Valid (59 capturas originales, 284 cajas reales)

| ID | Variante | Modelo | ¿Fue a test? | mAP@0.5 | mAP@0.5:0.95 | AP_small | Precisión | Recall (IC 95 %) | F1 (IC 95 %) | Umbral conf. | Tiles/captura | s/captura |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | Foto completa 640 | YOLO11n | sí | 0.011 | 0.006 | 0.000 | 0.045 | 0.011 (0.000–0.025) | 0.017 (0.000–0.040) | 0.01 | 1.0 | 0.036 |
| A1 | Foto completa 1280 | YOLO11n | no | 0.031 | 0.013 | 0.002 | 0.094 | 0.088 (0.054–0.128) | 0.091 (0.057–0.127) | 0.03 | 1.0 | 0.046 |
| A2 | Modelo A0 + tiles 640 en inferencia | YOLO11n | no | 0.001 | 0.000 | 0.000 | 0.014 | 0.025 (0.003–0.055) | 0.018 (0.003–0.039) | 0.03 | 31.2 | 0.486 |
| A2 | Modelo A0 + tiles 640 y foto completa en inferencia | YOLO11n | no | 0.004 | 0.002 | 0.000 | 0.017 | 0.032 (0.008–0.066) | 0.022 (0.006–0.044) | 0.03 | 32.2 | 0.551 |
| A3 | Tiles 640, 1 negativo por positivo | YOLO11n | no | 0.022 | 0.006 | 0.008 | 0.090 | 0.067 (0.033–0.104) | 0.077 (0.040–0.115) | 0.05 | 31.2 | 0.449 |
| A3 | Tiles 640 neg1 + foto completa en inferencia | YOLO11n | no | 0.022 | 0.006 | 0.008 | 0.090 | 0.070 (0.036–0.108) | 0.079 (0.041–0.118) | 0.05 | 32.2 | 0.518 |
| A3-abl | Tiles 1024, 1 negativo por positivo | YOLO11n | no | 0.018 | 0.007 | 0.009 | 0.110 | 0.070 (0.032–0.114) | 0.086 (0.042–0.135) | 0.04 | 14.7 | 0.366 |
| A3-abl | Tiles 640, sin negativos | YOLO11n | sí | **0.039** | **0.014** | **0.017** | **0.137** | **0.130** (0.081–0.179) | **0.133** (0.088–0.178) | 0.09 | 31.2 | 0.457 |
| A3-abl | Tiles 640, 3 negativos por positivo | YOLO11n | no | 0.006 | 0.002 | 0.003 | 0.034 | 0.049 (0.024–0.074) | 0.040 (0.020–0.063) | 0.01 | 31.2 | 0.418 |

### Test (66 capturas originales, 311 cajas reales)

| ID | Variante | Modelo | mAP@0.5 | mAP@0.5:0.95 | AP_small | Precisión | Recall (IC 95 %) | F1 (IC 95 %) | Umbral conf. | Tiles/captura | s/captura |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | Foto completa 640 | YOLO11n | 0.002 | 0.000 | 0.002 | 0.045 | 0.010 (0.000–0.023) | 0.016 (0.000–0.036) | 0.01 | 1.0 | 0.038 |
| A3-abl | Tiles 640, sin negativos | YOLO11n | **0.097** | **0.037** | **0.044** | **0.165** | **0.177** (0.125–0.238) | **0.171** (0.128–0.216) | 0.09 | 32.0 | 0.478 |

- Umbral de confianza fijado en valid.

## E2 — RAW-FABRID: detección de anomalías (entrenamiento sólo con tela normal)

Ningún método ve defectos en entrenamiento. Comparación con un método clásico (B0) y con PaDiM (B1).

### Valid (1652 patches de 59 imágenes madre, 197 defectuosos)

| ID | Modelo | Variante | ¿Fue a test? | AUROC patch (IC 95 %) | AUROC píxel | AUPRO@0.3 | Precisión patch | Recall patch | F1 patch | s/patch |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | Estadística local | clásico, media y desvío en ventana 15×15 | sí | 0.599 (0.537–0.656) | 0.637 | 0.286 | 0.175 | 0.711 | 0.281 | 0.068 |
| B1 | PaDiM | ResNet18, 100 canales | sí | 0.827 (0.768–0.881) | 0.894 | 0.757 | 0.851 | 0.437 | 0.577 | 0.058 |
| B2 | PatchCore | base: ResNet18, coreset 1 %, 256 px, 500 normales | no | 0.868 (0.824–0.908) | 0.878 | 0.722 | 0.757 | 0.569 | 0.649 | 0.064 |
| B2-abl | PatchCore | base con WideResNet50 | no | 0.898 (0.859–0.934) | 0.929 | **0.791** | 0.837 | 0.548 | 0.663 | 0.263 |
| B2-abl | PatchCore | base con coreset 10 % | no | 0.864 (0.817–0.910) | 0.899 | 0.751 | 0.894 | 0.513 | 0.652 | 0.369 |
| B2-abl | PatchCore | base con 100 normales | no | 0.855 (0.802–0.905) | 0.870 | 0.714 | 0.761 | 0.548 | 0.637 | 0.038 |
| B2-abl | PatchCore | base con 2000 normales | no | 0.882 (0.842–0.920) | 0.888 | 0.750 | 0.753 | 0.589 | 0.661 | 0.162 |
| B2-abl | PatchCore | base con patches reducidos a 128 px | sí | **0.911** (0.867–0.949) | **0.932** | 0.773 | **0.916** | **0.716** | **0.803** | 0.010 |

- Valores con 3 decimales, como los muestra el notebook 03.
- B2-abl PatchCore (base con patches reducidos a 128 px): elegida en valid por AUROC por patch.

### Test (4004 patches de 143 imágenes madre, 490 defectuosos)

| ID | Modelo | Variante | AUROC patch (IC 95 %) | AUROC píxel | AUPRO@0.3 | Precisión patch | Recall patch | F1 patch | s/patch |
|---|---|---|---|---|---|---|---|---|---|
| B0 | Estadística local | clásico, media y desvío en ventana 15×15 | 0.596 (0.563–0.626) | 0.594 | 0.270 | 0.160 | 0.645 | 0.257 | 0.069 |
| B1 | PaDiM | ResNet18, 100 canales | 0.864 (0.834–0.896) | **0.925** | **0.804** | **0.915** | 0.418 | 0.574 | 0.058 |
| B2-abl | PatchCore | base con patches reducidos a 128 px | **0.902** (0.873–0.930) | 0.912 | 0.745 | 0.876 | **0.661** | **0.753** | 0.010 |

- Valores con 3 decimales, como los muestra el notebook 03.

## E3 — FabriD v3: segmentación supervisada vs. anomalías

**Única comparación directa entre paradigmas:** mismas imágenes de test, mismos mapas binarios por píxel y umbral de cada método fijado en valid. Las métricas por clase del segmentador previstas en el protocolo todavía no se calcularon.

### Valid (375 imágenes de 363 fuentes)

| Modelo | Paradigma | Variante | ¿Fue a test? | AUROC imagen (IC 95 %) | AUROC píxel | AUPRO@0.3 | Dice | IoU | s/imagen |
|---|---|---|---|---|---|---|---|---|---|
| YOLO11n-seg | supervisado | imgsz 640, hasta 100 épocas, paciencia 30 | sí | **0.939** | **0.863** | **0.773** | **0.523** | **0.354** | 0.030 |
| PatchCore | anomalías | ResNet18, 256 px, coreset 1 % | sí | 0.607 | 0.743 | 0.413 | 0.104 | 0.055 | 0.062 |

- PatchCore (ResNet18, 256 px, coreset 1 %): no es la configuración elegida en E2 (patches reducidos a 128 px).

### Test (363 imágenes de 348 fuentes)

| Modelo | Paradigma | Variante | AUROC imagen (IC 95 %) | AUROC píxel | AUPRO@0.3 | Dice | IoU | s/imagen |
|---|---|---|---|---|---|---|---|---|
| YOLO11n-seg | supervisado | imgsz 640, hasta 100 épocas, paciencia 30 | **0.874** (0.824–0.918) | **0.891** | **0.831** | **0.560** | **0.389** | 0.208 |
| PatchCore | anomalías | ResNet18, 256 px, coreset 1 % | 0.643 (0.577–0.711) | 0.732 | 0.331 | 0.096 | 0.051 | 0.124 |

- Umbral fijado en valid.
- PatchCore (ResNet18, 256 px, coreset 1 %): no es la configuración elegida en E2 (patches reducidos a 128 px).

## E4 — Transferencia FabriD → CISUTAC sin ajuste

Modelo de E3 aplicado sin ajuste al test de CISUTAC, con las métricas de detección de E1.

### Test

| Modelo | Variante | mAP@0.5 | AP_small | Recall (IC 95 %) |
|---|---|---|---|---|
| YOLO11n-seg de E3 | Transferencia sin ajuste, máscaras convertidas a cajas: hole → agujero, stain → mancha | pendiente | pendiente | pendiente |

- Falta el best.pt del E3.
