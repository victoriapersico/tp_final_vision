# Predicciones del bloque A (CISUTAC)

Copia de los `predicciones.json` y `metricas.json` que genera el notebook 02 en
`results/cisutac/bloque_a/evaluaciones/<nombre>/`. La carpeta `results/` está en el
`.gitignore` (tiene pesos y archivos pesados), así que estos archivos se copian acá a mano
para poder hacer el análisis de errores sin volver a correr nada.

## Estructura

```
holes/<evaluacion>/   corridas de Holes (solo valid)
spots/<evaluacion>/   corridas de Spots (valid y test)
```

El nombre de cada carpeta es `<dataset>_<variante>__<split>_<modo de inferencia>`, por ejemplo
`spots_A3_tiles640_neg0__test_tiles640_o20`.

## Formato

- `predicciones.json`: lista con todas las cajas predichas, sin filtrar por confianza, en
  coordenadas de la imagen original:
  `{"image_id": 1, "bbox": [x, y, w, h], "score": 0.158}`.
  `image_id` es el id del ground truth COCO del split (`load_ground_truth(dataset, split)`
  en `src/cisutac_baseline.py`), que sirve para recuperar el nombre del archivo.
- `metricas.json`: métricas de esa evaluación, incluido `umbral_confianza` (el umbral de F1
  máximo elegido en valid). Para el análisis de errores conviene quedarse con las cajas que
  tienen `score >= umbral_confianza`, igual que hace `plot_examples`.

## Faltantes

La corrida con presupuesto extendido de Holes (`02b`) se ejecutó en otro Kaggle. Sus
predicciones hay que agregarlas en `holes/` cuando quien la corrió las descargue.
