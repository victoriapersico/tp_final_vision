# E4 — Transferencia FabriD → CISUTAC, sin ajuste

**Estado al 16/11/2026:** el pipeline está listo y probado de punta a punta, pero **falta FabriD v3**, que todavía se está descargando. Todavía no se entrenó ni se evaluó ningún modelo real, y el test de CISUTAC no se usó.

Protocolo: [`protocolo_experimental.md`](../protocolo_experimental.md), secciones 2, 5 y 7. Notebook: [`04_transferencia_fabrid_cisutac.ipynb`](../../notebooks/04_transferencia_fabrid_cisutac.ipynb).

## Qué se agregó o cambió

| Archivo | Qué hace |
|---|---|
| [`scripts/prepare_fabrid_yolo.py`](../../scripts/prepare_fabrid_yolo.py) (nuevo) | Arma FabriD para Ultralytics con el split agrupado `fab4_split_agrupado.csv`, sin los 19 ambiguos y sin tocar el export. `--clases` elige las clases (las 4 para el E3; hole y stain para el E4) y `--formato` elige entre polígonos y cajas. Escribe enlaces al export, labels, `data.yaml` y cajas COCO por clase para evaluar. **Sirve también para el E3.** |
| [`src/transferencia.py`](../../src/transferencia.py) (nuevo) | Mapeo de clases hole → Holes y stain → Spots, rutas de FabriD y la tabla de escala. |
| [`src/cisutac_baseline.py`](../../src/cisutac_baseline.py) (cambio compatible) | Parámetros opcionales en `make_predictor` y `evaluate`: `classes` (filtra la clase antes de la NMS, también en SAHI), `tile_imgsz` (un tile de 320 entra ampliado a 640) y `gt`/`image_dir` (evaluar FabriD con las mismas métricas). Sin estos parámetros se comporta igual que antes. |
| [`src/eda_utils.py`](../../src/eda_utils.py) | `link_or_copy` pasa acá desde `prepare_cisutac_yolo.py` para que la usen los dos scripts de preparación. |
| [`notebooks/04_transferencia_fabrid_cisutac.ipynb`](../../notebooks/04_transferencia_fabrid_cisutac.ipynb) (nuevo) | Narrativa del E4, con la misma estructura que el notebook 02 y `MODO_PRUEBA`. La sección 9 es el anexo con FabricSpotDefect, apagado por defecto (`ANEXO_FSD = False`). |
| [`scripts/prepare_fsd_yolo.py`](../../scripts/prepare_fsd_yolo.py) (nuevo) | Prepara los originales de FabricSpotDefect para el anexo, con las correcciones que se describen más abajo. |
| [`borrador_informe_e4.md`](borrador_informe_e4.md) (nuevo) | Método, escala, evaluación y limitaciones del E4 para el informe, todavía sin resultados. |

## Por qué se prueban tiles: la escala

Lado equivalente mediano, en px, con el que el defecto llega al modelo (sale de las tablas del EDA, sin modelo):

| | FabriD (export) | CISUTAC, foto completa 640 | CISUTAC, tile 640 | CISUTAC, tile 320 ampliado a 640 |
|---|---:|---:|---:|---:|
| Agujeros (hole) | 31,6 | 4,2 | 24,7 | 49,5 |
| Manchas (stain) | 60,0 | 3,0 | 18,2 | 36,4 |

- **Con la foto completa**, los defectos llegan entre 8 y 20 veces más chicos que en FabriD.
- **Con tiles de 640**, los agujeros quedan a una escala parecida a la de FabriD.
- **Las manchas** se acercan recién con tiles de 320, que son unos 135 por foto, 4 veces los de 640.

El tamaño de tile se elige con valid.

## Cómo se probó sin FabriD

- **Filtro de clase**, con YOLO11n de COCO como modelo de reemplazo sobre CISUTAC valid. En foto completa da las mismas detecciones que el filtro de Ultralytics, y en tiles sólo queda la clase pedida. Todas las cajas caen dentro de la foto original.
- **Tile de 320 con `tile_imgsz` 640:** el tile entra a la red como 640×640.
- **Preparación de FabriD**, con un export falso de 11 imágenes que cubre todos los casos:
  - normal, sólo hole y sólo stain;
  - sólo otra clase, mixta y ambigua;
  - hole y stain juntos;
  - un derivado movido de train a test;
  - un polígono que se sale del borde.

  Exclusiones, remapeo de clases, recorte al borde y cajas COCO en px dieron lo esperado. Ultralytics lee los dos formatos (cajas y polígonos) sin archivos corruptos, y con el de cajas entrena una época.
- **Notebook 04 completo**, con datos aislados en una carpeta temporal: CISUTAC enlazado, el export falso y un "test" que era una copia de valid, para no tocar el test real. No dio errores y generó tablas y figuras.
- **Compatibilidad:**
  - `evaluate` sin los parámetros nuevos funciona igual que antes;
  - `prepare_cisutac_yolo.py` corre con la función movida;
  - `generate_cisutac_eda.py` regenera los mismos archivos (mismos checksums).

## Al terminar la descarga

1. Verificar `data/fabrid_v3/annotations_export/` (2.407 imágenes, ver `docs/DATASETS.md`) y leer el README del export para ver si Roboflow ajustó el contraste (decisión 5).
2. Correr `python scripts/prepare_fabrid_yolo.py --clases hole stain --formato cajas`. Según el manifiesto, tiene que dar:

   | Split | Imágenes | Normales | Instancias hole | Instancias stain |
   |---|---:|---:|---:|---:|
   | train | 964 | 429 | 341 | 454 |
   | valid | 239 | 93 | 86 | 133 |
   | test | 203 | 85 | 69 + ? | 94 + ? |

   En test hay 4 imágenes con hole y stain juntos, con 10 instancias entre las dos clases. También tiene que excluir 19 ambiguas, 76 mixtas y 906 imágenes que sólo tienen otras clases. El script además verifica, imagen por imagen, que la cantidad de polígonos coincida con el manifiesto.
3. Correr el notebook 04 con `MODO_PRUEBA = True` y después completo.
4. Correr el test una sola vez al final, con los umbrales y el tile elegidos en valid.

## Anexo exploratorio: FabricSpotDefect → CISUTAC Spots

**Pendiente de aprobación del grupo:** no está en el E4 del protocolo. Se corrió mientras se descargaba FabriD, porque los datos ya estaban, y **sólo en valid**.

FabricSpotDefect es una segunda fuente industrial de manchas. Sus manchas llegan al modelo con un lado mediano de 36 px, entre los 60 px de FabriD y los 18 px de CISUTAC con tiles de 640.

### Problemas de datos que el EDA no había detectado

1. **Cajas giradas en 228 originales** (198 de train, 26 de valid y 4 de test). El archivo está acostado, pero las cajas COCO corresponden a la foto girada 90° en sentido horario: ancho y alto quedan invertidos y el archivo no tiene etiqueta EXIF que lo explique.
   - Lo verifiqué dibujando las cajas en las tres orientaciones; sólo con el giro horario caen sobre las manchas.
   - Las cajas sin convertir caen fuera de la foto en 146 de 762 casos.
   - `prepare_fsd_yolo.py` pasa las cajas a la orientación del archivo.

   > **Afecta al EDA de la Integrante 3:** en [`fsd1_ejemplos_cajas.png`](../fabrid/fsd1_ejemplos_cajas.png), los dos primeros paneles son fotos de este grupo y sus cajas están corridas. Los conteos de FSD-1 no cambian.
2. **Duplicados entre train y valid.** Las 12 fotos de valid con nombre numérico son en su mayoría re-subidas del lote de WhatsApp del 08/03/2024, que está en train. Las encontré comparando miniaturas en las cuatro rotaciones (dHash solo no alcanza: las telas lisas dan valores parecidos aunque sean distintas) y las revisé a ojo:

   | Foto de train | Foto de valid | Relación |
   |---|---|---|
   | IMG-20240308-WA0065 | 60 | Copia |
   | IMG-20240308-WA0069 | 47 | Copia |
   | IMG-20240308-WA0078 | 49 (y probablemente 48) | Copia |
   | IMG-20240308-WA0095 | 2 | Copia |
   | IMG-20240308-WA0102 | 9 | Copia |
   | IMG-20240308-WA0070 | 46 y 44 | Copia girada; misma pieza (escritura «SPD-421») |
   | IMG-20240308-WA0117 | 50 | Copia girada |
   | IMG-20240308-WA0061 | 73 | Misma pieza, otro encuadre |

   Como en FAB-4, la evaluación tiene prioridad: esas 8 fotos de train pasan a valid. Con test no hubo coincidencias. No se reagrupó por fecha de captura, así que puede haber otras fotos de una misma pieza en splits distintos; las métricas de FabricSpotDefect en su propio dominio son una referencia optimista.
3. **Se excluyen las 36 fotos de train sin cajas**, que no están verificadas como sanas, y no se usan los aumentados.

Resultado: 599/227/152 imágenes y 2.151/597/540 manchas en train/valid/test, el mismo total de 3.288 manchas que el COCO original. Ultralytics lee las 978 imágenes sin descartar ninguna.

### Resultados en valid

*Pendiente: el entrenamiento está corriendo.*

## Decisiones pendientes del grupo

Hasta que el grupo decida, el código usa la opción indicada como "por defecto"; cada una se cambia en un solo lugar.

| # | Decisión | Por defecto | Cómo cambiarla |
|---|---|---|---|
| 1 | ¿Reusar el modelo del E3? Si es YOLO11n-seg, sirve tal cual: se usan sus predicciones de hole y stain. Si es un segmentador por píxel, conviene un detector aparte, porque pasar mapas a cajas agrega decisiones que no tiene el E1. | Detector propio: YOLO11n de cajas, misma receta que el A0 | `PESOS_FABRID` en el notebook |
| 2 | ¿Qué imágenes de FabriD entran? | Sólo hole y stain, más las normales como fondo; se excluyen las 76 mixtas. Al dejar afuera las marcas de aguja también se evita el atajo de las imágenes de 512×512. | `--mixtas` |
| 3 | ¿Entrenar con el export de 640 o con los originales de 4160×3120? | Export de 640, como el E3 | — |
| 4 | ¿Qué se elige en valid de CISUTAC? | Umbral y tamaño de tile, como en el E1, declarado como "sin ajuste de pesos". La alternativa más estricta es el umbral de valid de FabriD. | `umbral` en `evaluar_cisutac` |
| 5 | ¿Se aplica a CISUTAC el ajuste de contraste del export, si lo hubo? | No | A revisar con el README del export |
| 6 | Comparación con el E1 | El notebook lee las evaluaciones de test del E1 desde `results/cisutac/bloque_a/` cuando existan. Una sola semilla, declarada como limitación. | — |
| 7 | ¿Sumar el anexo FabricSpotDefect → Spots al informe? Si se aprueba, también se evalúa su test una sola vez. | Apagado en el notebook; corrido sólo en valid | `ANEXO_FSD` en el notebook |

Además, `requirements.txt` no lista `pandas` (lo usan los notebooks 02 y 04) ni las dependencias de entrenamiento. Es el pendiente 9 del EDA de CISUTAC.
