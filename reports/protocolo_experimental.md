# ReWear AI — Protocolo experimental preliminar

**Versión:** 12/11/2026 · **Estado:** listo para revisión del equipo; todavía no se entrenaron modelos.

## 1. Pregunta central

¿Conservar el detalle mediante *tiles* mejora la localización de agujeros y manchas pequeños en prendas usadas frente a reducir la foto completa, y qué se aprende al comparar detección supervisada con PatchCore dentro de dominios compatibles?

> **Tile:** recorte de una imagen grande que permite analizarla sin achicar tanto el defecto.  
> **PatchCore:** método que aprende cómo se ve una tela normal y marca como anómala una zona distinta de ese patrón.

## 2. Experimentos fijados

| ID | Tarea y dataset | Entrenamiento | Evaluación independiente | Salida y unidad | Métricas principales |
|---|---|---|---|---|---|
| E1-H | Detección de agujeros, CISUTAC Holes | 300 capturas originales | valid 43; test 64 | Cajas por captura y por defecto | precision, recall, F1, mAP@0.5, mAP@0.5:0.95 y AP/recall por tamaño |
| E1-S | Detección de manchas, CISUTAC Spots | 361 capturas originales | valid 59; test 66 | Cajas por captura y por defecto | Las mismas que E1-H |
| E2 | Anomalías, RAW-FABRID | 14.196 patches `train/good` | 4.969 `test/good` + 687 `test/defect` | Puntaje por patch y mapa por píxel | AUROC patch, AUROC píxel y AUPRO |
| E3 | Segmentación vs. anomalías, FabriD | Split agrupado; sólo 607 normales concordantes para PatchCore | Test común: 85 normales + 278 defectuosas | Puntaje por imagen y mapa binario por píxel | AUROC imagen/píxel, AUPRO, IoU y Dice binarios; métricas por clase sólo para el segmentador |
| E4 | Transferencia a prendas, sin ajuste | Modelo industrial ya entrenado | Test congelado de CISUTAC | Salida convertida a cajas si corresponde | Mismas métricas de detección antes y después del cambio de dominio |

Los manifiestos son `reports/cisutac/cis5_split_propuesto.csv` y `reports/fabrid/fab4_split_agrupado.csv`. Los originales y sus particiones oficiales no se modifican.

## 3. Comparación principal de resolución

En E1 se entrenará el mismo detector base, inicialmente **YOLO11n**, bajo dos condiciones:

1. **Imagen completa:** lado mayor redimensionado a 640 px.
2. **Tiles:** entrenamiento e inferencia con recortes iniciales de 640 px y 20 % de solapamiento.

Tamaño de tile y solapamiento alternativos se eligen únicamente con validación. El test se usa una vez para la comparación final. Se mantiene igual el modelo, inicialización, cantidad de épocas y criterio de parada para que el cambio medido sea el procesamiento de escala.

Además de calidad, se informará tiempo por captura, número de tiles y memoria máxima cuando pueda medirse en el mismo hardware.

## 4. Qué significa «defecto pequeño»

La definición principal seguirá COCO: una caja es **small** si su área es menor a `32² px` en las coordenadas originales usadas para evaluar. Las predicciones de tiles o imágenes reducidas se remapean primero a la captura original.

Como análisis comprensible, también se reportará el lado equivalente `sqrt(área)` en tres grupos fijados antes de entrenar: `<16 px`, `16–32 px` y `>32 px`. No se cambiarán estos umbrales después de mirar el test.

## 5. Splits, independencia y casos excluidos

- **CISUTAC:** sólo 407 originales de Holes y 486 de Spots. Las copias 640×640 no cuentan como nuevas fotos. Split agrupado por día/sesión; si aparece evidencia de la misma prenda en días distintos se versiona el manifiesto antes del entrenamiento final.
- **RAW-FABRID:** patches de una misma `parent_raw` no son observaciones independientes. Intervalos de confianza y remuestreo se agrupan por madre; resultados también se desglosan por rollo. Las dos máscaras huérfanas se conservan pero se excluyen.
- **FabriD:** usar el split agrupado por `source_id`. Excluir los 19 casos ambiguos de normalidad. No usar derivados de una misma captura en lados distintos de train/test.
- **FabricSpotDefect:** sólo benchmark complementario de manchas. Las 2.300 versiones COCO y YOLO aumentadas representan los mismos ejemplos y nunca se reparten entre train/test. Las 36 imágenes sin cajas no se suponen sanas.
- **RISE:** no se usa: sólo hay 3 muestras locales, las etiquetas `holes`/`stains` son incompletas y no existen localizaciones.

> **Leakage o fuga:** ocurre cuando una foto, un recorte o una versión casi igual aparece en entrenamiento y test, haciendo que el resultado parezca mejor de lo que realmente es.

## 6. Selección y reproducibilidad

- Semilla base para splits y muestreo: **2026**.
- Comparación final supervisada: semillas **2026, 2027 y 2028** si el costo permite repetir; si se usa una sola, debe declararse como limitación.
- Umbral de confianza, umbral de anomalía, tamaño/solapamiento de tiles y parada temprana se eligen en **validación**, nunca en test.
- Reportar media y dispersión entre semillas; para RAW-FABRID, usar bootstrap agrupado por `parent_raw`.
- Guardar configuración, versión del dataset, manifiesto, semilla y hardware junto a cada resultado.

## 7. Cuándo una comparación es válida

- Supervisado vs. PatchCore sólo se compara directamente en **FabriD**, con las mismas imágenes de test y mapas binarios compatibles.
- `mAP` de un detector y `AUROC` de PatchCore contestan preguntas distintas; no se comparan como si una cifra mayor indicara un método mejor.
- En transferencia FabriD → CISUTAC se mapean `hole → agujero` y `stain → mancha`. La salida segmentada puede convertirse a cajas y evaluarse sin *fine-tuning* con las métricas CISUTAC.
- RAW-FABRID → CISUTAC queda exploratorio: al no tener prendas normales verificadas, CISUTAC no permite medir bien los falsos positivos de un detector de anomalías.

> **Domain shift o cambio de dominio:** el modelo se entrena con un tipo de imagen y luego ve imágenes tomadas en condiciones distintas, por lo que su rendimiento puede caer aunque el defecto tenga el mismo nombre.

## 8. Criterio de éxito y límites

La hipótesis principal se apoya si tiles mejora recall/AP de defectos pequeños en el mismo test sin un aumento de costo desproporcionado. Una mejora global sin mejora en defectos pequeños no confirma la hipótesis.

El protocolo no promete que un modelo industrial funcione sobre ropa usada. La transferencia mide esa dificultad; si las salidas no pueden transformarse de manera honesta a la misma unidad, se informa sólo como evidencia cualitativa.

