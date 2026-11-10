# ReWear AI — Auditoría y recomendación de datasets

Fecha de auditoría: 10 de noviembre de 2026. No se entrenó ningún modelo ni se modificaron imágenes originales.

## Resumen ejecutivo

La combinación más defendible es: **CISUTAC Holes/Spots para detección supervisada en prendas usadas**, **RAW-FABRID como benchmark principal de PatchCore**, y **FabriD v3 como benchmark supervisado de segmentación industrial y segundo ensayo de anomalías**. FabricSpotDefect sirve como apoyo para manchas, pero sus imágenes aumentadas no deben entrar en validación/test. RISE es valioso como fuente futura de prendas normales o clasificación débil, pero no como ground truth de localización; el corpus completo no fue descargado.

No conviene fusionar los datasets. Las prendas usadas fotografiadas, los paños industriales, las cámaras, fondos, resoluciones y protocolos de anotación son dominios distintos. Los resultados deben reportarse por dataset y por tipo de tarea.

## Inventario

| Dataset | Datos | Anotación | Normales útiles | Licencia | Uso aconsejado |
|---|---:|---|---:|---|---|
| CISUTAC Holes | 761 imágenes | 1.497 instancias, boxes y 3 polígonos | 0 | CC BY 4.0 | detección supervisada de agujeros en prendas |
| CISUTAC Spots | 906 imágenes | 4.777 instancias, boxes y 99 polígonos | 0 | CC BY 4.0 | detección supervisada de manchas en prendas |
| FabricSpotDefect | 1.014 originales; 2.300 aumentadas representadas dos veces (COCO/YOLO) | 3.288 instancias originales; boxes y 200 segmentaciones | no confiables | CC BY 4.0 | benchmark complementario de manchas/agujeros |
| FabriD v3 | 2.407 originales | 2.502 polígonos YOLO-seg; cuatro clases | 620 | CC BY 4.0 | segmentación y anomalías industriales |
| RAW-FABRID | 709 high-res + 19.852 parches de imagen | 361 boxes/polígonos, 204 máscaras; 687 máscaras de parches | 507 high-res / 14.196 parches train | CC BY 4.0 | benchmark principal PatchCore/segmentación |
| RISE front-only | 31.638 filas (28.248 train + 3.390 test) | etiquetas de condición a nivel imagen; sin cajas ni máscaras | candidatos, no ground truth limpio | CC BY 4.0 | negativos débiles y clasificación, sujeto a auditoría |

Fuentes oficiales: [CISUTAC Holes](https://huggingface.co/datasets/wargoninnovation/Garment_condition_holes), [CISUTAC Spots](https://huggingface.co/datasets/wargoninnovation/Garment_condition_spots), [FabricSpotDefect](https://data.mendeley.com/datasets/6574nhzm8x/1), [FabriD](https://data.mendeley.com/datasets/vncdwps24r/3), [RAW-FABRID](https://data.mendeley.com/datasets/db6g85xsyg/1) y [RISE front-only](https://huggingface.co/datasets/fnauman/fashion-second-hand-front-only-rgb).

## Hallazgos por dataset

### CISUTAC Holes y Spots

Ambos son el dominio más cercano a ReWear AI: prendas posconsumo, especialmente chaquetas, pantalones y remeras. Poseen splits `train`/`valid`, pero no `test`; Holes tiene 697/64 imágenes y Spots 830/76. No hay imágenes normales ni etiquetas vacías, por lo que no sirven por sí solos para entrenar PatchCore.

Los defectos son extremadamente pequeños. En Spots, la mediana de área normalizada es `0,00385 %` y 4.712/4.777 instancias ocupan menos del 1 % de la imagen. En Holes, la mediana es `0,00697 %` y 1.493/1.497 están bajo 1 %. Esto favorece entrenamiento con alta resolución, tiles o crops y métricas de objetos pequeños; redimensionar agresivamente puede borrar la señal.

Limitaciones: split reducido, ausencia de test, posibles capturas relacionadas y anotaciones mayormente rectangulares. Recomendación: reconstruir una evaluación agrupada por prenda/captura cuando sea posible y conservar el split oficial sólo como referencia.

### FabricSpotDefect

La carpeta `Orginal` contiene 1.014 imágenes: 643 train, 219 valid y 152 test. Sus COCO suman 3.288 anotaciones de dos categorías (`spot` y `Hole-EnBF`); 200 anotaciones contienen segmentación y el resto son esencialmente cajas. Hay 36 imágenes train sin anotaciones, pero la publicación no las documenta como normales verificadas, por lo que no deben asumirse automáticamente como negativas.

La carpeta aumentada contiene 2.300 imágenes en cada una de dos representaciones equivalentes (COCO y YOLO), totalizando 4.600 archivos físicos. No debe contarse como 4.600 ejemplos independientes. Las resoluciones originales son muy heterogéneas; la versión aumentada está normalizada a 416×416.

Recomendación: usar sólo originales para una evaluación honesta; aplicar aumentación únicamente dentro de train. Es un complemento útil, aunque el dominio de captura y la taxonomía no coinciden exactamente con CISUTAC.

### FabriD v3

Se descargó la versión 3 completa. Los originales son RGB JPEG: 620 defect-free, 467 needle mark, 459 hole, 454 missing yarn y 407 stain. Las resoluciones van desde 512×512 hasta 4160×3120, con varias orientaciones. El export anotado conserva 2.407 imágenes redimensionadas (máximo 640 px) y polígonos YOLO-seg.

Splits: train 1.685, valid 364, test 358. Hay 431/94/88 etiquetas vacías, coherentes con los 613 ejemplos defect-free presentes en el export (la diferencia frente a 620 originales merece revisión de correspondencia). Las 2.502 instancias están balanceadas razonablemente: stain 718, needle mark 636, missing yarn 580 y hole 568. La mediana de área es 0,9046 %; 1.331 instancias están bajo 1 %.

Alerta: el nombre base detecta **14 capturas fuente repetidas entre splits**. Antes de reportar métricas debe reconstruirse el split agrupando por captura original. Para PatchCore, las 620 imágenes defect-free son suficientes para un piloto, pero son paños industriales, no prendas completas.

### RAW-FABRID

Las 709 imágenes originales son PNG de 1792×1024. Aunque el README las describe como grayscale, PIL las reporta como RGB (probablemente tres canales equivalentes); las máscaras sí son `L` de un canal. El CSV marca 507 buenas y 202 defectuosas, mientras que existen 204 máscaras. Las dos máscaras no marcadas como defectuosas son:

- `Rollo2A_0000031_D_07-05-2024_10-39-06.546.png`
- `Rollo2A_0000032_D_07-05-2024_10-39-07.598.png`

El COCO contiene 361 defectos, todos con bbox y polígono. La mediana ocupa 0,06935 % de la imagen y 327/361 están bajo 1 %. La estructura MVTec incluye 14.196 `train/good`, 4.969 `test/good`, 687 `test/defect` y 687 máscaras. El CSV de parches preserva `parent_raw`.

Los 4.969 parches buenos de test y los 687 defectuosos provienen de las mismas 202 capturas defectuosas; esto es correcto para evaluar localización, pero la independencia estadística se define por captura/rollo, no por parche. Para comparaciones robustas, agrupar por `parent_raw` y realizar además una variante leave-one-roll-out.

### RISE front-only RGB

No se descargó el corpus. La consulta remota confirmó 14 Parquet por **2.843.670.317 bytes (2,84 GB decimal)**, no 26 GB. Sólo se guardaron tres muestras (una candidata normal y dos con manchas), 268 KB en total, junto con sus metadatos.

El esquema incluye imagen, condición 1–5, pilling 1–5, `damage`, `stains`, `holes`, `smell`, material y atributos de prenda. No hay bbox ni máscara. En train, `holes` falta en 26.883/28.248 filas y `stains` en 20.070; en test faltan en 3.145/3.390 y 2.467/3.390. La codificación también es inconsistente (`No`, `Yes`, `Minor`, `Major`, variantes minúsculas y `none`). Un valor nulo no demuestra ausencia: varias variables se introdujeron en fechas distintas.

Por eso no se puede afirmar cuántas imágenes “sanas” existen sólo con los agregados. Un filtro conservador futuro debería exigir simultáneamente condición 5, pilling 5, stains explícitamente `No/none`, holes explícitamente `none` y `damage` nulo, seguido de auditoría visual. Ese requisito excluirá muchos registros antiguos y puede sesgar la muestra. Para PatchCore en prendas reales, RISE sólo sería una fuente de negativos débiles, no un benchmark de verdad-terreno.

## ¿FabriD y RAW-FABRID comparten imágenes?

No hay evidencia de imágenes compartidas y todo indica que son colecciones distintas. FabriD usa fotografías RGB de teléfono/cámara, múltiples resoluciones y cuatro clases nominales; RAW-FABRID usa una máquina industrial con cámara line-scan, resolución fija y defectos agrupados en una sola clase. Los nombres, fechas, estructura y protocolos de adquisición no coinciden. Pueden compartir el dominio general de tela cruda y objetivos de investigación, pero **no deben tratarse como versiones del mismo conjunto**.

## Plan experimental recomendado

1. **Baseline supervisado en prendas:** entrenar detectores separados para CISUTAC Spots y Holes. Evaluar por AP de objetos pequeños y sensibilidad a resolución/tiling.
2. **Benchmark de anomalías:** PatchCore sobre RAW-FABRID MVTec, primero con el split provisto y luego con evaluación agrupada por captura/rollo. Reportar AUROC imagen, AUROC píxel y AUPRO.
3. **Segundo dominio industrial:** PatchCore con las normales de FabriD; reconstruir splits por captura y comparar contra un segmentador supervisado de cuatro clases.
4. **Transferencia de dominio:** evaluar modelos industriales sobre CISUTAC sin mezclar entrenamiento. Esto mide cuánto falla el traslado paño→prenda.
5. **RISE opcional:** sólo después de autorizar los 2,84 GB, descargar y construir una lista conservadora de normales con revisión visual. No usar nulos como negativos.
6. **FabricSpotDefect:** usar como benchmark complementario; mantener originales en valid/test y aumentaciones exclusivamente en train.

## Archivos reproducibles

- `notebooks/01_eda_datasets_rewear_ai.ipynb`: visualización e inspección interactiva.
- `eda_utils.py`: inventario, áreas poligonales, duplicados exactos y análisis de anotaciones.
- `generate_eda_summary.py`: regenera `reports/eda_summary.json`.
- `reports/eda_summary.json`: resultados cuantitativos locales.

## Integridad y organización

Los ZIP oficiales se conservaron. FabriD v3 ocupa aproximadamente 13 GB incluyendo ZIPs y extracción; RAW-FABRID 8,2 GB; FabricSpotDefect extraído 920 MB. Existe un intento agregado inválido y claramente renombrado como `data/fabrid_v3/FabriD_v3.failed-partial` (8,5 MB); no se usa en ningún análisis. Los archivos completos fueron validados con la prueba interna de ZIP; `RAW_FABRID_v1.zip` además coincide con el SHA-256 oficial `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`.
