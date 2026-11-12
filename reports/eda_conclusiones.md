# ReWear AI — Conclusiones del EDA

**Fecha:** 12/11/2026 · **Alcance:** datos y diseño experimental; no se entrenaron modelos.

## Resultado principal

El problema central quedó bien delimitado: los defectos de CISUTAC son reales y muy pequeños respecto de la foto completa. Al reducir la prenda a 640 px, la mediana del lado equivalente baja a **4,2 px en agujeros** y **3,0 px en manchas**. Por eso el aporte más claro y medible será comparar imagen completa contra tiles, manteniendo el mismo detector y test.

> **Lado equivalente:** lado que tendría un cuadrado con la misma área que la caja del defecto; resume su tamaño en un número fácil de comparar.

## Qué aporta cada dataset

| Dataset | Decisión | Razón principal |
|---|---|---|
| CISUTAC Holes / Spots | **Núcleo supervisado** | Son prendas usadas y tienen cajas de agujeros/manchas; responden directamente a la aplicación elegida. |
| RAW-FABRID | **Núcleo de anomalías** | Tiene muchos patches normales, máscaras y estructura lista para PatchCore. |
| FabriD v3 | **Comparación controlada** | Permite aplicar segmentación y PatchCore sobre un test común, después de corregir el split mediante un manifiesto. |
| FabricSpotDefect | **Complementario** | Suma primeros planos de manchas, pero no ofrece normales fiables y no contiene agujeros anotados en la práctica. |
| RISE | **No usar por ahora** | Sólo hay 3 muestras locales, etiquetas incompletas y ninguna caja o máscara. |

## Hallazgos por bloque

### 1. CISUTAC — Integrante 1

- Hay **407 fotos originales y 785 agujeros**, y **486 fotos originales y 2.498 manchas**. El resto son principalmente copias redimensionadas, no observaciones nuevas.
- El 58 % de agujeros y 67 % de manchas son pequeños según COCO. A 640 px, 71 % y 77 % respectivamente quedan por debajo de 8 px de lado equivalente.
- Se fijó un split derivado, agrupado por día/sesión: Holes 300/43/64 y Spots 361/59/66 para train/valid/test.
- CISUTAC no contiene prendas normales verificadas; sirve para detección supervisada, pero no para entrenar PatchCore ni medir correctamente falsos positivos en ropa sana.

### 2. RAW-FABRID — Integrante 2

- Contiene **709 originales** (507 normales, 202 defectuosas) y **19.852 patches**.
- PatchCore entrenará sólo con 14.196 `train/good` y se evaluará con 4.969 `test/good` y 687 `test/defect`.
- No hay madres compartidas entre train y test, pero muchos patches provienen de la misma captura. Por eso la incertidumbre se agrupa por `parent_raw` y se informan resultados por rollo.
- Dos máscaras no tienen imagen, CSV ni COCO asociados: se preservan y se excluyen, sin corregir el dataset original.

### 3. FabriD y FabricSpotDefect — Integrante 3

- FabriD tiene **2.407 originales RGB**, 620 declarados normales y cuatro defectos: mancha, agujero, marca de aguja e hilo faltante.
- Se confirmaron 41 derivados relacionados de 14 capturas en distintos splits. El manifiesto propuesto agrupa por fuente y mueve 27 derivados sin modificar originales.
- Quedan **607 normales concordantes**; 19 casos ambiguos se excluyen de la comparación común.
- Existe un posible atajo: 372/467 imágenes de marca de aguja son 512×512 y no hay normales con esa resolución. El modelo podría aprender el lote de captura en vez del defecto.
- En FabricSpotDefect las 3.288 anotaciones son `spot`; la clase declarada `Hole-EnBF` tiene cero instancias. Sus 2.300 aumentados COCO/YOLO son dos formatos de los mismos ejemplos.

### 4. Integración — Integrante 4

Los dominios son visualmente muy distintos: prenda completa, primer plano RGB de tela y line-scan industrial. Esto justifica estudiar transferencia, pero impide mezclar los datasets o comparar cifras de tareas diferentes como si fueran equivalentes.

La transferencia **FabriD hole/stain → CISUTAC agujero/mancha** es viable con limitaciones: se evalúa sin ajuste, se convierten las regiones predichas a cajas y se usan las mismas métricas CISUTAC. RAW-FABRID → CISUTAC queda exploratorio porque CISUTAC no tiene prendas sanas para medir falsos positivos.

## Experimentos siguientes

1. **Baseline supervisado:** YOLO11n separado para CISUTAC Holes y Spots, usando sólo originales.
2. **Aporte principal:** comparar foto completa a 640 frente a entrenamiento e inferencia con tiles de 640/20 % iniciales; variantes elegidas sólo con validación.
3. **Baseline de anomalías:** PatchCore en RAW-FABRID, con AUROC patch, AUROC píxel y AUPRO, agrupando la incertidumbre por captura madre.
4. **Comparación de paradigmas:** segmentador y PatchCore en el mismo test de FabriD, con mapas y métricas binarias compatibles.
5. **Transferencia opcional:** modelo FabriD sobre CISUTAC sin ajuste; nunca reemplaza los tres experimentos centrales.

> **AUPRO:** resume cuánto cubre el mapa de anomalía las zonas realmente defectuosas sin marcar demasiado tejido sano.

## Límites que deben decirse claramente

- Los cálculos geométricos justifican probar tiles, pero todavía no demuestran que mejore un modelo.
- La agrupación CISUTAC por día reduce fuga, aunque no prueba que una misma prenda nunca aparezca en días distintos.
- FabriD conserva sesgos de captura; RAW-FABRID es industrial y no representa fotos de e-commerce.
- Las métricas se comparan sólo cuando tarea, test y unidad coinciden. `mAP` y `AUROC` no son intercambiables.
- El protocolo queda congelado sólo después de la revisión de los cuatro integrantes.

## Decisión final del EDA

El proyecto puede avanzar a baselines. La prioridad es **CISUTAC + tiles** y **PatchCore en RAW-FABRID**; FabriD aporta la comparación controlada y la transferencia entre dominios queda como análisis complementario. No hace falta sumar datasets, nuevas etiquetas ni modelos antes de obtener esos primeros resultados.

Los detalles reproducibles están en `reports/protocolo_experimental.md`; las auditorías completas permanecen en `reports/cisutac/`, `reports/raw_fabrid/` y `reports/fabrid/`.
