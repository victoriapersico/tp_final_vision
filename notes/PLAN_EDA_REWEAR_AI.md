# ReWear AI — Plan de trabajo del EDA

**Trabajo Práctico Final de Visión Artificial (I308)**  
**Equipo:** 4 estudiantes de 3.er año de Ingeniería en Inteligencia Artificial  
**Horizonte del TP:** 4 semanas · **Objetivo de este documento:** organizar la primera semana de trabajo y dejar los experimentos listos para comenzar, **sin entrenar todavía modelos**.

> **Instrucción para todo el equipo y sus agentes:** este documento es el plan común del EDA. Antes de crear archivos, revisar el repositorio `tp_final_vision`, el `README.md`, `notes/AGENTS.md` y los materiales de auditoría existentes. **Reutilizar y extender el EDA ya hecho**, no rehacerlo ni construir infraestructura innecesaria.

## Actualización de estado — 11 de noviembre de 2026

El bloque **Integrante 1 — CISUTAC (CIS-1 a CIS-6)** fue completado, revisado y aceptado para continuar el EDA. Se fijan estas decisiones antes de avanzar:

1. **Unidad experimental:** usar solamente las capturas originales: **407 Holes / 785 instancias** y **486 Spots / 2.498 instancias**. Las copias de 640×640 de las mismas fotos no se usan como observaciones independientes ni se mezclan con los originales en entrenamiento/evaluación.
2. **Split CISUTAC v1 aprobado:** usar el manifiesto derivado `reports/cisutac/cis5_split_propuesto.csv`, agrupado por día de captura. No modifica los datos oficiales. Queda congelado antes del entrenamiento: Holes 300/43/64 y Spots 361/59/66 capturas en train/valid/test. No hay fechas, capturas ni IDs de sesión/prenda repetidos entre particiones.
3. **Experimento de escala aprobado:** comparar foto completa con lado mayor 640 frente a entrenamiento e inferencia mediante tiles. La configuración inicial de tiles es **640 px con 20 % de solapamiento**; tamaños/solapamientos alternativos se seleccionan únicamente con validación. El test queda reservado para la evaluación final.
4. **Hipótesis a comprobar:** preservar la escala mediante tiles debe mejorar recall y AP de defectos pequeños respecto del resize completo, a cambio de mayor costo. Los cálculos geométricos del EDA justifican el experimento, pero no prueban todavía una mejora del modelo.
5. **Siguiente bloque:** Integrante 2 continúa con RAW-FABRID (RAW-1 a RAW-6). No corresponde volver a ejecutar el inventario CISUTAC salvo que cambie la versión de los datos.

La aprobación del split es metodológica y provisional respecto de identidad física: la búsqueda visual no fue exhaustiva y un código hexadecimal identifica una sesión, no necesariamente una prenda única. Si aparece evidencia concreta de la misma prenda en días distintos, se corrige el manifiesto **antes** del entrenamiento final y se registra una nueva versión; nunca se modifica mirando resultados del test.

### Actualización Integrante 2 — RAW-FABRID

El bloque **RAW-1 a RAW-6** también fue completado. Se confirmaron 709 originales (507 normales y 202 defectuosas), 19.852 patches, 361 instancias high-res y 687 patches defectuosos con máscara. Las dos máscaras adicionales son huérfanas no vacías, sin imagen, CSV ni COCO asociados; se conservan, pero se excluyen de los experimentos.

El protocolo PatchCore primario queda definido a nivel de patch: entrenar sólo con `train/good`, evaluar con `test/good` y `test/defect`, y reportar AUROC de patch, AUROC de píxel y AUPRO. No hay madres compartidas entre train y test, pero todos los patches de test se agrupan en 202 madres y los siete rollos aparecen en ambos conjuntos. Por ello, la incertidumbre debe agruparse por `parent_raw` y los resultados deben desglosarse por rollo. Una evaluación auxiliar a nivel de original requerirá reservar madres normales completas mediante un manifiesto estratificado por rollo **antes** del entrenamiento.

### Actualización Integrante 3 — FabriD y FabricSpotDefect

El bloque **FAB-1 a FAB-6 y FSD-1** fue completado. Los 14 nombres de captura presentes en varios splits corresponden a 41 derivados visualmente relacionados, no a simples coincidencias. El manifiesto `reports/fabrid/fab4_split_agrupado.csv` los agrupa por `source_id` y mueve 27 derivados de train al split de evaluación donde ya estaba su captura.

La conciliación de normales deja 607 casos concordantes, 13 originales `defect free` con polígonos y 6 etiquetas vacías sin original normal. Los 19 ambiguos se excluyen del protocolo común. La comparación segmentador/PatchCore es viable con limitaciones sobre un test compartido de 85 normales y 278 defectuosas, usando mapas binarios y métricas compatibles. Debe declararse el posible atajo de captura: 372/467 `Needle Mark Defect` son 512×512, resolución ausente en normales.

FabricSpotDefect queda como complemento de manchas: COCO declara `Hole-EnBF`, pero sus 3.288 anotaciones pertenecen a `spot` y `Hole-EnBF` tiene cero instancias. Las 36 imágenes train sin cajas no se consideran normales verificadas; los 2.300 aumentados en COCO y YOLO son dos representaciones de los mismos ejemplos fuente.

### Actualización Integrante 4 — Integración y protocolo

El bloque **INT-1 a INT-7** fue completado el 12 de noviembre de 2026. El notebook final se redujo a una lectura guiada de aproximadamente 10 minutos: integra los cuatro aportes con la secuencia pregunta, evidencia y decisión, mientras conserva enlaces a los análisis detallados.

Se fijaron el protocolo, las métricas y la unidad de evaluación en `reports/protocolo_experimental.md`. La comparación directa supervisado/PatchCore se limita al test común de FabriD; la transferencia FabriD `hole/stain` → CISUTAC `agujero/mancha` es viable con limitaciones y sin ajuste, mientras que RAW-FABRID → CISUTAC queda exploratoria. RISE no se descargará ni se usará en esta etapa.

El EDA técnico queda cerrado. El único paso humano pendiente es que los cuatro integrantes lean y aprueben el protocolo antes de comenzar los entrenamientos.

---

## 0. Objetivo y alcance del proyecto

**ReWear AI** estudia cómo detectar y localizar defectos visuales **pequeños**, principalmente **agujeros y manchas**, en fotografías de ropa usada destinadas a publicaciones de e-commerce.

El objetivo académico no es prometer un detector comercial perfecto. Queremos formular hipótesis verificables, establecer baselines y medir qué pasa cuando los defectos son diminutos y las imágenes provienen de contextos distintos.

**Preguntas que guiarán el TP:**

1. ¿Qué tan bien localiza un detector supervisado los agujeros y manchas pequeños de fotografías reales de prendas usadas?
2. ¿Qué impacto tiene conservar la escala del defecto mediante **tiling** (recortes/tiles), frente a redimensionar la fotografía completa? ¿Cuál es su costo computacional?
3. ¿Qué puede detectar un método de anomalías como **PatchCore**, entrenado con muestras normales, sobre tejidos industriales?
4. Cuando los datos lo permiten, ¿cómo se comparan los métodos supervisados y de anomalías **dentro de un mismo dominio**, con métricas compatibles?
5. ¿Cuánto se degrada el desempeño al pasar de imágenes industriales de tejidos a fotos de prendas usadas, cuando exista una comparación técnicamente válida?

**Aporte metodológico esperado:** evaluación controlada de **escala, tamaño de tiles y solapamiento** para defectos pequeños, más una comparación bien delimitada de paradigmas supervisado/anomalías. *SAHI y PatchCore son baselines/herramientas existentes; usarlos no constituye por sí solo una arquitectura nueva.*

### Fuera de alcance en esta etapa

- No entrenar ni ajustar modelos durante el EDA.
- No implementar aún SAHI, PatchCore, YOLO ni otro detector; el EDA solo debe justificar cómo se evaluarán.
- No hacer análisis adicional de robustez a iluminación/sombras como línea independiente.
- No implementar incertidumbre, abstención ni evaluación de **suficiencia de fotografías**.
- No descargar RISE completo por ahora ni recolectar, crear o etiquetar datasets propios.
- No fusionar los datasets, inventar negativos o cambiar anotaciones originales.

---

## 1. Punto de partida: lo que ya está hecho

Codex ya descargó y auditó los conjuntos principales; **no repetir inventarios extensos sin un motivo**. Los materiales existentes que deben consultarse primero son:

- `01_eda_datasets_rewear_ai.ipynb` (o su ubicación actual bajo `notebooks/`).
- `dataset_audit.md` (o su ubicación bajo `reports/`).
- `eda_summary.json` (bajo `reports/` si fue reorganizado).
- `eda_utils.py` y el script que genera el resumen, si están en el repo.

Verificar las rutas reales antes de escribir código; estos nombres describen archivos de la auditoría, **no obligan a reorganizar carpetas**.

### Datasets y función concreta

| Dataset | Datos verificados en la auditoría | Normales confiables para anomalías | Anotaciones | Papel en el TP |
|---|---|---|---|---|
| [CISUTAC Holes](https://huggingface.co/datasets/wargoninnovation/Garment_condition_holes) | **761 imágenes; 1.497 agujeros anotados** | No | Mayormente cajas; 3 polígonos | **Principal**: detección supervisada de agujeros en prendas usadas |
| [CISUTAC Spots](https://huggingface.co/datasets/wargoninnovation/Garment_condition_spots) | **906 imágenes; 4.777 manchas anotadas** | No | Mayormente cajas; 99 polígonos | **Principal**: detección supervisada de manchas en prendas usadas |
| [RAW-FABRID](https://data.mendeley.com/datasets/db6g85xsyg/1) | **709 imágenes originales; 19.852 patches** | Sí: 14.196 patches `train/good` | Cajas, polígonos y máscaras | **Principal para anomalías**: benchmark PatchCore y escala |
| [FabriD v3](https://data.mendeley.com/datasets/vncdwps24r/3) | **2.407 imágenes; 620 originales sin defectos** | Sí, con verificación de correspondencia | Clases y polígonos YOLO-seg | Segundo dominio: segmentación supervisada y comparación controlada con anomalías |
| [FabricSpotDefect](https://data.mendeley.com/datasets/6574nhzm8x/1) | **1.014 originales; 3.288 anotaciones originales** | No verificadas | Cajas y algunas segmentaciones | **Complementario**: manchas/agujeros en primeros planos |
| [RISE front-only](https://huggingface.co/datasets/fnauman/fashion-second-hand-front-only-rgb) | 31.638 registros remotos; solo **3 muestras locales** | No verificados; etiquetas incompletas | Etiquetas débiles a nivel imagen | **Opcional y fuera del EDA principal**. No usar como verdad-terreno ni como negativos seguros |

**No confundir:** 1.497 y 4.777 son **instancias de defectos** de CISUTAC, no cantidad de imágenes. Tampoco asumir que fotos sin una anotación explícita están sanas.

### Hallazgos conocidos que hay que aprovechar

- La mediana del área anotada en CISUTAC es ínfima: **Holes ≈ 0,00697 %**, **Spots ≈ 0,00385 %** del área de imagen. Casi todas las instancias están por debajo de 1 %. El EDA debe traducir esto a **anchos y altos en píxeles**, antes y después de resize.
- **CISUTAC no tiene imágenes normales verificadas ni split `test` oficial** (solo train/valid). No se puede evaluar sobre él una tasa de falsos positivos en prendas sanas ni AUROC imagen normal/anómalo de forma completa.
- **FabriD:** se detectaron **14 posibles capturas fuente** presentes en más de un split; confirmar antes de definir nuevas particiones. Además, hay **620** normales originales frente a **613 etiquetas vacías** en el export anotado: investigar correspondencia, no suponer error automáticamente.
- **RAW-FABRID:** el CSV identifica **202** originales defectuosas, pero hay **204** máscaras; comprobar los casos concretos y documentarlos. Los patches contienen identificador de imagen madre (`parent_raw`) y rollo; no tratar patches de una misma captura como observaciones independientes.
- **FabricSpotDefect:** las versiones aumentadas aparecen bajo COCO y YOLO; no son ejemplos independientes. Los ejemplos sin anotación no se consideran normales sin verificación.

---

## 2. Organización del equipo: cuatro responsables y una entrega conjunta

**Regla:** una persona lidera cada bloque, otra lo revisa; todos comparten criterios y contribuyen al notebook final. No hacen cuatro EDAs desconectados.

| Responsable | Bloque | Entrega primaria | Revisión cruzada sugerida |
|---|---|---|---|
| **Integrante 1** | **EDA CISUTAC Holes + Spots** | Distribuciones, escala real, ejemplos y evaluación posible en prendas | Integrante 4 |
| **Integrante 2** | **EDA RAW-FABRID** | Anomalías, máscaras, trazabilidad de patches, discrepancias y splits | Integrante 3 |
| **Integrante 3** | **EDA FabriD v3 + chequeo breve de FabricSpotDefect** | Segmentación, normales, diferencias entre particiones y posibles duplicados | Integrante 2 |
| **Integrante 4** | **Integración y protocolo experimental** | Notebook integrado, comparación de dominios, políticas de splits y métricas | Integrante 1; revisión final de todos |

**Para repartir las tareas:** sustituir “Integrante 1–4” por los nombres acordados en GitHub Issues o en el tablero del grupo. No hace falta agregar una herramienta nueva de gestión.

---

## 3. Protocolo común para todos los bloques

Antes de empezar con tareas específicas:

- [x] **C0.1 — Revisión de materiales.** Leer `dataset_audit.md`, notebook previo y scripts utilitarios. Enumerar qué resultados ya existen y cuáles falta producir. **Resultado:** una lista corta de faltantes, no otro inventario duplicado.
- [x] **C0.2 — Confirmación de rutas.** Verificar que cada integrante tiene acceso a los datasets locales indicados en el README. Usar rutas configurables y relativas al proyecto; nunca codificar rutas de una computadora.
- [x] **C0.3 — Mismas definiciones.** Registrar para cada defecto: ancho y alto de la caja en píxeles (`w`, `h`), área de caja (`w×h`), área relativa (`área caja / (ancho imagen × alto imagen)`); si hay máscara/polígono, registrar su área **por separado**. No mezclar áreas de bounding boxes con áreas de máscaras.
- [x] **C0.4 — Reproducibilidad.** Guardar semillas para las visualizaciones aleatorias, listar versiones de datasets y explicar cada filtro. No modificar fotos, anotaciones ni splits oficiales en disco: las alternativas se proponen mediante **manifiestos separados**.
- [x] **C0.5 — Visualizaciones útiles.** Cada figura debe contestar una pregunta de investigación. Priorizar pocas figuras claras y representativas, con título, unidades, cantidad de observaciones y conclusiones breves. No generar galerías gigantes.
- [x] **C0.6 — Privacidad del test.** El test final, una vez fijado, no se usa para elegir hiperparámetros ni tamaños de tiles. Las decisiones del EDA deben basarse preferentemente en train/valid; si se inspecciona test para auditar estructura o integridad, registrarlo y **no usar sus resultados para optimizar modelos**.

**Convención sugerida:** un notebook principal `notebooks/01_eda_datasets_rewear_ai.ipynb` con funciones reutilizables en un archivo existente (`eda_utils.py` o `src/eda_utils.py`, según la estructura real). Archivos de salida compactos bajo `reports/`. No agregar frameworks, bases de datos, dashboards ni clases complejas.

---

## 4. Tareas detalladas por integrante

### Integrante 1 — CISUTAC: prendas usadas y tamaño real de los defectos

**Objetivo:** justificar por qué el tamaño de los defectos exige estudiar la escala/tiling, y determinar qué evaluación supervisada es posible.

- [x] **CIS-1. Validar estructura y splits.** Confirmar las **761 imágenes de Holes** (697 train, 64 valid) y **906 de Spots** (830 train, 76 valid), sus etiquetas YOLO, categorías y ausencia de normales/test. Comparar con la auditoría; explicar solo discrepancias nuevas.
  - **Entregable:** tabla de inventario simple de ambos datasets.
- [x] **CIS-2. Ver ejemplos con cajas.** Mostrar una grilla pequeña con imágenes reales de train/valid y sus bounding boxes, y otra con el zoom de la zona defectuosa. Incluir algunos casos fáciles, difíciles y con múltiples instancias; la selección debe ser reproducible.
  - **Entregable:** 2–3 figuras comentadas y breve nota sobre calidad/visibilidad de etiquetas.
- [x] **CIS-3. Distribución de defectos.** Calcular defectos por imagen, cantidad de cajas, ancho/alto absolutos, área absoluta y área relativa. Graficar histograma (escala logarítmica si ayuda) y reportar percentiles **p10, p50, p90**; separar Holes y Spots.
  - **Entregable:** tabla de cuantiles + 1–2 gráficos; porcentajes de defectos por debajo de umbrales documentados.
- [x] **CIS-4. Simular resize sin entrenar.** Para una selección representativa, calcular cómo cambia el **tamaño de los defectos en píxeles** al redimensionar la imagen completa a resoluciones candidatas (por ejemplo, lado máximo 640 y 1024, conservando proporción). Mostrar uno o dos ejemplos con zoom antes/después. No afirmar pérdida real de información sin observar el resultado; registrar si quedan cajas de pocos píxeles.
  - **Entregable:** tabla comparativa de `w`, `h` y área en píxeles original vs. resize; implicación para resolución/tiling.
- [x] **CIS-5. Auditar splits y relaciones.** Buscar hashes exactos, nombres similares y, si es factible, posibles duplicados perceptuales. Si hay identificadores de prenda/sesión, proponer agrupación. No inventar identidades de prendas a partir de parecido visual ambiguo.
  - **Entregable:** reporte de pares sospechosos y decisión **provisional** sobre test independiente (no aplicar un split final antes de consensuarlo).
- [x] **CIS-6. Propuesta de evaluación.** Explicar por qué el proyecto necesita evaluar por separado agujeros y manchas inicialmente; proponer `mAP@0.5`, `mAP@0.5:0.95`, precision, recall y una definición explícita de **defecto pequeño**. Aclarar que `AP_small` COCO tiene una convención propia (área menor a `32²` px en el sistema de coordenadas de la evaluación) y que los umbrales personalizados **se deben definir antes del entrenamiento**.
  - **Entregable:** 1 sección «Implicaciones para el baseline supervisado».

**Criterio de terminado:** otra persona del equipo puede explicar, con números y ejemplos, **por qué no alcanza con hacer resize tradicional** y qué medición hará falta para verificar la mejora por tiles.

### Integrante 2 — RAW-FABRID: anomalías, máscaras y dependencia entre patches

**Objetivo:** determinar qué tan confiable es el benchmark para PatchCore y cómo evaluar anomalías sin fugas de información ni pseudorreplicación.

- [x] **RAW-1. Confirmar inventario existente.** Verificar 709 originales de 1792×1024, distribución normal/defectuosa, 19.852 parches, máscaras y agrupación por rollo. Documentar cómo están codificados los canales: PIL informa RGB, aunque las imágenes podrían ser visualmente monocromáticas.
  - **Entregable:** inventario resumido; no volver a generar el conteo masivo ya hecho si no cambió nada.
- [x] **RAW-2. Visualizar ejemplos correctamente.** Mostrar originales normales/defectuosos y superponer máscaras; añadir ejemplos de patches con y sin defecto y su relación con la imagen madre.
  - **Entregable:** una grilla de originales/máscaras y otra de patches.
- [x] **RAW-3. Medir escala de anomalías.** A partir de COCO/máscaras, calcular área relativa, bbox y percentiles; comparar tamaño del defecto en imagen original vs. patch, **sin confundir un cambio de denominador con un cambio en el defecto físico**.
  - **Entregable:** tabla de percentiles y conclusión sobre la utilidad de la resolución original/parches.
- [x] **RAW-4. Resolver discrepancia 202/204.** Revisar las dos imágenes indicadas en la auditoría, cruzando `has_defect`, COCO y máscaras. Documentar si se trata de discrepancia de metadatos, máscaras vacías, etc. **No cambiar etiquetas oficiales en esta etapa.**
  - **Entregable:** tabla «archivo → fuentes que concuerdan/discrepan → recomendación de tratamiento experimental».
- [x] **RAW-5. Auditar dependencia y posibles fugas.** Usar `parent_raw` y `roll_id` para comprobar la separación train/test de imágenes madres; observar que `test/good` y `test/defect` comparten imágenes madres de test, lo que **no es por sí mismo fuga train–test**, pero sí implica dependencia estadística. Verificar la composición por rollo.
  - **Entregable:** tabla de imágenes madres/rollos por split, intersecciones y riesgo estadístico.
- [x] **RAW-6. Diseñar evaluación PatchCore (sin implementarlo).** Definir qué partición corresponde a `train/good`, qué conjuntos se usan para AUROC imagen, AUROC píxel y AUPRO, y a qué unidad corresponde cada métrica. **Importante:** el split de patches ofrecido por la fuente contiene parches normales incluso dentro de capturas defectuosas; decidir explícitamente si el AUROC será por **patch**, **imagen original** o ambos, y qué agregación requeriría el nivel original. Proponer evaluación por captura/rollo si resulta viable.
  - **Entregable:** protocolo preliminar de anomalías y principales limitaciones.

**Criterio de terminado:** el equipo puede describir **qué recibe PatchCore como datos normales, qué evalúa y cuáles son las unidades independientes**.

### Integrante 3 — FabriD v3 y control acotado de FabricSpotDefect

**Objetivo:** preparar un segundo dominio compatible con métodos supervisados y de anomalías, evitando problemas de particiones.

- [x] **FAB-1. Confirmar esquema y clases.** Documentar imágenes RGB originales, anotaciones YOLO-seg, clases `stain`, `hole`, `needle mark`, `missing yarn`, normales originales y export redimensionado.
  - **Entregable:** resumen de clases y tipos de anotación.
- [x] **FAB-2. Visualizar y comparar normal/defecto.** Mostrar ejemplos de cada clase con polígonos y, por separado, ejemplos normales. Comparar visualmente condiciones de captura y resoluciones; documentar si normales y defectuosas provienen de contextos similares o hay indicios de un atajo visual para el modelo.
  - **Entregable:** 2 figuras + evaluación breve del posible sesgo de captura.
- [x] **FAB-3. Áreas y tamaños.** Calcular número de instancias, distribuciones de tamaño de polígonos y sus cajas equivalentes, percentiles y diferencias entre clases; identificar casos muy pequeños.
  - **Entregable:** tabla por clase y un gráfico comparativo.
- [x] **FAB-4. Investigar 14 capturas repetidas.** Inspeccionar y confirmar si corresponden realmente a la misma foto fuente, a derivados o a coincidencias de nombre. Proponer un **split agrupado por imagen/captura fuente**, preservando la separación train/valid/test; registrar una tabla de afectados y justificación. No modificar datos originales.
  - **Entregable:** reporte de leakage y propuesta de nuevo manifiesto de particiones.
- [x] **FAB-5. Conciliar las normales.** Investigar la diferencia entre **620 imágenes normales originales** y **613 etiquetas vacías en el export**. Distinguir imágenes faltantes, nombres transformados y etiquetas vacías que no correspondan a la misma fuente.
  - **Entregable:** tabla de correspondencia o explicación documentada de lo que no se pudo verificar.
- [x] **FAB-6. Factibilidad de comparación supervisado vs. PatchCore.** Proponer un conjunto de prueba común, separado por fuente, con normales y defectuosas; explicar qué etiquetas sirven para **detección/segmentación** y cuáles para **clasificación de anomalías**. La comparación deberá compartir imágenes y métricas compatibles; no comparar sin más mAP con AUROC.
  - **Entregable:** esquema de splits y métricas comparables, pendiente de aprobación grupal.
- [x] **FSD-1. Chequeo complementario de FabricSpotDefect (acotado).** Verificar los **1.014 originales**, las clases reales `spot` y `Hole-EnBF`, el número de segmentaciones, la representación duplicada de las 2.300 imágenes aumentadas y el papel de los 36 train sin anotaciones. Mostrar 2–3 ejemplos con cajas. No volver a realizar un EDA exhaustivo.
  - **Entregable:** una tabla corta sobre si aporta algo distinto a CISUTAC/FabriD y si conviene incluirlo en los experimentos finales.

**Criterio de terminado:** el equipo sabe si FabriD permite una **comparación controlada válida** y cómo manejar los registros potencialmente filtrados entre splits.

### Integrante 4 — Integración, comparación de dominios y protocolo de evaluación

**Objetivo:** convertir los tres EDAs en un plan experimental coherente y reproducible; no se limita a «juntar gráficos».

- [x] **INT-1. Validar entregables y coherencia.** Revisar definiciones de tamaño, clases, nombres, particiones y unidades utilizadas por los otros integrantes. Corregir inconsistencias **en los análisis derivados**, nunca en las anotaciones originales.
  - **Entregable:** tabla consolidada de datos y advertencias metodológicas.
- [x] **INT-2. Comparar dominios.** Construir una tabla que contraste **fotografías de prendas usadas**, **tejidos RGB en condiciones variables** e **inspección industrial controlada**: resolución, fondo, formato, clases, máscaras/cajas, presencia de normales y pertinencia comercial. Mostrar una sola figura con ejemplos representativos de cada dominio.
  - **Entregable:** tabla + figura + interpretación del riesgo de domain shift.
- [x] **INT-3. Comprobar factibilidad de transferencia.** Determinar qué clases pueden mapearse legítimamente (`hole`/`stain`) y cuáles no. Proponer cómo evaluar **sin ajuste** un modelo industrial sobre CISUTAC, si la salida y las etiquetas lo hacen posible; si no existe comparación válida, declarar el experimento exploratorio o descartarlo. No usar una cifra entre métricas incompatibles para declarar caída de rendimiento.
  - **Entregable:** decisión «viable / viable con limitaciones / no viable», por pareja de datasets y tipo de modelo.
- [x] **INT-4. Fijar métricas y protocolo preliminar.** Preparar `reports/protocolo_experimental.md` con: tarea por dataset, unidad de evaluación, métricas, definición de defecto pequeño, política de splits, criterios de comparación, semillas y qué se selecciona por validación. Incluir inferencia con/sin tiles, tamaños/solapamientos como **variables posteriores**, no como configuraciones elegidas mirando test.
  - **Entregable:** protocolo para aprobar entre los cuatro antes de entrenar.
- [x] **INT-5. Integrar notebook.** Actualizar el notebook existente con las secciones del EDA que faltan. Mantener código mínimo visible y cálculos reutilizables en funciones. No pegar implementaciones largas dentro de celdas ni duplicar gráficos. Explicar en español **qué muestra cada resultado y qué decisión permite tomar**.
  - **Entregable:** un `01_eda_datasets_rewear_ai.ipynb` legible y reproducible, más figuras compactas en `reports/` cuando ayuden al informe.
- [x] **INT-6. Cerrar el EDA con decisiones.** Redactar `reports/eda_conclusiones.md`: resultados relevantes, limitaciones, datasets elegidos, criterios de splits, 2 baselines previstos y experimentos propuestos para las semanas siguientes.
  - **Entregable:** resumen de 1–2 páginas que se pueda utilizar para el informe y explicarle al profesor la metodología.
- [x] **INT-7. Revisión RISE (solo documentación).** Mantener anotado por qué **no** se usará por ahora: 3 muestras locales, campos `holes`/`stains` incompletos, ausencia de localizaciones. No descargar ni etiquetar más datos salvo decisión explícita del grupo.

**Criterio de terminado:** cualquier integrante puede explicar **qué se entrenará después, con qué datos y cómo se medirá**, sin contradicciones entre notebook, informe y metodología.

---

## 5. Estructura propuesta del notebook final de EDA

**Extender el notebook existente; una sección por objetivo, no uno por dataset si eso produce repeticiones.**

1. **Introducción y preguntas del EDA:** por qué defectos pequeños, e-commerce y cambio de dominio.
2. **Inventario de datasets:** unidades claras (imágenes vs. instancias vs. parches), splits y tipos de etiquetas.
3. **Ejemplos visuales anotados:** vistas completas y zoom de defectos; normales cuando existan.
4. **Distribución de defectos:** frecuencia, cantidad por imagen, áreas y dimensiones en píxeles y porcentajes.
5. **Resolución y resize:** simulación de escalas y tamaño visible del defecto; justificación preliminar de tiling.
6. **Integridad y etiquetas:** máscaras, cajas, duplicados, casos inconsistentes y datos faltantes.
7. **Splits y leakage:** agrupación por captura/prenda/rollo y decisiones propuestas.
8. **Comparación entre dominios:** diferencias visuales y compatibilidad de tareas.
9. **Implicaciones para la evaluación:** qué métricas son válidas con cada dataset.
10. **Conclusiones y protocolo:** decisiones concretas, incertidumbres pendientes y próximos experimentos.

Para cada sección: **pregunta → cálculo/visualización → hallazgo → decisión**. Si una figura no cambia una decisión ni ayuda a justificarla, omitirla.

---

## 6. Decisiones que el EDA debe dejar listas antes del entrenamiento

| Decisión | Hipótesis inicial | Qué tiene que confirmar el EDA |
|---|---|---|
| **Dataset supervisado principal** | **Aprobado:** CISUTAC Holes/Spots por separado, usando sólo originales | Split v1 agrupado por día en `reports/cisutac/cis5_split_propuesto.csv`; revisar sólo si aparece evidencia nueva de identidad física |
| **Baseline supervisado** | **Aprobado en protocolo:** un detector preentrenado basado en CNN (baseline preliminar YOLO11n) | La configuración final y loss se documentan al iniciar entrenamiento; el resultado preliminar a 640 no es una métrica final |
| **Escala / tiling** | **Aprobado:** foto completa a 640 vs. entrenamiento+inferencia por tiles; inicio 640/20 % | Elegir variantes sólo con valid; medir AP_small, recall por tamaño y costo en el test congelado |
| **Baseline de anomalías** | **Protocolo EDA aprobado:** PatchCore con `train/good` de RAW-FABRID | Evaluación primaria por patch; incertidumbre agrupada por madre y desglose por rollo. El nivel original requiere un manifiesto auxiliar de madres normales |
| **Comparación entre enfoques** | **Viable con limitaciones en FabriD:** test común y mapas binarios | Usar manifiesto agrupado, 607 normales concordantes y excluir 19 ambiguos; declarar sesgo de lote/resolución |
| **Transferencia industrial → ropa usada** | Prueba complementaria sin fine-tuning | Compatibilidad real de clases/salidas y existencia de métricas comparables |
| **FabricSpotDefect** | Complemento, no núcleo | Si añade información útil sin sobrecargar el alcance |
| **RISE** | No usar por ahora | No requiere más análisis durante esta primera semana |

### Métricas previstas para el trabajo posterior

- **Detección supervisada:** precision, recall, F1, `mAP@0.5`, `mAP@0.5:0.95`, resultados por clase y sensibilidad en defectos pequeños. Fijar las definiciones en el protocolo; cuando se reporte `AP_small`, aclarar la escala/convención utilizada.
- **Detección de anomalías:** AUROC de clasificación de anomalía a nivel de **unidad definida** (patch o imagen), AUROC a nivel de píxel, **AUPRO** cuando existan máscaras y protocolo compatible. No inventar métricas de localización para datasets solo con cajas.
- **Costo:** tiempo de inferencia por imagen original, cantidad de tiles generados y, cuando corresponda, consumo de memoria/recursos bajo hardware comparable.
- **Generalización:** reportar cambio de una **misma métrica** para una **misma tarea** y clases compatibles en dominios distintos. Si no se puede, presentar análisis cualitativo/exploratorio sin conclusiones cuantitativas injustificadas.

**Nota metodológica:** el profesor sugirió SAHI y Anomalib como herramientas. Durante el EDA solo verificamos la factibilidad del protocolo; todavía no implementamos modelos. Más adelante, la loss supervisada depende del detector efectivamente elegido. PatchCore no usa una loss supervisada estándar de cajas/clases.

---

## 7. Cronograma realista: primera semana de EDA

| Momento | Actividad | Responsables | Resultado de salida |
|---|---|---|---|
| **Día 1** | Leer auditoría, acordar definiciones y rutas, asignar tareas, revisar trabajo existente | Los 4 | Lista de faltantes y responsabilidades |
| **Días 2–3** | EDA específico de CISUTAC, RAW-FABRID, FabriD/FabricSpotDefect en paralelo | Integrantes 1, 2 y 3; Integrante 4 coordina protocolo | Figuras, tablas y hallazgos revisables |
| **Día 4** | Revisión cruzada, resolver discrepancias prioritarias, integrar y redactar protocolo | Los 4; Integrante 4 integra | Notebook consolidado y protocolo preliminar |
| **Día 5** | Revisar juntos el notebook de punta a punta, aprobar criterios de splits, métricas y baselines | Los 4 | EDA terminado y diseño experimental congelado |

**Si algo no se resuelve en esta semana:** registrarlo con impacto y decisión conservadora. No demorar todo el TP intentando explicar cada anomalía secundaria. Los riesgos que **sí** bloquean métricas honestas (leakage, test compartido, clases mal definidas) deben tratarse antes del entrenamiento/evaluación correspondiente.

### Panorama del mes completo (solo contexto)

- **Semana 1:** EDA + splits + protocolo y métricas aprobados.
- **Semana 2:** baselines reproducibles: detector supervisado en CISUTAC y PatchCore en RAW-FABRID; documentar loss/configuración cuando aplique.
- **Semana 3:** experimentos controlados de tiling (p. ej., SAHI, tamaños y solapamiento); comparación controlada en FabriD si es válida; transferencia exploratoria si alcanza el tiempo.
- **Semana 4:** ablaciones, métricas finales, análisis de errores, tiempos y costo, informe de máximo 8 páginas, póster y defensa.

**Prioridad si se acorta el tiempo:** EDA confiable + baseline supervisado + tiling bien evaluado + PatchCore de referencia. FabriD y transferencia deben sumar rigor, no desplazar el núcleo del TP.

---

## 8. Colaboración en GitHub sin sobreingeniería

- Trabajar sobre **`tp_final_vision`**; respetar el README y el `notes/AGENTS.md`.
- Preferir **una rama o PR pequeña por bloque** del EDA; no editar simultáneamente las mismas celdas del notebook principal. Para evitar conflictos, los integrantes pueden generar tablas/figuras mediante módulos o resultados derivados y el Integrante 4 integra el notebook.
- Antes de hacer commit: ejecutar la parte modificada, verificar rutas relativas, confirmar que los gráficos muestran unidades correctas y que no se rompió el EDA existente.
- **No subir `data/`, ZIPs, imágenes originales, pesos de modelos, cachés ni archivos pesados.** Sí subir código, notebooks limpios, reportes derivados compactos y documentación. Respetar licencias y atribuciones.
- **No añadir herramientas de orquestación, DVC, pipelines automáticos complejos ni más modelos** por iniciativa propia. Una función reutilizable clara es mejor que una abstracción genérica.
- No modificar los originales; cualquier split nuevo o corrección de metadatos va en un archivo derivado con justificación.
- Cada responsable escribe en su entrega: **qué calculó, hallazgo, limitación y recomendación**. No solo gráficos.

---

## 9. Checklist de aceptación del EDA (reunión de cierre)

El EDA se considera terminado cuando el equipo puede marcar **todo lo aplicable**:

### Datos y visualizaciones

- [x] Se describieron correctamente las unidades **imagen / defecto / patch / captura fuente**.
- [x] Hay ejemplos anotados y zoom de defectos para los datasets principales.
- [x] Se reportan tamaños absolutos/relativos y percentiles por clase cuando corresponda.
- [x] Hay evidencia visual y numérica de qué ocurre al redimensionar las imágenes.
- [x] Se distinguen fotos de ropa usada de imágenes industriales; no se fusionaron datos.

### Calidad y splits

- [x] Se investigaron los 14 posibles duplicados de FabriD y se definió cómo evitar leakage.
- [x] Se documentó la diferencia 202 vs. 204 de RAW-FABRID y su tratamiento experimental.
- [x] Se revisó trazabilidad `parent_raw`/`roll_id` y dependencia entre patches.
- [x] Se definió una política honesta para CISUTAC, que no tiene test oficial ni normales.
- [x] No se trataron etiquetas faltantes/archivos sin cajas como negativos automáticamente.
- [x] Los splits futuros se podrán reproducir y no se cambian basándose en resultados del test.

### Metodología y documentación

- [x] Se asignó un dataset y una tarea concreta a cada experimento propuesto.
- [x] Están definidas las métricas y su **unidad de evaluación**; se evitarán comparaciones incompatibles.
- [x] El protocolo explica qué significa «defecto pequeño» para los análisis posteriores.
- [x] El notebook de EDA corre con las rutas documentadas y no modifica datos originales.
- [x] `reports/eda_conclusiones.md` y `reports/protocolo_experimental.md` resumen decisiones, límites y pasos siguientes.
- [ ] Los cuatro integrantes revisaron y aprobaron el protocolo antes de entrenar.

---

## 10. Entregables concretos de esta semana

1. **Notebook EDA consolidado**: `notebooks/01_eda_datasets_rewear_ai.ipynb` (adaptar al nombre real si ya fue movido).
2. **Utilidades reutilizables**: extender `eda_utils.py` existente; evitar varios módulos que hagan lo mismo.
3. **Resultados derivados compactos**: `reports/` con tablas o figuras seleccionadas, sin datos crudos.
4. **Conclusiones ejecutivas**: `reports/eda_conclusiones.md` (1–2 páginas de hallazgos y decisiones).
5. **Protocolo experimental**: `reports/protocolo_experimental.md` (splits, métricas, baselines, hipótesis, pruebas posteriores y restricciones de comparación).
6. **Lista corta de pendientes**: hallazgos bloqueantes/no bloqueantes, responsables y decisiones tomadas.

### Preguntas que todos deben poder responder al terminar

- ¿Cuáles son **exactamente** los datasets principales y por qué?
- ¿Qué tamaño tienen los defectos y cuántos píxeles conservarían después de un resize?
- ¿Qué experimentos se pueden comparar de forma válida y cuáles **no**?
- ¿Cómo evitamos fuga de datos con prendas, capturas y parches relacionados?
- ¿Con qué baseline empezaríamos, qué mediríamos y cuál sería el cambio experimental propio?
- ¿Qué límites hay que comunicar explícitamente al profesor para no exagerar conclusiones?

> **Criterio final:** un EDA completo no es el que contiene más gráficos. Es el que deja **decisiones justificadas y reproducibles** para entrenar baselines y evaluar una mejora real durante las tres semanas siguientes.

---

**Fuentes de las cantidades y alertas:** auditoría del equipo `dataset_audit.md` (10/11/2026) y `eda_summary.json`. Las métricas, tareas y comparaciones futuras de este documento son **propuestas metodológicas**, no resultados de modelos ya entrenados.
