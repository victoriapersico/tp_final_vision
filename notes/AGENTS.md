# AGENTS.md — ReWear AI

> Instrucciones compartidas para Codex y otros agentes que colaboren en este repositorio. **Leer antes de modificar código, datos, notebooks o documentación.** Este archivo define el alcance acordado del proyecto; no reemplaza la consigna de la materia ni el `README.md` de instalación.

## 1. Contexto académico y equipo

**ReWear AI es nuestro Trabajo Práctico Final (TPF) de Visión Artificial (I308)**, de la carrera de **Ingeniería en Inteligencia Artificial de la Universidad de San Andrés (UdeSA)**. Somos **cuatro estudiantes de tercer año**. Queremos hacer un trabajo sólido, entendible y reproducible, con una contribución metodológica propia, **sin sobreingeniería**.

La consigna del TPF pide abordar un problema con **modelos de aprendizaje profundo y datos reales**, explicar las arquitecturas y las decisiones, evaluar cuantitativamente e incluir **al menos una modificación, mejora o uso propio**. Los entregables incluyen **código, referencias/enlaces a datasets, informe de hasta 8 páginas A4, póster y defensa oral**. Como elegimos un tema propio, su aprobación corresponde al profesor según la consigna; no asumir aprobación formal solamente por comentarios favorables.

El objetivo es un **TP universitario de investigación aplicada**, no construir ahora una plataforma comercial completa ni una aplicación productiva. La posible aplicación en e-commerce sirve para motivar el problema.

## 2. La idea: ¿qué es ReWear AI?

Queremos **detectar y localizar defectos visuales pequeños en textiles**, especialmente **agujeros y manchas** en prendas usadas que se publican para reventa online. Nos interesan defectos sutiles que pueden perderse al reducir una imagen y que son difíciles de detectar con modelos convencionales.

Estudiaremos dos enfoques de aprendizaje profundo:

1. **Supervisado:** entrenar un detector o segmentador con ejemplos y etiquetas de defectos.
2. **Detección de anomalías:** entrenar con textiles **sin defectos** para aprender patrones normales y detectar regiones anómalas; **PatchCore** es el baseline previsto.

También queremos estudiar:

- **Defectos pequeños y resolución:** imagen completa redimensionada vs. análisis a mayor resolución o mediante **parches/tiles**.
- **Generalización entre dominios:** diferencias entre telas fotografiadas en entornos industriales o controlados y fotografías de **prendas completas usadas**.
- **Contribución propia:** una modificación **simple, medible y justificada** relacionada con procesamiento de detalle (por ejemplo, estrategia *coarse-to-fine* o *tiling* para defectos pequeños). **La implementación concreta todavía debe elegirse a partir de los baselines y del EDA**; no inventar una arquitectura nueva por anticipado.

Preguntas orientadoras:

- ¿Cuándo conviene detección supervisada y cuándo detección de anomalías, **en tareas y datasets comparables**?
- ¿Cuánto influye la resolución en la detección/localización de defectos pequeños?
- ¿Qué ocurre al evaluar un método fuera del dominio en el que fue desarrollado?
- ¿Nuestra modificación mejora de manera verificable un baseline y con qué costo computacional?

**No prometer equivalencia experimental** entre un detector supervisado en CISUTAC y PatchCore en RAW-FABRID: son tareas, distribuciones y anotaciones diferentes. Para comparaciones directas, usar **el mismo conjunto de evaluación, métricas compatibles y un protocolo válido**. Si no se puede, reportar resultados separados y explicar la limitación.

### Fuera del alcance actual

**No implementar ni abrir líneas de investigación nuevas sobre:**

- Robustez específica a iluminación, sombras, perspectiva y demás condiciones visuales mediante un banco de pruebas propio.
- Estimación/calibración de incertidumbre o mecanismos de abstención.
- Evaluación de suficiencia de fotografías, recomendación de tomar otra foto o análisis multivista.
- Recolección de fotografías propias, nuevas etiquetas manuales o construcción de un dataset nuevo.
- Frontend, app móvil, API comercial, despliegue cloud, autenticación, bases de datos o infraestructura MLOps.

La **generalización entre dominios sí forma parte del proyecto**; no confundirla con agregar una línea independiente de robustez a perturbaciones visuales.

## 3. Qué vimos en la materia

Somos estudiantes de tercer año. Al explicar código o tomar decisiones, apoyarse en la teoría vista en I308, pero **sin forzar técnicas irrelevantes** para este problema.

- **Fundamentos de imágenes:** representación digital, color, modelos de color, operadores puntuales, histogramas y transformaciones de intensidad.
- **Procesamiento clásico:** filtros espaciales y frecuenciales, convolución, gradientes, bordes y extracción/detección de características visuales.
- **Geometría de imágenes:** transformaciones 2D, homografías, DLT y RANSAC; modelo de cámara, calibración y perspectiva por N puntos.
- **Visión multivista y 3D:** triangulación, geometría epipolar, matrices fundamental y esencial, rectificación estéreo, disparidad, profundidad, reconstrucción 3D, SfM y SLAM.
- **Aprendizaje automático y profundo aplicado a visión:** clasificación, kNN y clasificadores lineales; regularización, optimización, redes neuronales/MLP y **redes convolucionales (CNN)**.

**Aplicación a este TP:** representación, resolución, convolución/CNN, preprocesamiento, entrenamiento, regularización, evaluación y localización de defectos son centrales. SIFT, RANSAC, calibración, estéreo o SLAM **no deben agregarse porque sí**: solo si resuelven una necesidad concreta y acordada. PatchCore incorpora un método adicional a lo visto en clase y deberá explicarse de forma comprensible, con referencias y un baseline reproducible.

## 4. Datasets y propósito de cada uno

**Usar exclusivamente datasets públicos existentes. No crear ni anotar uno propio.** Conservar los archivos originales y respetar la licencia y atribución de cada fuente. No fusionar automáticamente datasets de dominios distintos.

| Dataset | Qué contiene / anotaciones | Uso acordado | Restricción importante |
|---|---|---|---|
| **[CISUTAC Holes](https://huggingface.co/datasets/wargoninnovation/Garment_condition_holes)** | **761 imágenes** de prendas usadas, **1.497 instancias** de agujeros; principalmente cajas YOLO | **Detección supervisada en prendas completas** y posible evaluación externa de localización | Sin imágenes normales verificadas; no sirve solo para entrenar PatchCore ni para medir falsos positivos sobre prendas sanas |
| **[CISUTAC Spots](https://huggingface.co/datasets/wargoninnovation/Garment_condition_spots)** | **906 imágenes**, **4.777 instancias** de manchas; principalmente cajas YOLO | **Detección supervisada en prendas completas** y evaluación de manchas pequeñas | Mismas limitaciones respecto de normales; pocas máscaras/polígonos |
| **[FabricSpotDefect](https://data.mendeley.com/datasets/6574nhzm8x/1)** | **1.014 imágenes originales**; anotaciones COCO de manchas y otra clase de defecto tipo agujero; principalmente cajas | **Benchmark complementario supervisado** de defectos en primeros planos | Las versiones aumentadas/COCO/YOLO no son nuevos originales; no poner aumentaciones en validación/test; imágenes sin anotación no son necesariamente sanas |
| **[FabriD v3](https://data.mendeley.com/datasets/vncdwps24r/3)** | **2.407 imágenes RGB**, de ellas **620 normales en el inventario original**; agujeros, manchas, marcas de aguja e hilos faltantes; polígonos de segmentación | **Segmentación supervisada** y **segundo experimento de anomalías/PatchCore** en el mismo dominio | El EDA detectó **14 posibles capturas fuente presentes en distintos splits**; resolver antes de entrenar/evaluar formalmente |
| **[RAW-FABRID](https://data.mendeley.com/datasets/db6g85xsyg/1)** | **709 originales**, parches y máscaras; estructura tipo MVTec con **14.196 parches normales de train**, **4.969 normales de test**, **687 defectuosos de test** | **Benchmark principal de PatchCore**, localización de anomalías y **experimentos de resolución/parches** | Datos industriales; mantener independencia por imagen fuente/rollo; discrepancia entre **204 máscaras** y **202 imágenes defectuosas** en CSV |
| **[RISE — second-hand front-only](https://huggingface.co/datasets/fnauman/fashion-second-hand-front-only-rgb)** | **31.638 registros** de prendas usadas; atributos de condición, manchas y agujeros **a nivel imagen**, muchos valores faltantes | **Candidato opcional a estudiar** para imágenes de prendas completas sin defectos; **no es dataset principal ni se descargó completo** | No tiene cajas/máscaras; un nulo **no equivale** a ausencia de defecto; no usar sus supuestos «sanos» como ground truth confiable sin justificación |

**Aclaración de conteos:** cantidad de **imágenes** y cantidad de **instancias/anotaciones** son distintas. No repetir el error de llamar «1.497 imágenes» a las 1.497 instancias de CISUTAC Holes, ni «4.777 imágenes» a las 4.777 manchas de CISUTAC Spots.

Estado actual: **CISUTAC, FabricSpotDefect, FabriD v3 y RAW-FABRID descargados/inspeccionados**. RISE solo se examinó mediante metadatos y muestras. Consultar `reports/dataset_audit.md` y `reports/eda_summary.json` (o las rutas actuales equivalentes) antes de asumir que un detalle de los datos está resuelto.

## 5. Plan experimental (orden y prioridades)

### Etapa 0 — Datos y EDA (realizada; revisar hallazgos pendientes)

- Inventario, tamaños, distribución de clases, resolución, tipo de etiqueta y tamaño relativo de defectos.
- Visualización de imágenes y anotaciones.
- Auditoría de duplicados y riesgo de *data leakage*.
- Resultado actual: `01_eda_datasets_rewear_ai.ipynb`, utilidades y reporte de auditoría.

**Antes de reportar resultados de modelos:** revisar los 14 posibles duplicados entre splits de FabriD, la discrepancia de máscaras/CSV de RAW-FABRID y las particiones apropiadas de CISUTAC. Registrar las decisiones y no sobrescribir los originales.

### Etapa 1 — Baselines supervisados

- Entrenar detectores sencillos y reproducibles **por separado** en CISUTAC Holes y Spots, usando sus anotaciones de cajas.
- Utilizar **FabriD** para una baseline de segmentación supervisada si aporta una comparación clara dentro de ese mismo dataset.
- Evaluar calidad de localización y especialmente detección de defectos pequeños.
- Preferir modelos conocidos o implementaciones de referencia, citadas. Evitar desarrollar un detector desde cero si uno existente resuelve el baseline.

### Etapa 2 — Baselines de anomalías

- Implementar/usar **PatchCore** primero sobre **RAW-FABRID** (estructura tipo MVTec), con entrenamiento **solo en normales**.
- Repetir en **FabriD** con las normales adecuadas y un protocolo de evaluación consistente.
- Documentar exactamente qué datos entran en train/valid/test; las métricas por parche no equivalen automáticamente a métricas por prenda/imagen fuente.

### Etapa 3 — Nuestra contribución: defectos pequeños

- Comparar de forma controlada el procesamiento estándar con uno que **preserve más detalle**: mayor resolución, parches/tiles o estrategia simple *coarse-to-fine*.
- Elegir **una mejora principal** en función de los resultados de los baselines.
- Probar una o dos ablaciones que aíslen su efecto; no sumar cinco variantes solo por complejidad.
- Evaluar si mejora la localización/recall de defectos pequeños y registrar su costo de procesamiento.

### Etapa 4 — Generalización entre dominios

- Estudiar qué ocurre al evaluar fuera del dominio de entrenamiento: tejidos/paños industriales frente a imágenes de prendas usadas.
- **No mezclar datasets ni métricas sin justificar compatibilidad.** Documentar conversión de clases/anotaciones, diferencias de resolución y ausencia de negativos fiables en CISUTAC.
- Si algún cruce no permite una evaluación cuantitativa honesta, mostrar un análisis cualitativo acotado y explicitar sus límites. **No inventar resultados comparables.**

### Etapa 5 — Informe, póster y entrega

- Reportar metodología, particiones, baselines, mejora propia, métricas, figuras y limitaciones.
- Informe **máximo 8 páginas A4** (la cátedra sugiere estilo IEEE); póster **A0 o A1 vertical** y defensa oral, según la consigna.
- Conservar enlaces, atribuciones, versiones, hiperparámetros y semillas para que el trabajo sea reproducible.

**Estado de ejecución actual: solo EDA y preparación de datos. No entrenar modelos hasta que el equipo indique comenzar la siguiente etapa.**

## 6. Métricas y validez experimental

Elegir métricas **según tarea y anotaciones**, no una única métrica para todo:

- **Detección supervisada con cajas:** precisión, recall y AP/mAP (declarar IoU y criterios). Analizar por **tamaño relativo del defecto** cuando sea posible.
- **Segmentación:** IoU/Dice cuando existan máscaras válidas.
- **Anomalías (PatchCore):** AUROC a nivel imagen, AUROC a nivel píxel y AUPRO **solo donde existan datos/etiquetas apropiados**.
- **Mejora de resolución/parches:** comparar con el baseline en **mismo test, mismo protocolo**, e informar resolución, tamaño de parche, solapamiento, memoria y/o tiempo si son relevantes.

Prevenir *data leakage*: no colocar imágenes relacionadas o parches de una misma captura en lados independientes de train/test; considerar agrupación por prenda, captura fuente o rollo según dataset. **Fijar semillas y splits**. No ajustar hiperparámetros usando el test. Las diferencias entre datasets se describen como **cambio de dominio**, no como una comparación directa de precisión si cambian las etiquetas o las condiciones de evaluación.

## 7. Cómo deben trabajar los agentes (cuatro integrantes, un repositorio)

Estas reglas aplican a **cualquier agente usado por cualquiera de los cuatro integrantes**:

1. **Antes de empezar**, leer `AGENTS.md`, `README.md` y los archivos pertinentes; revisar el estado actual del repo. No asumir que otro agente todavía no implementó algo.
2. Hacer **solo la tarea pedida**. Si surge una mejora interesante fuera del alcance, **proponerla primero** y esperar decisión del grupo. No abrir funcionalidades ni paquetes nuevos por iniciativa propia.
3. **No sobreingenierizar:** preferir funciones y scripts pequeños, interfaces simples, y librerías ya usadas. No introducir abstracciones genéricas, factories, registries, sistemas de configuración complejos, microservicios, Docker, CI sofisticado ni frameworks de seguimiento de experimentos sin necesidad real.
4. **Separar responsabilidades sin fragmentar de más:** notebooks para narrativa, llamadas a funciones, visualizaciones y resultados; código reutilizable en `src/` (o módulos existentes). No copiar implementaciones enormes en varias celdas ni crear un archivo por función.
5. Todo el contenido dirigido al equipo (**README, notebooks, gráficos, informes y explicaciones**) debe estar **en español**; nombres de bibliotecas, clases y APIs pueden mantenerse en inglés. Código legible, con comentarios solo cuando aporten contexto.
6. **Preservar el trabajo de los demás:** no hacer refactors masivos, renombrar carpetas, borrar archivos, reescribir notebooks ajenos ni cambiar dependencias importantes sin coordinación. Si hay cambios paralelos, trabajar en ramas o PR pequeños y revisar antes de integrar a `main`.
7. Antes de dar una tarea por terminada, **verificar que el código relevante corre** (al menos una prueba pequeña), que las rutas son relativas/configurables y que los resultados se pueden regenerar. Si falta un dataset o GPU, dejar instrucciones precisas; no simular ejecuciones.
8. **No descargar varios GB, entrenar por horas, instalar dependencias pesadas ni gastar recursos de GPU/cloud sin autorización explícita**. Nunca suponer que los cuatro tienen los mismos datos o hardware.
9. Documentar decisiones experimentales en el lugar adecuado: breve explicación en notebook y, cuando corresponda, actualización del reporte/README. Registrar qué cambió, cómo se comprobó y qué queda pendiente.
10. Al finalizar una modificación, informar **archivos cambiados, pruebas hechas y cuestiones abiertas**, sin afirmar que algo funciona si no fue ejecutado.

### Repositorio y datos

- Mantener una estructura **simple** que pueda evolucionar: `notebooks/`, `src/`, `reports/`, `docs/`, `scripts/` **solo si hacen falta**; `data/` local y fuera del versionado.
- **No subir datos originales, ZIPs, pesos grandes, credenciales ni entornos**. Verificar el `.gitignore` y el listado de archivos a commitear. Si algo grande ya está trackeado, detenerse y avisar antes de modificar historial Git.
- `README.md` y `docs/DATASETS.md` deben permitir que los compañeros **descarguen por su cuenta** los datasets desde sus fuentes oficiales y los ubiquen con la estructura que usa el código. Mantener enlaces, versión **FabriD v3**, licencias y rutas verificadas.
- Evitar rutas absolutas de una computadora. Ofrecer una única forma sencilla de indicar dónde está `data/` (ruta relativa o variable de entorno documentada).
- No modificar archivos originales ni anotaciones oficiales; cualquier split derivado, conversión o corrección debe escribirse aparte, ser reproducible y quedar documentado.
- No hacer `git push`, publicar el repo, ni ejecutar acciones remotas destructivas sin indicación explícita del integrante responsable.

## 8. Criterio para considerar que el proyecto está bien hecho

Buscamos **una pregunta acotada, buenos baselines, una mejora propia identificable y evidencia cuantitativa honesta**. Importa más que cada experimento tenga sentido y pueda explicarse en la defensa que acumular modelos o funcionalidades. El resultado debe parecer trabajo cuidado de **cuatro estudiantes de tercer año**, no una plataforma de ingeniería de producción.

**Si una decisión aumenta mucho la complejidad sin ayudar a contestar nuestras preguntas de investigación, no hacerla.**
