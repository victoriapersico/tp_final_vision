# ReWear AI

ReWear AI es el Trabajo Práctico Final de Visión Artificial de cuatro estudiantes de tercer año de Ingeniería en Inteligencia Artificial (UdeSA). Investigamos cómo detectar y localizar defectos visuales pequeños —principalmente manchas y agujeros— en textiles y prendas usadas.

El proyecto compara detección supervisada con detección de anomalías mediante PatchCore. También estudia la pérdida de defectos pequeños al reducir resolución, el procesamiento por parches y la generalización entre paños industriales y fotografías de prendas completas. No se busca construir una aplicación comercial.

## Estado actual

La descarga y el EDA inicial están terminados. Se auditaron resoluciones, anotaciones, tamaños de defectos, normales disponibles y riesgos de fuga entre particiones. Ya están implementados los notebooks de los baselines supervisados en CISUTAC y de anomalías en RAW-FABRID; sus corridas definitivas y conclusiones siguen pendientes.

## Estructura

```text
tp_final_vision/
├── docs/          # instrucciones detalladas de datasets
├── notebooks/     # EDA y experimentos
├── reports/       # auditoría y resúmenes del EDA
├── results/       # resultados futuros (vacío por ahora)
├── scripts/       # descarga y regeneración de análisis
├── src/           # funciones reutilizables
└── data/          # datos locales; nunca se versionan
```

## Instalación

```bash
git clone https://github.com/victoriapersico/tp_final_vision.git
cd tp_final_vision
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Preparar los datasets

Los datos no están incluidos en Git. Las instrucciones verificadas de descarga, licencia, tamaños y estructura están en [`docs/DATASETS.md`](docs/DATASETS.md).

Para CISUTAC se puede usar el script conservador:

```bash
python scripts/download_datasets.py holes
python scripts/download_datasets.py spots
```

El script conserva descargas parciales y no vuelve a bajar un dataset si encuentra su estructura completa. Las fuentes de Mendeley requieren descarga manual desde sus páginas oficiales.

Por defecto el código busca `data/` dentro del repositorio. Si los datos están en otra ubicación, configurar una sola variable:

```bash
export REWEAR_DATA_DIR="/ruta/local/a/data"
```

En Windows PowerShell:

```powershell
$env:REWEAR_DATA_DIR="C:\ruta\local\a\data"
```

## Ejecutar el EDA

Desde la raíz del repositorio:

```bash
jupyter lab notebooks/01_eda_datasets_rewear_ai.ipynb
```

Para regenerar el resumen cuantitativo:

```bash
python scripts/generate_eda_summary.py
```

El informe interpretativo está en [`reports/dataset_audit.md`](reports/dataset_audit.md) y los resultados estructurados en [`reports/eda_summary.json`](reports/eda_summary.json).

## Correr los baselines en Kaggle

Los entrenamientos se corren en [Kaggle](https://www.kaggle.com) (cuenta gratuita con el teléfono verificado para habilitar GPU e internet). En `kaggle/` hay dos lanzadores que clonan este repo, bajan los datos de sus fuentes oficiales y ejecutan el notebook correspondiente; no hace falta subir datos ni copiar notebooks.

| Lanzador | Ejecuta | Datos que baja | Accelerator |
|---|---|---|---|
| `kaggle/kaggle_bloque_a.ipynb` | `02_baseline_supervisado_cisutac.ipynb` (una sesión por `DATASET`: `holes` o `spots`) | CISUTAC desde Hugging Face (~470 MB) | **GPU T4 x2** |
| `kaggle/kaggle_bloque_b.ipynb` | `03_anomalias_raw_fabrid.ipynb` | RAW-FABRID desde Mendeley (4,1 GB, con SHA-256 verificado) | **None** (corre en CPU) |

1. En Kaggle: *Create → New Notebook → File → Import Notebook* y subir el lanzador.
2. En *Settings*: elegir el *Accelerator* de la tabla e *Internet: On*.
3. Probar con `MODO_PRUEBA = True` y *Run All*.
4. Corrida real con `MODO_PRUEBA = False` y *Save Version → Save & Run All (Commit)*; sigue corriendo aunque se cierre la compu.
5. Descargar de *Output* el `results_bloque_*.zip` y descomprimirlo en la raíz del repo; también queda el notebook ejecutado con todas sus salidas.

Si una sesión se corta (límite de unas 12 h), agregar su *Output* como *Add Input* y volver a correr: los notebooks reutilizan lo que ya está en `results/`. La sesión interactiva con GPU encendida consume la cuota semanal (30 h) aunque no se esté ejecutando nada.

## Reproducibilidad del equipo

- Usar las mismas versiones y estructuras documentadas en `docs/DATASETS.md`.
- Ejecutar comandos desde la raíz del repositorio.
- Configurar `REWEAR_DATA_DIR` si los datos no están en `./data`.
- No modificar originales ni anotaciones oficiales.
- Registrar splits derivados y semillas antes de entrenar.
- No comparar directamente métricas de tareas o dominios incompatibles.

## Licencias

El código y los datos tienen licencias independientes. Los seis datasets auditados declaran CC BY 4.0; cada uso debe conservar la atribución a sus autores y citar la fuente oficial. Consultar [`docs/DATASETS.md`](docs/DATASETS.md) antes de redistribuir cualquier material.
