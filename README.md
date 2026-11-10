# ReWear AI

ReWear AI es el Trabajo Práctico Final de Visión Artificial de cuatro estudiantes de tercer año de Ingeniería en Inteligencia Artificial (UdeSA). Investigamos cómo detectar y localizar defectos visuales pequeños —principalmente manchas y agujeros— en textiles y prendas usadas.

El proyecto compara detección supervisada con detección de anomalías mediante PatchCore. También estudia la pérdida de defectos pequeños al reducir resolución, el procesamiento por parches y la generalización entre paños industriales y fotografías de prendas completas. No se busca construir una aplicación comercial.

## Estado actual

La descarga y el EDA inicial están terminados. Se auditaron resoluciones, anotaciones, tamaños de defectos, normales disponibles y riesgos de fuga entre particiones. Todavía **no se entrenaron modelos**.

Antes de trabajar, todos los integrantes y agentes deben leer [`notes/AGENTS.md`](notes/AGENTS.md), que reúne el alcance y las decisiones compartidas.

## Estructura

```text
tp_final_vision/
├── docs/          # instrucciones detalladas de datasets
├── notebooks/     # EDA y experimentos
├── notes/         # contexto y pautas compartidas
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

## Reproducibilidad del equipo

- Usar las mismas versiones y estructuras documentadas en `docs/DATASETS.md`.
- Ejecutar comandos desde la raíz del repositorio.
- Configurar `REWEAR_DATA_DIR` si los datos no están en `./data`.
- No modificar originales ni anotaciones oficiales.
- Registrar splits derivados y semillas antes de entrenar.
- No comparar directamente métricas de tareas o dominios incompatibles.

## Licencias

El código y los datos tienen licencias independientes. Los seis datasets auditados declaran CC BY 4.0; cada uso debe conservar la atribución a sus autores y citar la fuente oficial. Consultar [`docs/DATASETS.md`](docs/DATASETS.md) antes de redistribuir cualquier material.
