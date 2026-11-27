"""Versión script (.py) de kaggle_semillas_holes.ipynb, para importar en Kaggle con *Import Notebook → Link*.

Al importar un .ipynb desde una URL "raw" de GitHub, Kaggle lo crea como script y no lo ejecuta;
este archivo tiene el mismo código, así que se puede importar como script sin problemas.

# Bloque A en Kaggle: semillas y test final de CISUTAC **Holes** (corrida real)

Este notebook **no tiene lógica propia**: clona el repo, baja CISUTAC de Hugging Face, arma el split agrupado
por día y ejecuta `notebooks/02c_semillas_cisutac.ipynb`: A0, A1 (1280) y A3 con las semillas 0, 1 y 2,
evaluadas en valid y en test con las configuraciones ya congeladas en el notebook 02.

**Settings (panel derecho):** *Accelerator* = **GPU T4 x2** (o P100) · *Internet* = **On**.

1. (Opcional) Agregar con **Add Input** el Output de la corrida del notebook 02 de este dataset
   (`results_bloque_a_<dataset>.zip`): la semilla 0 reutiliza esos pesos en vez de reentrenar.
2. Probar con `MODO_PRUEBA = True` y **Run All** (unos minutos).
3. Corrida real: `MODO_PRUEBA = False` y **Save Version → Save & Run All (Commit)**. Estimado: 2–3 h de GPU por dataset.
4. Al terminar, en *Output*: `results_semillas_<dataset>.zip` (descomprimir en la raíz del repo) y el notebook 02c ejecutado.

Si una corrida se corta, agregar su Output con **Add Input** y volver a correr: se reutiliza lo ya entrenado."""


# ===== Parámetros: ya fijados para la corrida real de Holes =====
DATASET = "holes"
MODO_PRUEBA = False      # corrida real
RAMA = "sofi"            # rama del repo a usar


NOTEBOOK = "02c_semillas_cisutac.ipynb"
PARAMETROS = {"DATASET": DATASET, "MODO_PRUEBA": MODO_PRUEBA}
ETIQUETA = DATASET
ZIP_RESULTADOS = f"results_semillas_{DATASET}"
PAQUETES = "ultralytics sahi pycocotools papermill 'huggingface_hub[cli]'"

import os, subprocess, sys, shutil, glob, json
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "tp_final_vision"
DATA = Path("/tmp/rewear_data")   # fuera de /kaggle/working: los datos no ocupan lugar en el Output
os.environ["REWEAR_DATA_DIR"] = str(DATA)

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

if not REPO.exists():
    sh(f"git clone --depth 1 -b {RAMA} https://github.com/victoriapersico/tp_final_vision.git {REPO}")
else:
    sh(f"git -C {REPO} pull")
sh(f"git -C {REPO} log --oneline -1")
sh(f"{sys.executable} -m pip install -q {PAQUETES}")

sh("nvidia-smi --query-gpu=name,memory.total --format=csv")


# Si se agregó como Input el Output de una versión anterior de este notebook, se recuperan sus resultados
# para no repetir lo que ya se calculó (los notebooks reutilizan todo lo que encuentran en results/).
for prev in glob.glob("/kaggle/input/**/tp_final_vision/results", recursive=True):
    print("Recuperando resultados previos de", prev)
    shutil.copytree(prev, REPO / "results", dirs_exist_ok=True)
for z in [*glob.glob(f"/kaggle/input/**/results_bloque_a_{DATASET}.zip", recursive=True),
          *glob.glob(f"/kaggle/input/**/{ZIP_RESULTADOS}.zip", recursive=True)]:
    print("Recuperando resultados previos de", z)
    shutil.unpack_archive(z, REPO)


# CISUTAC Holes y Spots (~230 MB cada uno) y split agrupado por día (reports/cisutac/cis5_split_propuesto.csv).
# prepare_cisutac_yolo.py arma los dos datasets juntos, por eso se bajan ambos.
sh(f"cd {REPO} && {sys.executable} scripts/download_datasets.py all")
sh(f"cd {REPO} && {sys.executable} scripts/prepare_cisutac_yolo.py")


# Copia del notebook con los parámetros fijados y ejecución de punta a punta.
src = REPO / "notebooks" / NOTEBOOK
nb = json.loads(src.read_text(encoding="utf-8"))
celda = next(c for c in nb["cells"] if c["cell_type"] == "code" and "MODO_PRUEBA =" in "".join(c["source"]))
lineas = "".join(celda["source"]).splitlines(keepends=True)
for i, linea in enumerate(lineas):
    for nombre, valor in PARAMETROS.items():
        if linea.startswith(f"{nombre} ="):
            lineas[i] = f"{nombre} = {valor!r}  # fijado por el lanzador de Kaggle" + chr(10)
celda["source"] = "".join(lineas)
print("".join(l for l in lineas if "lanzador de Kaggle" in l))

entrada = REPO / "notebooks" / f"_kaggle_{NOTEBOOK}"
entrada.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
salida = WORK / f"{Path(NOTEBOOK).stem}_ejecutado_{ETIQUETA}{'_prueba' if MODO_PRUEBA else ''}.ipynb"

import papermill as pm
try:
    pm.execute_notebook(str(entrada), str(salida), cwd=str(REPO / "notebooks"), log_output=True, kernel_name="python3")
finally:
    # Aunque falle a mitad de camino, se empaqueta lo que se haya llegado a calcular.
    shutil.make_archive(str(WORK / ZIP_RESULTADOS), "zip", REPO, "results")
    print("Resultados en", WORK / f"{ZIP_RESULTADOS}.zip")
    print("Notebook ejecutado en", salida)
