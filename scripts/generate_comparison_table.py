"""Junta en una tabla las métricas de todos los experimentos (E1–E4) y la escribe en reports/.

Sólo lee archivos versionados en el repositorio (no hace falta tener datos ni GPU):

- E1 (CISUTAC, detección con cajas): los metricas.json de reports/cisutac/predicciones_bloque_a/. La A3 de
  Holes con presupuesto extendido no tiene metricas.json: se lee de la salida del notebook ejecutado 02b.
- E2 (RAW-FABRID, anomalías): tablas de salida del notebook ejecutado 03 y la diferencia pareada escrita en
  el texto de sus conclusiones.
- E3 (FabriD v3, segmentación vs. anomalías): reports/fabrid/04_validacion_v3.json (valid) y las tablas de
  salida del notebook ejecutado 04b (test).
- Resultados que todavía no existen (test de Holes, YOLO26n, E4): lista PENDIENTES, sin números.

Las celdas de los notebooks se buscan por su código, no por su número. Antes de escribir se controlan
conteos, umbrales y coherencia entre fuentes; si algo no coincide, el script se detiene y dice qué falló.

Escribe reports/tabla_comparativa.csv (una fila por modelo, variante y split) y reports/tabla_comparativa.md
(una sección por experimento). Sólo se comparan filas del mismo `grupo_comparacion` (experimento y split).

Uso:
  python scripts/generate_comparison_table.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREDICCIONES_CISUTAC = REPORTS / "cisutac" / "predicciones_bloque_a"
EJECUTADOS = ROOT / "notebooks" / "ejecutados"
NB_02B = EJECUTADOS / "02b_ejecutado_a3_extendida_holes.ipynb"
NB_03 = EJECUTADOS / "03_anomalias_raw_fabrid_resultados.ipynb"
NB_04B = EJECUTADOS / "04b_ejecutado_test_fabrid.ipynb"
JSON_04 = REPORTS / "fabrid" / "04_validacion_v3.json"
SALIDA_CSV = REPORTS / "tabla_comparativa.csv"
SALIDA_MD = REPORTS / "tabla_comparativa.md"

COLUMNAS = [
    "experimento", "dataset", "tarea", "paradigma", "modelo", "variante_id", "variante", "corrida", "split",
    "estado", "fue_a_test", "grupo_comparacion", "unidad_evaluacion", "n_evaluadas",
    # Detección con cajas (E1, E4): precisión, recall y F1 por caja con IoU ≥ 0,5 y umbral elegido en valid.
    "map50", "map50_95", "ap_small", "precision_caja", "recall_caja", "recall_caja_ic95_inf",
    "recall_caja_ic95_sup", "f1_caja", "f1_caja_ic95_inf", "f1_caja_ic95_sup", "umbral_confianza",
    "tiles_por_captura",
    # Anomalías y segmentación (E2, E3): precisión, recall y F1 por patch con umbral elegido en valid.
    "auroc_patch", "auroc_patch_ic95_inf", "auroc_patch_ic95_sup", "auroc_imagen", "auroc_imagen_ic95_inf",
    "auroc_imagen_ic95_sup", "auroc_pixel", "aupro_0.3", "precision_patch", "recall_patch", "f1_patch",
    "dice_micro", "iou_micro",
    "segundos_por_unidad", "fuente", "nota",
]

UNIDAD_CAJA = "caja (IoU ≥ 0,5) en la captura original"
DATASET_E1 = {"holes": "CISUTAC Holes", "spots": "CISUTAC Spots"}
EXPERIMENTO_E1 = {"holes": "E1-H", "spots": "E1-S"}
# Capturas originales por split según reports/protocolo_experimental.md (Holes test todavía no evaluado).
CAPTURAS_E1 = {("holes", "valid"): 43, ("spots", "valid"): 59, ("spots", "test"): 66}

# (entrenamiento, inferencia) → (ID, descripción). A2 reutiliza el modelo de A0 con tiles sólo en inferencia.
VARIANTES_E1 = {
    ("A0_completa640", "completa640"): ("A0", "Foto completa 640"),
    ("A1_completa1280", "completa1280"): ("A1", "Foto completa 1280"),
    ("A0_completa640", "tiles640_o20"): ("A2", "Modelo A0 + tiles 640 en inferencia"),
    ("A0_completa640", "tiles640_o20+completa"): ("A2", "Modelo A0 + tiles 640 y foto completa en inferencia"),
    ("A3_tiles640_neg1", "tiles640_o20"): ("A3", "Tiles 640, 1 negativo por positivo"),
    ("A3_tiles640_neg1", "tiles640_o20+completa"): ("A3", "Tiles 640 neg1 + foto completa en inferencia"),
    ("A3_tiles640_neg0", "tiles640_o20"): ("A3-abl", "Tiles 640, sin negativos"),
    ("A3_tiles640_neg3", "tiles640_o20"): ("A3-abl", "Tiles 640, 3 negativos por positivo"),
    ("A3_tiles1024_neg1", "tiles1024_o20"): ("A3-abl", "Tiles 1024, 1 negativo por positivo"),
    ("A3_tiles640_neg1_ext100", "tiles640_o20"): ("A3-abl", "Tiles 640 neg1, presupuesto extendido a 100 épocas"),
}

# Corrida del notebook 03 → (ID, modelo, descripción). Configuración base de las celdas de parámetros.
VARIANTES_E2 = {
    "B0_estadistica_local": ("B0", "Estadística local", "clásico, media y desvío en ventana 15×15"),
    "B1_padim_resnet18": ("B1", "PaDiM", "ResNet18, 100 canales"),
    "B2_patchcore_resnet18_c0.01": ("B2", "PatchCore", "base: ResNet18, coreset 1 %, 256 px, 500 normales"),
    "B2_patchcore_wide_resnet50_2_c0.01": ("B2-abl", "PatchCore", "base con WideResNet50"),
    "B2_patchcore_resnet18_c0.1": ("B2-abl", "PatchCore", "base con coreset 10 %"),
    "B2_patchcore_resnet18_c0.01_n100": ("B2-abl", "PatchCore", "base con 100 normales"),
    "B2_patchcore_resnet18_c0.01_n2000": ("B2-abl", "PatchCore", "base con 2000 normales"),
    "B2_patchcore_resnet18_c0.01_r128": ("B2-abl", "PatchCore", "base con patches reducidos a 128 px"),
}

_E1_H = {"experimento": "E1-H", "dataset": "CISUTAC Holes", "tarea": "detección con cajas",
         "paradigma": "supervisado", "unidad_evaluacion": UNIDAD_CAJA}
PENDIENTES = [
    {**_E1_H, "modelo": "YOLO26n", "variante_id": "A3", "variante": "Tiles 640, 1 negativo por positivo",
     "split": "valid", "fue_a_test": "a definir (issue #7)", "nota": ["En curso en el issue #6."]},
    {**_E1_H, "modelo": "YOLO11n", "variante_id": "A0", "variante": "Foto completa 640", "split": "test",
     "nota": ["Test de Holes todavía no evaluado."]},
    {**_E1_H, "modelo": "YOLO11n o YOLO26n", "variante_id": "A3",
     "variante": "Elegida en el issue #7, con selección de checkpoint según el diagnóstico del issue #5",
     "split": "test", "nota": ["Test de Holes todavía no evaluado."]},
    {"experimento": "E4", "dataset": "FabriD v3 → CISUTAC (test congelado)", "tarea": "detección con cajas",
     "paradigma": "supervisado", "modelo": "YOLO11n-seg de E3", "variante_id": "",
     "variante": "Transferencia sin ajuste, máscaras convertidas a cajas: hole → agujero, stain → mancha",
     "split": "test", "unidad_evaluacion": UNIDAD_CAJA, "nota": ["Falta el best.pt del E3."]},
]

CONTROLES = []      # mensajes de los controles aprobados, para mostrarlos al final
DETALLES = {}       # grupo_comparacion → tamaño del conjunto evaluado, para el markdown
DIFERENCIAS = {}    # experimento → texto de la diferencia pareada en test


def exigir(condicion, mensaje):
    if not condicion:
        raise SystemExit(f"ERROR: {mensaje}")


def control(condicion, mensaje):
    exigir(condicion, f"control fallido: {mensaje}")
    CONTROLES.append(mensaje)


def fila(valores):
    exigir(set(valores) <= set(COLUMNAS), f"columnas desconocidas: {set(valores) - set(COLUMNAS)}")
    resultado = {c: valores.get(c) for c in COLUMNAS}
    resultado["nota"] = list(valores.get("nota") or [])
    resultado["grupo_comparacion"] = f"{resultado['experimento']}/{resultado['split']}"
    return resultado


# ---------------------------------------------------------------------------------------------------------
# Lectura de notebooks ejecutados

class _LectorTabla(HTMLParser):
    """Filas de una tabla HTML como listas de textos (th y td)."""

    def __init__(self):
        super().__init__()
        self.filas, self._fila, self._celda = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._fila = []
        elif tag in ("th", "td") and self._fila is not None:
            self._celda = ""

    def handle_data(self, data):
        if self._celda is not None:
            self._celda += data

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._celda is not None:
            self._fila.append(self._celda.strip())
            self._celda = None
        elif tag == "tr" and self._fila is not None:
            self.filas.append(self._fila)
            self._fila = None


def leer_tabla_html(html):
    """DataFrame de pandas mostrado en un notebook → {índice: {columna: texto}}."""
    lector = _LectorTabla()
    lector.feed(html)
    columnas = lector.filas[0][1:]
    # La fila con el nombre del índice (p. ej. "corrida") tiene todas las celdas vacías y se descarta.
    datos = [f for f in lector.filas[1:] if len(f) == len(columnas) + 1 and any(f[1:])]
    return {f[0]: dict(zip(columnas, f[1:])) for f in datos}


@lru_cache(maxsize=None)
def _celdas(notebook):
    exigir(notebook.is_file(), f"no existe {notebook.relative_to(ROOT)}")
    return json.loads(notebook.read_text(encoding="utf-8"))["cells"]


def buscar_celda(notebook, fragmento):
    """(número, celda) de la única celda del notebook cuyo texto contiene `fragmento`."""
    encontradas = [(i, c) for i, c in enumerate(_celdas(notebook)) if fragmento in "".join(c["source"])]
    exigir(len(encontradas) == 1,
           f"{notebook.name}: se esperaba una celda con «{fragmento}» y hay {len(encontradas)}")
    return encontradas[0]


def tablas_html(celda):
    return [leer_tabla_html("".join(o["data"]["text/html"])) for o in celda.get("outputs", [])
            if o["output_type"] in ("display_data", "execute_result") and "text/html" in o["data"]]


def texto_impreso(celda):
    return "".join("".join(o["text"]) for o in celda.get("outputs", []) if o["output_type"] == "stream")


def fuente_notebook(notebook, numero):
    return f"{notebook.relative_to(ROOT).as_posix()}, celda {numero}"


def numero(texto):
    return None if texto in ("", "NaN", "None") else float(texto)


def con_intervalo(texto):
    """'0.868 (0.824–0.908)' → (0.868, 0.824, 0.908)."""
    m = re.fullmatch(r"([\d.]+) \(([\d.]+)–([\d.]+)\)", texto)
    exigir(m is not None, f"valor con intervalo inesperado: {texto!r}")
    return tuple(float(x) for x in m.groups())


# ---------------------------------------------------------------------------------------------------------
# E1 — CISUTAC

def partes_corrida(corrida):
    """'spots_A3_tiles640_neg0__test_tiles640_o20' → ('spots', 'A3_tiles640_neg0', 'test', 'tiles640_o20')."""
    dataset, resto = corrida.split("_", 1)
    entrenamiento, evaluacion = resto.split("__")
    split, inferencia = evaluacion.split("_", 1)
    return dataset, entrenamiento, split, inferencia


def fila_e1(m, fuente, nota=()):
    dataset, entrenamiento, split, inferencia = partes_corrida(m["corrida"])
    exigir((entrenamiento, inferencia) in VARIANTES_E1, f"variante de E1 sin descripción: {m['corrida']}")
    variante_id, descripcion = VARIANTES_E1[(entrenamiento, inferencia)]
    ic_recall = m["ic95_recall"] or (None, None)
    ic_f1 = m["ic95_f1"] or (None, None)
    return fila({
        "experimento": EXPERIMENTO_E1[dataset], "dataset": DATASET_E1[dataset], "tarea": "detección con cajas",
        "paradigma": "supervisado", "modelo": "YOLO11n", "variante_id": variante_id, "variante": descripcion,
        "corrida": m["corrida"], "split": split, "estado": "evaluado", "unidad_evaluacion": UNIDAD_CAJA,
        "n_evaluadas": m["imagenes"], "map50": m["mAP@0.5"], "map50_95": m["mAP@0.5:0.95"],
        "ap_small": m["AP_small"], "precision_caja": m["precision"], "recall_caja": m["recall"],
        "recall_caja_ic95_inf": ic_recall[0], "recall_caja_ic95_sup": ic_recall[1], "f1_caja": m["f1"],
        "f1_caja_ic95_inf": ic_f1[0], "f1_caja_ic95_sup": ic_f1[1], "umbral_confianza": m["umbral_confianza"],
        "tiles_por_captura": m["tiles_por_imagen"], "segundos_por_unidad": m["segundos_por_imagen"],
        "fuente": fuente, "nota": list(nota),
    })


def cargar_e1():
    archivos = sorted(PREDICCIONES_CISUTAC.glob("*/*/metricas.json"))
    exigir(archivos, f"no hay metricas.json en {PREDICCIONES_CISUTAC.relative_to(ROOT)}")
    metricas = [json.loads(a.read_text(encoding="utf-8")) for a in archivos]
    control(len({frozenset(m) for m in metricas}) == 1,
            f"E1: los {len(archivos)} metricas.json tienen las mismas claves")
    for archivo, m in zip(archivos, metricas):
        dataset, _, split, _ = partes_corrida(m["corrida"])
        exigir(m["corrida"] == archivo.parent.name and (dataset, split) == (m["dataset"], m["split"]),
               f"nombre de carpeta y contenido no coinciden en {archivo.relative_to(ROOT)}")

    control(not any(m["dataset"] == "holes" and m["split"] == "test" for m in metricas),
            "E1-H: no hay corridas de test de Holes en el repo (las filas pendientes siguen vigentes)")
    for (dataset, split), capturas in CAPTURAS_E1.items():
        grupo = [m for m in metricas if (m["dataset"], m["split"]) == (dataset, split)]
        cajas = {m["cajas_reales"] for m in grupo}
        control(grupo and {m["imagenes"] for m in grupo} == {capturas} and len(cajas) == 1,
                f"E1 {dataset} {split}: las {len(grupo)} corridas evalúan {capturas} capturas y "
                f"{min(cajas)} cajas reales")
        DETALLES[f"{EXPERIMENTO_E1[dataset]}/{split}"] = f"{capturas} capturas originales, {min(cajas)} cajas reales"

    por_clave = {partes_corrida(m["corrida"]): m for m in metricas}
    for (dataset, entrenamiento, split, inferencia), m in por_clave.items():
        if split != "test":
            continue
        valid = por_clave.get((dataset, entrenamiento, "valid", inferencia))
        control(valid is not None and valid["pesos"] == m["pesos"]
                and valid["umbral_confianza"] == m["umbral_confianza"],
                f"E1 {m['corrida']}: mismos pesos y umbral de confianza ({m['umbral_confianza']}) que en valid")

    filas = []
    for archivo, m in sorted(zip(archivos, metricas), key=lambda par: par[1]["corrida"]):
        nota = ["Umbral de confianza fijado en valid."] if m["split"] == "test" else []
        filas.append(fila_e1(m, archivo.relative_to(ROOT).as_posix(), nota))
    filas.append(cargar_e1_extendida(next(m for m in metricas if m["corrida"] ==
                                          "holes_A0_completa640__valid_completa640")))

    evaluadas_test = {(f["experimento"], partes_corrida(f["corrida"])[1], partes_corrida(f["corrida"])[3])
                      for f in filas if f["split"] == "test"}
    for f in filas:
        if f["split"] != "valid":
            continue
        _, entrenamiento, _, inferencia = partes_corrida(f["corrida"])
        if (f["experimento"], entrenamiento, inferencia) in evaluadas_test:
            f["fue_a_test"] = "sí"
        elif f["experimento"] == "E1-H" and f["variante_id"] == "A0":
            f["fue_a_test"] = "pendiente"
        elif f["experimento"] == "E1-H" and f["variante_id"].startswith("A3"):
            f["fue_a_test"] = "a definir (issue #7)"
        else:
            f["fue_a_test"] = "no"
    return filas


def cargar_e1_extendida(referencia):
    """A3 de Holes con presupuesto extendido (notebook 02b): sólo valid, sin metricas.json en el repo."""
    numero_celda, celda = buscar_celda(NB_02B, "eval_ext = cb.evaluate(")
    tablas = tablas_html(celda)
    exigir(len(tablas) == 2, "02b: se esperaban la tabla de métricas y la de recall por tamaño")
    (corrida, t), = tablas[0].items()
    (_, por_tamano), = tablas[1].items()
    cajas = {k: int(v.split("/")[1]) for k, v in por_tamano.items()}
    cajas_ref = {k: int(v.split("/")[1]) for k, v in referencia["encontrados_por_tamano"].items()}
    control(cajas == cajas_ref,
            f"E1-H A3 extendida (02b): mismas cajas reales por tamaño que las demás corridas de Holes valid "
            f"({sum(cajas.values())})")
    recall, lo, hi = con_intervalo(t["recall (IC 95 %)"])
    m = {"corrida": corrida, "imagenes": referencia["imagenes"], "mAP@0.5": numero(t["mAP@0.5"]),
         "mAP@0.5:0.95": numero(t["mAP@0.5:0.95"]), "AP_small": numero(t["AP_small"]),
         "precision": numero(t["precision"]), "recall": recall, "ic95_recall": (lo, hi), "f1": numero(t["f1"]),
         "ic95_f1": None, "umbral_confianza": numero(t["umbral_confianza"]),
         "tiles_por_imagen": numero(t["tiles_por_imagen"]), "segundos_por_imagen": numero(t["segundos_por_imagen"])}
    nota = ["Sin metricas.json en el repo; valores de la salida del notebook 02b, con 3 decimales y sin IC de F1."]
    return fila_e1(m, fuente_notebook(NB_02B, numero_celda), nota)


# ---------------------------------------------------------------------------------------------------------
# E2 — RAW-FABRID

def fila_e2(nombre, t, split, n_patches, fuente):
    variante_id, modelo, descripcion = VARIANTES_E2[nombre]
    auroc, lo, hi = con_intervalo(t["AUROC_patch (IC 95 %)"])
    return fila({
        "experimento": "E2", "dataset": "RAW-FABRID", "tarea": "anomalías", "paradigma": "anomalías",
        "modelo": modelo, "variante_id": variante_id, "variante": descripcion, "corrida": nombre, "split": split,
        "estado": "evaluado", "unidad_evaluacion": "patch 256 px y píxel", "n_evaluadas": n_patches,
        "auroc_patch": auroc, "auroc_patch_ic95_inf": lo, "auroc_patch_ic95_sup": hi,
        "auroc_pixel": numero(t["AUROC_pixel"]), "aupro_0.3": numero(t["AUPRO@0.3"]),
        "precision_patch": numero(t["precision"]), "recall_patch": numero(t["recall"]), "f1_patch": numero(t["f1"]),
        "segundos_por_unidad": numero(t["segundos_por_patch"]), "fuente": fuente,
        "nota": ["Valores con 3 decimales, como los muestra el notebook 03."],
    })


def cargar_e2():
    numero_celda, celda = buscar_celda(NB_03, "split_test_parents")
    (particiones,) = tablas_html(celda)
    conteo = {s: {k: int(v) for k, v in particiones[s].items()} for s in ("valid", "test")}
    control(sum(conteo[s]["normales"] for s in conteo) == 4969 and sum(conteo[s]["defectuosos"] for s in conteo) == 687,
            f"E2: valid ({conteo['valid']['patches']}) + test ({conteo['test']['patches']}) cubren los "
            f"4969 test/good + 687 test/defect del protocolo")
    for s, c in conteo.items():
        DETALLES[f"E2/{s}"] = f"{c['patches']} patches de {c['madres']} imágenes madre, {c['defectuosos']} defectuosos"

    numero_base, celda = buscar_celda(NB_03, 'base = ["B0_estadistica_local"')
    base = tablas_html(celda)[0]
    numero_abl, celda = buscar_celda(NB_03, "ABL_BACKBONE = ")
    ablaciones = tablas_html(celda)[0]
    b2 = "B2_patchcore_resnet18_c0.01"
    # Se compara por valor: pandas muestra 5120.0 o 5120 según haya NaN en la columna.
    como_valor = lambda t: {k: v if "(" in v else numero(v) for k, v in t.items()}
    control(como_valor(base[b2]) == como_valor(ablaciones[b2]),
            "E2: el PatchCore base es idéntico en la tabla base y en la de ablaciones")
    valid = {**base, **ablaciones}
    exigir(set(valid) == set(VARIANTES_E2), f"E2: variantes inesperadas en valid: {set(valid) ^ set(VARIANTES_E2)}")

    _, celda = buscar_celda(NB_03, "candidatas = ")
    elegida = re.search(r"Mejor PatchCore en valid: (\S+)", texto_impreso(celda)).group(1)
    patchcores = [n for n in valid if n.startswith("B2_")]
    control(elegida == max(patchcores, key=lambda n: con_intervalo(valid[n]["AUROC_patch (IC 95 %)"])[0]),
            f"E2: la PatchCore que fue a test ({elegida}) es la de mayor AUROC por patch en valid")
    numero_test, celda = buscar_celda(NB_03, "if EVALUAR_TEST:")
    test = tablas_html(celda)[0]
    control(set(test) == {"B0_estadistica_local", "B1_padim_resnet18", elegida},
            "E2: test evaluó sólo B0, B1 y la PatchCore elegida en valid")

    filas = []
    for nombre in VARIANTES_E2:
        numero_fuente = numero_base if nombre in base else numero_abl
        f = fila_e2(nombre, valid[nombre], "valid", conteo["valid"]["patches"], fuente_notebook(NB_03, numero_fuente))
        f["fue_a_test"] = "sí" if nombre in test else "no"
        if nombre == elegida:
            f["nota"].insert(0, "Elegida en valid por AUROC por patch.")
        filas.append(f)
    for nombre in VARIANTES_E2:
        if nombre in test:
            filas.append(fila_e2(nombre, test[nombre], "test", conteo["test"]["patches"],
                                 fuente_notebook(NB_03, numero_test)))

    # La diferencia pareada no es una salida de celda: está escrita en las conclusiones del notebook.
    _, celda = buscar_celda(NB_03, "bootstrap pareado por imagen madre")
    m = re.search(r"diferencia B2 − B1 con bootstrap pareado por imagen madre \(([^)]*)\): "
                  r"([+-]\d+,\d+); IC 95 % \[([+-]\d+,\d+), ([+-]\d+,\d+)\]", "".join(celda["source"]))
    exigir(m is not None, "E2: no se encontró la diferencia pareada B2 − B1 en las conclusiones del notebook 03")
    estimacion, lo, hi = (float(x.replace(",", ".")) for x in m.groups()[1:])
    DIFERENCIAS["E2"] = (f"Diferencia pareada B2 − B1 (PatchCore elegida − PaDiM) en AUROC por patch: "
                         f"{estimacion:+.3f} (IC 95 %: {lo:+.3f} a {hi:+.3f}; {m.group(1)}). Tomada del texto de "
                         f"conclusiones del notebook 03, no de una salida de celda.")
    return filas


# ---------------------------------------------------------------------------------------------------------
# E3 — FabriD v3

def fila_e3(nombre, m, split, descripcion, fuente, ic=(None, None), nota=()):
    supervisado = nombre == "YOLO11n-seg"
    return fila({
        "experimento": "E3", "dataset": "FabriD v3", "tarea": "segmentación" if supervisado else "anomalías",
        "paradigma": "supervisado" if supervisado else "anomalías", "modelo": nombre, "variante_id": "",
        "variante": descripcion, "corrida": nombre, "split": split, "estado": "evaluado",
        "unidad_evaluacion": "imagen y píxel (mapa binario)", "n_evaluadas": int(m["imagenes"]),
        "auroc_imagen": float(m["AUROC_imagen"]), "auroc_imagen_ic95_inf": ic[0], "auroc_imagen_ic95_sup": ic[1],
        "auroc_pixel": float(m["AUROC_pixel"]), "aupro_0.3": float(m["AUPRO@0.3"]),
        "dice_micro": float(m["Dice_micro"]), "iou_micro": float(m["IoU_micro"]),
        "segundos_por_unidad": float(m["segundos_por_imagen"]), "fuente": fuente, "nota": list(nota),
    })


def cargar_e3():
    validacion = json.loads(JSON_04.read_text(encoding="utf-8"))
    config = validacion["configuracion"]
    metricas_valid = validacion["metricas_valid"]
    control(validacion["test_evaluado"] is False and config["modo_prueba"] is False
            and all(m["split"] == "valid" and m["imagenes"] == 375 and m["modo_prueba"] is False
                    for m in metricas_valid.values()),
            "E3: la validación del JSON es completa (375 imágenes) y se hizo con test cerrado")

    numero_celda, celda = buscar_celda(NB_04B, "if not ABRIR_TEST:")
    tablas = tablas_html(celda)
    exigir(len(tablas) == 2, "04b: se esperaban la tabla de métricas de test y la de intervalos")
    metricas_test, intervalos = tablas
    control(set(metricas_test) == set(metricas_valid) == {"YOLO11n-seg", "PatchCore"}
            and all(m["split"] == "test" and m["imagenes"] == "363" for m in metricas_test.values()),
            "E3: test evaluó YOLO11n-seg y PatchCore sobre las mismas 363 imágenes")
    control(all(math.isclose(float(metricas_test[n]["umbral_valid"]), metricas_valid[n]["umbral_valid"], abs_tol=1e-5)
                for n in metricas_valid), "E3: los umbrales usados en test son los fijados en valid")
    ic = {k: json.loads(v["IC95"]) for k, v in intervalos.items()}
    estimaciones = {k: float(v["estimacion"]) for k, v in intervalos.items()}
    diferencia = "Diferencia AUROC_imagen YOLO-PatchCore"
    control(all(math.isclose(estimaciones[f"{n} AUROC_imagen"], float(metricas_test[n]["AUROC_imagen"]), abs_tol=1e-6)
                for n in metricas_test)
            and math.isclose(estimaciones[diferencia], float(metricas_test["YOLO11n-seg"]["AUROC_imagen"])
                             - float(metricas_test["PatchCore"]["AUROC_imagen"]), abs_tol=1e-5),
            "E3: la tabla de intervalos coincide con las métricas de test y con su diferencia")
    for split, tabla in (("valid", metricas_valid), ("test", metricas_test)):
        m = tabla["YOLO11n-seg"]
        DETALLES[f"E3/{split}"] = f"{int(m['imagenes'])} imágenes de {int(m['fuentes'])} fuentes"

    sup, pc = config["supervisado"], config["pc"]
    descripcion = {"YOLO11n-seg": f"imgsz {sup['imgsz']}, hasta {sup['epochs']} épocas, paciencia {sup['patience']}",
                   "PatchCore": f"{pc['backbone'].replace('resnet', 'ResNet')}, {pc['size']} px, "
                                f"coreset {pc['coreset'] * 100:g} %"}
    nota = {"YOLO11n-seg": [], "PatchCore": []}
    if pc["size"] != 128:
        nota["PatchCore"].append("No es la configuración elegida en E2 (patches reducidos a 128 px).")
    fuente_valid = JSON_04.relative_to(ROOT).as_posix()
    filas = [fila_e3(n, metricas_valid[n], "valid", descripcion[n], fuente_valid, nota=nota[n]) for n in metricas_valid]
    for f in filas:
        f["fue_a_test"] = "sí"
    filas += [fila_e3(n, metricas_test[n], "test", descripcion[n], fuente_notebook(NB_04B, numero_celda),
                      ic=ic[f"{n} AUROC_imagen"], nota=nota[n] + ["Umbral fijado en valid."]) for n in metricas_test]
    lo, hi = ic[diferencia]
    DIFERENCIAS["E3"] = (f"Diferencia pareada YOLO11n-seg − PatchCore en AUROC por imagen: "
                         f"{estimaciones[diferencia]:+.3f} (IC 95 %: {lo:+.3f} a {hi:+.3f}; bootstrap pareado por "
                         f"`source_id`, 2000 réplicas).")
    return filas


# ---------------------------------------------------------------------------------------------------------
# Salidas

ORDEN_EXPERIMENTO = {"E1-H": 0, "E1-S": 1, "E2": 2, "E3": 3, "E4": 4}
ORDEN_ID = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A3-abl": 4, "B0": 0, "B1": 1, "B2": 2, "B2-abl": 3, "": 0}


def ordenar(filas):
    return sorted(filas, key=lambda f: (ORDEN_EXPERIMENTO[f["experimento"]], f["split"] != "valid",
                                        ORDEN_ID[f["variante_id"]], f["estado"] == "pendiente"))


def escribir_csv(filas):
    with SALIDA_CSV.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        for f in filas:
            escritor.writerow({c: (" ".join(v) if c == "nota" else "" if v is None
                                   else round(v, 6) if isinstance(v, float) else v) for c, v in f.items()})


def metrica(titulo, clave, ic=None, decimales=3, negrita=True):
    return {"titulo": titulo, "clave": clave, "ic": ic, "decimales": decimales, "negrita": negrita}


def texto(titulo, clave):
    return {"titulo": titulo, "clave": clave, "texto": True}


IC_RECALL = ("recall_caja_ic95_inf", "recall_caja_ic95_sup")
IC_F1 = ("f1_caja_ic95_inf", "f1_caja_ic95_sup")
IC_AUROC_PATCH = ("auroc_patch_ic95_inf", "auroc_patch_ic95_sup")
IC_AUROC_IMAGEN = ("auroc_imagen_ic95_inf", "auroc_imagen_ic95_sup")
COLUMNAS_E1 = [
    texto("ID", "variante_id"), texto("Variante", "variante"), texto("Modelo", "modelo"),
    metrica("mAP@0.5", "map50"), metrica("mAP@0.5:0.95", "map50_95"), metrica("AP_small", "ap_small"),
    metrica("Precisión", "precision_caja"), metrica("Recall (IC 95 %)", "recall_caja", IC_RECALL),
    metrica("F1 (IC 95 %)", "f1_caja", IC_F1), metrica("Umbral conf.", "umbral_confianza", negrita=False, decimales=2),
    metrica("Tiles/captura", "tiles_por_captura", negrita=False, decimales=1),
    metrica("s/captura", "segundos_por_unidad", negrita=False),
]
COLUMNAS_E2 = [
    texto("ID", "variante_id"), texto("Modelo", "modelo"), texto("Variante", "variante"),
    metrica("AUROC patch (IC 95 %)", "auroc_patch", IC_AUROC_PATCH), metrica("AUROC píxel", "auroc_pixel"),
    metrica("AUPRO@0.3", "aupro_0.3"), metrica("Precisión patch", "precision_patch"),
    metrica("Recall patch", "recall_patch"), metrica("F1 patch", "f1_patch"),
    metrica("s/patch", "segundos_por_unidad", negrita=False),
]
COLUMNAS_E3 = [
    texto("Modelo", "modelo"), texto("Paradigma", "paradigma"), texto("Variante", "variante"),
    metrica("AUROC imagen (IC 95 %)", "auroc_imagen", IC_AUROC_IMAGEN), metrica("AUROC píxel", "auroc_pixel"),
    metrica("AUPRO@0.3", "aupro_0.3"), metrica("Dice", "dice_micro"), metrica("IoU", "iou_micro"),
    metrica("s/imagen", "segundos_por_unidad", negrita=False),
]
COLUMNAS_E4 = [texto("Modelo", "modelo"), texto("Variante", "variante"), metrica("mAP@0.5", "map50"),
               metrica("AP_small", "ap_small"), metrica("Recall (IC 95 %)", "recall_caja", IC_RECALL)]

SECCIONES = [
    ("E1-H", "E1-H — CISUTAC Holes: detección de agujeros con cajas (supervisado)", COLUMNAS_E1),
    ("E1-S", "E1-S — CISUTAC Spots: detección de manchas con cajas (supervisado)", COLUMNAS_E1),
    ("E2", "E2 — RAW-FABRID: detección de anomalías (entrenamiento sólo con tela normal)", COLUMNAS_E2),
    ("E3", "E3 — FabriD v3: segmentación supervisada vs. anomalías", COLUMNAS_E3),
    ("E4", "E4 — Transferencia FabriD → CISUTAC sin ajuste", COLUMNAS_E4),
]
INTRO_SECCION = {
    "E1-H": "YOLO11n preentrenado en COCO. Compara reducir la foto completa (A0, A1) con recorrerla en tiles (A2, A3).",
    "E1-S": "Mismo diseño que E1-H, con manchas.",
    "E2": "Ningún método ve defectos en entrenamiento. Comparación con un método clásico (B0) y con PaDiM (B1).",
    "E3": ("**Única comparación directa entre paradigmas:** mismas imágenes de test, mismos mapas binarios por "
           "píxel y umbral de cada método fijado en valid. Las métricas por clase del segmentador previstas en el "
           "protocolo todavía no se calcularon."),
    "E4": "Modelo de E3 aplicado sin ajuste al test de CISUTAC, con las métricas de detección de E1.",
}
# Métricas del resumen de test: (clave, nombre, intervalo). Para E1 el mAP no tiene IC, así que se suma recall.
RESUMEN = {
    "E1-H": [("map50", "mAP@0.5 (métrica principal, sin IC)", None), ("recall_caja", "Recall de cajas", IC_RECALL)],
    "E1-S": [("map50", "mAP@0.5 (métrica principal, sin IC)", None), ("recall_caja", "Recall de cajas", IC_RECALL)],
    "E2": [("auroc_patch", "AUROC por patch (criterio de selección)", IC_AUROC_PATCH)],
    "E3": [("aupro_0.3", "AUPRO@0.3 (criterio de selección en valid, sin IC)", None),
           ("auroc_imagen", "AUROC por imagen", IC_AUROC_IMAGEN)],
    "E4": [],
}


def formato(f, col, maximo=None):
    valor = f[col["clave"]]
    if col.get("texto"):
        return (valor or "—").replace("|", "\\|")
    if f["estado"] == "pendiente":
        return "pendiente"
    if valor is None:
        return "—"
    d = col["decimales"]
    resultado = f"{valor:.{d}f}"
    if maximo is not None and round(valor, d) == maximo:
        resultado = f"**{resultado}**"
    if col.get("ic") and f[col["ic"][0]] is not None:
        resultado += f" ({f[col['ic'][0]]:.{d}f}–{f[col['ic'][1]]:.{d}f})"
    return resultado


def tabla_md(filas, columnas):
    evaluadas = [f for f in filas if f["estado"] == "evaluado"]
    maximos = {}
    for col in columnas:
        valores = [round(f[col["clave"]], col["decimales"]) for f in evaluadas
                   if not col.get("texto") and col["negrita"] and f[col["clave"]] is not None]
        maximos[col["clave"]] = max(valores) if len(valores) > 1 else None
    lineas = ["| " + " | ".join(c["titulo"] for c in columnas) + " |",
              "|" + "|".join("---" for _ in columnas) + "|"]
    lineas += ["| " + " | ".join(formato(f, c, maximos.get(c["clave"])) for c in columnas) + " |" for f in filas]
    return lineas


def etiqueta(f):
    partes = " ".join(p for p in (f["variante_id"], f["modelo"]) if p)
    return f"{partes} ({f['variante']})"


def notas_md(filas):
    """Notas de las filas, agrupadas: una nota que vale para todas las filas se escribe una sola vez."""
    por_nota = {}
    for f in filas:
        for nota in f["nota"]:
            por_nota.setdefault(nota, []).append(etiqueta(f))
    return [f"- {nota}" if len(etiquetas) == len(filas) else f"- {'; '.join(etiquetas)}: {minuscula(nota)}"
            for nota, etiquetas in por_nota.items()]


def minuscula(frase):
    return frase[:1].lower() + frase[1:]


def nombre_corto(f):
    return f"{f['variante_id']} ({f['variante']})" if f["experimento"].startswith("E1") else f"{f['modelo']} ({f['variante']})"


def valor_con_ic(f, clave, ic):
    resultado = f"{f[clave]:.3f}"
    return resultado + (f" ({f[ic[0]]:.3f}–{f[ic[1]]:.3f})" if ic and f[ic[0]] is not None else "")


def resumen_test(experimento, filas):
    test = [f for f in filas if f["experimento"] == experimento and f["split"] == "test"]
    evaluadas = [f for f in test if f["estado"] == "evaluado"]
    pendientes = [f for f in test if f["estado"] == "pendiente"]
    titulo = next(t for e, t, _ in SECCIONES if e == experimento).split(":")[0]
    partes = []
    for clave, nombre, ic in RESUMEN[experimento]:
        if len(evaluadas) < 2:
            break
        mejor, *otras = sorted(evaluadas, key=lambda f: -f[clave])
        comparaciones = []
        for o in otras:
            frase = f"{nombre_corto(o)} {valor_con_ic(o, clave, ic)}"
            if ic:
                superponen = mejor[ic[0]] <= o[ic[1]] and o[ic[0]] <= mejor[ic[1]]
                frase += f": los IC {'se superponen' if superponen else 'no se superponen'}"
            comparaciones.append(frase)
        partes.append(f"{nombre}: mayor en {nombre_corto(mejor)} {valor_con_ic(mejor, clave, ic)}, frente a "
                      + "; frente a ".join(comparaciones) + ".")
    if experimento in DIFERENCIAS:
        partes.append(DIFERENCIAS[experimento])
    por_motivo = {}
    for f in pendientes:
        por_motivo.setdefault(minuscula(" ".join(f["nota"]).rstrip(".")), []).append(etiqueta(f))
    partes += [f"Pendiente, {motivo}: {'; '.join(etiquetas)}." for motivo, etiquetas in por_motivo.items()]
    return f"- **{titulo}.** " + " ".join(partes)


def nota_a1(filas):
    """A1 frente a la mejor variante con tiles en valid (mAP@0.5), para cada dataset de E1."""
    partes = []
    for experimento, dataset in (("E1-H", "Holes"), ("E1-S", "Spots")):
        valid = [f for f in filas if f["experimento"] == experimento and f["split"] == "valid" and f["estado"] == "evaluado"]
        a1 = next(f for f in valid if f["variante_id"] == "A1")
        tiles = max((f for f in valid if f["variante_id"].startswith("A3")), key=lambda f: f["map50"])
        partes.append(f"{dataset}: A1 {a1['map50']:.3f} frente a {tiles['map50']:.3f} de {nombre_corto(tiles)}")
    return ("> **Nota sobre E1.** La A1 (foto completa a 1280) quedó cerca de la A3 en valid pero no fue a test, así "
            "que todavía no se puede afirmar que los tiles le ganen a subir la resolución. mAP@0.5 en valid — "
            + "; ".join(partes) + ".")


def escribir_md(filas):
    lineas = [
        "# Tabla comparativa de modelos — ReWear AI",
        "",
        f"> Generado por `scripts/generate_comparison_table.py` a partir de archivos del repositorio. No editar a "
        f"mano: volver a correr el script. Todas las columnas están en `{SALIDA_CSV.relative_to(ROOT).as_posix()}`.",
        "",
        "## Cómo leer esta tabla",
        "",
        "- Cada fila es un modelo o variante evaluado en un split. **Sólo se comparan filas del mismo experimento y "
        "split** (columna `grupo_comparacion` del CSV).",
        "- mAP (cajas, E1), AUROC por patch (E2) y AUROC por imagen o píxel (E3) responden preguntas distintas: un "
        "número mayor en un experimento no indica un método mejor que otro de otro experimento.",
        "- Supervisado vs. anomalías sólo se compara directamente en **E3 (FabriD)**, con las mismas imágenes de test.",
        "- Valid se usó para elegir variantes, umbrales y épocas; test se evaluó una sola vez con lo elegido en valid.",
        "- IC 95 % por bootstrap: E1 remuestrea capturas (recall y F1), E2 agrupa por imagen madre (AUROC por patch) "
        "y E3 está pareado por `source_id` (AUROC por imagen, sólo test). Si los IC individuales se superponen no se "
        "puede afirmar que un modelo sea mejor; la diferencia pareada, cuando existe, es la evidencia más directa.",
        "- En negrita, el mejor valor de cada columna dentro de su tabla. Los tiempos dependen del hardware y de la "
        "unidad de cada experimento: no se comparan entre experimentos.",
        "- `pendiente`: resultado que todavía no existe; no se completa con estimaciones.",
        "",
        "## Resumen de test",
        "",
    ]
    lineas += [resumen_test(e, filas) for e, _, _ in SECCIONES]
    lineas += ["", nota_a1(filas)]
    for experimento, titulo, columnas in SECCIONES:
        lineas += ["", f"## {titulo}", "", INTRO_SECCION[experimento]]
        for split in ("valid", "test"):
            grupo = [f for f in filas if f["experimento"] == experimento and f["split"] == split]
            if not grupo:
                continue
            detalle = DETALLES.get(f"{experimento}/{split}")
            lineas += ["", f"### {split.capitalize()}" + (f" ({detalle})" if detalle else ""), ""]
            cols = columnas
            if split == "valid":
                cols = columnas[:3] + [texto("¿Fue a test?", "fue_a_test")] + columnas[3:]
            lineas += tabla_md(grupo, cols)
            notas = notas_md(grupo)
            if notas:
                lineas += [""] + notas
    SALIDA_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main():
    filas = cargar_e1() + cargar_e2() + cargar_e3() + [fila({**p, "estado": "pendiente"}) for p in PENDIENTES]
    claves = [(f["grupo_comparacion"], f["modelo"], f["variante_id"], f["variante"]) for f in filas]
    control(len(claves) == len(set(claves)), f"{len(filas)} filas sin repetir modelo y variante dentro de un grupo")
    filas = ordenar(filas)
    escribir_csv(filas)
    escribir_md(filas)
    print("Controles:")
    for mensaje in CONTROLES:
        print(f"  [ok] {mensaje}")
    pendientes = sum(f["estado"] == "pendiente" for f in filas)
    print(f"\nEscritos {SALIDA_CSV.relative_to(ROOT)} y {SALIDA_MD.relative_to(ROOT)}: "
          f"{len(filas)} filas ({len(filas) - pendientes} evaluadas, {pendientes} pendientes).")


if __name__ == "__main__":
    main()
