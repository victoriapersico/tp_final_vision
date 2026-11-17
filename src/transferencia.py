"""Funciones del E4: transferencia de un detector entrenado en FabriD a CISUTAC, sin ajuste.

Las usa `notebooks/04_transferencia_fabrid_cisutac.ipynb`. La inferencia y la evaluación son las de
`cisutac_baseline.py` (mismas métricas que el E1); acá sólo está lo propio del cambio de dominio:
- qué clase de FabriD corresponde a cada dataset de CISUTAC (hole → Holes, stain → Spots);
- dónde están los datos de FabriD y de FabricSpotDefect (anexo exploratorio) que escriben
  scripts/prepare_fabrid_yolo.py y scripts/prepare_fsd_yolo.py;
- a qué tamaño llegan los defectos al modelo en cada dominio y con cada forma de inferir.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

from cisutac_baseline import DATA, ROOT

CLASE_FABRID = {"holes": "hole", "spots": "stain"}


def fabrid_export_dir():
    return DATA / "fabrid_v3" / "annotations_export"


def fabrid_derived_dir(variant="hole_stain_cajas"):
    return DATA / "derived" / f"fabrid_{variant}"


def fsd_original_dir():
    """Originales de FabricSpotDefect (la carpeta se llama `Orginal` en el dataset)."""
    return next((DATA / "fabric_spot_defect").glob("*/FabricSpotDefect/Orginal"))


def fsd_derived_dir():
    return DATA / "derived" / "fsd_spot"


def load_ground_truth(folder, split, key):
    """Cajas COCO de una clase, escritas por prepare_fabrid_yolo.py o prepare_fsd_yolo.py."""
    path = Path(folder) / f"coco_{split}_{key}.json"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}: correr antes el script de preparación de ese dataset")
    return json.loads(path.read_text())


def load_fabrid_ground_truth(split, key, variant="hole_stain_cajas"):
    return load_ground_truth(fabrid_derived_dir(variant), split, key)


def balanced_ground_truth_subset(gt, n=3):
    """Subconjunto chico con positivos y una normal para verificar las métricas en modo prueba."""
    if n is None or n >= len(gt["images"]):
        return dict(gt)
    if n < 1:
        raise ValueError("n debe ser al menos 1")

    annotated = {annotation["image_id"] for annotation in gt["annotations"]}
    positives = [image for image in gt["images"] if image["id"] in annotated]
    normals = [image for image in gt["images"] if image["id"] not in annotated]
    selected = positives[:n - 1] + normals[:1] if positives and normals and n > 1 else (positives or normals)[:n]
    selected_ids = {image["id"] for image in selected}
    selected += [image for image in gt["images"] if image["id"] not in selected_ids][:n - len(selected)]
    selected_ids = {image["id"] for image in selected}
    return {
        **gt,
        "images": selected,
        "annotations": [annotation for annotation in gt["annotations"] if annotation["image_id"] in selected_ids],
    }


def class_id(weights, key):
    """Id de la clase del modelo cuyo nombre empieza con `key` (p. ej. "hole" → "hole Defect")."""
    from ultralytics import YOLO

    names = YOLO(str(weights)).names
    matches = [k for k, name in names.items() if name.split()[0].lower() == key]
    if len(matches) != 1:
        raise ValueError(f"No hay una única clase «{key}» en el modelo: {names}")
    return matches[0]


def scale_table(tiles=(640, 320), model_side=640):
    """Lado equivalente mediano (px) con el que cada defecto llega al modelo.

    FabriD: cajas del export (≤ 640 px), que entra entero al modelo (fab3_tamanos_clase.csv).
    CISUTAC: cajas de la original (instancias_original.csv) reducidas como en cada forma de inferir:
    foto completa con lado mayor `model_side`, o tiles de cada lado de `tiles` que entran a `model_side`.
    """
    import pandas as pd

    with (ROOT / "reports" / "fabrid" / "fab3_tamanos_clase.csv").open(newline="") as handle:
        fabrid = {row["clase"]: float(row["lado_bbox_p50_px"]) for row in csv.DictReader(handle)}
    with (ROOT / "reports" / "cisutac" / "instancias_original.csv").open(newline="") as handle:
        cisutac = [row for row in csv.DictReader(handle) if row["version"] == "original"]
    rows = {}
    for dataset, fabrid_class in (("Holes", "hole Defect"), ("Spots", "stain Defect")):
        subset = [row for row in cisutac if row["dataset"] == dataset]
        sides = [(float(r["ancho_px"]) * float(r["alto_px"])) ** 0.5 for r in subset]
        full = [side * model_side / max(int(r["img_ancho"]), int(r["img_alto"])) for side, r in zip(sides, subset)]
        rows[f"{dataset} ↔ {fabrid_class}"] = {
            "FabriD (export)": fabrid[fabrid_class],
            f"CISUTAC, foto completa {model_side}": median(full),
            **{f"CISUTAC, tile {tile} → {model_side}": median(sides) * model_side / tile for tile in tiles},
        }
    return pd.DataFrame(rows).T.round(1)
