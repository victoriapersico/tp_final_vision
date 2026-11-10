"""Prepara CISUTAC para entrenar y evaluar un detector, sin copiar ni modificar los originales.

Para Holes y Spots escribe en data/derived/<dataset>/ (no versionado; cada integrante lo corre):
- images/<split>/: enlaces simbólicos a las imágenes originales según el manifiesto de splits
  (copias si el sistema no admite enlaces). Las copias 640×640 quedan afuera.
- labels/<split>/: labels YOLO derivados con una caja por defecto. Los polígonos se reemplazan por su
  caja envolvente: Ultralytics descarta las imágenes cuyos labels mezclan cajas y polígonos.
- data.yaml para Ultralytics.
- coco_<split>.json: las mismas cajas en formato COCO, en px de la imagen original, para pycocotools
  (AP_small). El campo `area` es el área de la caja (misma definición que el EDA).
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eda_utils import cisutac_coco_ground_truth, cisutac_records, yolo_class_names

DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
DEFAULT_MANIFEST = ROOT / "reports" / "cisutac" / "cis5_split_propuesto.csv"


def link_or_copy(source, target):
    try:
        target.symlink_to(source)
    except OSError:  # p. ej. Windows sin permisos para enlaces simbólicos
        shutil.copy2(source, target)


parser = argparse.ArgumentParser(description="Prepara imágenes, labels, data.yaml y COCO de CISUTAC por split.")
parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="CSV con columnas dataset, captura y split_propuesto (por defecto, el de CIS-5)")
args = parser.parse_args()
with args.manifest.open(newline="") as handle:
    manifest = list(csv.DictReader(handle))

for name, folder in (("Holes", "cisutac_holes"), ("Spots", "cisutac_spots")):
    root = DATA / folder
    classes = yolo_class_names(root / "data.yaml")
    records = cisutac_records(root, classes)
    images = {im["captura"]: im for im in records["images"] if im["version"] == "original"}
    split_of = {row["captura"]: row["split_propuesto"] for row in manifest if row["dataset"] == name}
    if set(images) != set(split_of):
        sys.exit(f"{name}: el manifiesto no cubre exactamente las {len(images)} capturas originales")
    boxes = {capture: [] for capture in images}
    for inst in records["instances"]:
        if inst["version"] == "original":
            boxes[inst["captura"]].append(inst)

    out = DATA / "derived" / folder
    for stale in ("images", "labels"):  # se regeneran enteras para que no queden restos de otro split
        shutil.rmtree(out / stale, ignore_errors=True)
    for split in ("train", "valid", "test"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        captures = sorted(c for c, s in split_of.items() if s == split)
        for capture in captures:
            im = images[capture]
            link_or_copy(root / im["split"] / "images" / im["archivo"], out / "images" / split / im["archivo"])
            rows = [f"0 {i['x_centro']:.6f} {i['y_centro']:.6f} {i['ancho_rel']:.6f} {i['alto_rel']:.6f}\n"
                    for i in boxes[capture]]
            (out / "labels" / split / f"{Path(im['archivo']).stem}.txt").write_text("".join(rows))
        coco = cisutac_coco_ground_truth(records, captures, classes[0])
        (out / f"coco_{split}.json").write_text(json.dumps(coco, ensure_ascii=False))
        print(f"{name} {split}: {len(captures)} imágenes, {len(coco['annotations'])} cajas")
    (out / "data.yaml").write_text(
        f"# Generado por scripts/prepare_cisutac_yolo.py a partir de {args.manifest.name}; no editar a mano.\n"
        f"path: {json.dumps(str(out))}\ntrain: images/train\nval: images/valid\ntest: images/test\n"
        f"names:\n  0: {classes[0]}\n")
    print(f"{name}: escrito en {out}")
