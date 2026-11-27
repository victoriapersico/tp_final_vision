"""Prepara FabriD v3 para entrenar y evaluar con Ultralytics, sin copiar ni modificar el export original.

Usa el manifiesto agrupado por captura de FAB-4/FAB-6 (reports/fabrid/fab4_split_agrupado.csv): toma su
`split_propuesto` y deja afuera los 19 casos ambiguos. Lo comparten el E3 (segmentación de las 4 clases, con
polígonos) y el E4 (transferencia a CISUTAC con hole y stain, con cajas).

Qué imágenes entran:
- las normales concordantes, como fondo (label vacío);
- las defectuosas con todas sus instancias en las clases elegidas;
- las que mezclan clases elegidas con otras, sólo con --mixtas: las instancias de las otras clases quedarían
  como fondo, así que por defecto se excluyen;
- nunca las que sólo tienen otras clases ni los ambiguos.

Escribe en data/derived/fabrid_<clases>_<formato>/ (no versionado):
- images/<split>/: enlaces simbólicos al export (copias si el sistema no admite enlaces).
- labels/<split>/: las clases elegidas, renumeradas desde 0 en el orden de --clases. Con --formato cajas, cada
  polígono se reemplaza por su caja envolvente.
- data.yaml para Ultralytics, con los nombres originales de las clases.
- coco_<split>_<clase>.json: cajas de una clase en px del export, para src/detection_eval.py. Incluye todas las
  imágenes del split, así una predicción de esa clase sobre otra imagen cuenta como falso positivo.
- resumen.json: imágenes e instancias por split y motivos de exclusión.

Ejemplos:
  python scripts/prepare_fabrid_yolo.py                                      # E3: 4 clases, polígonos
  python scripts/prepare_fabrid_yolo.py --clases hole stain --formato cajas  # E4
"""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eda_utils import fabrid_export_records, link_or_copy

DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
EXPORT = DATA / "fabrid_v3" / "annotations_export"
DEFAULT_MANIFEST = ROOT / "reports" / "fabrid" / "fab4_split_agrupado.csv"

parser = argparse.ArgumentParser(description="Prepara FabriD v3 con el split agrupado por captura.")
parser.add_argument("--clases", nargs="+", default=["needle", "hole", "missing", "stain"],
                    help="primera palabra de cada clase, en minúscula: needle, hole, missing, stain")
parser.add_argument("--formato", choices=("poligonos", "cajas"), default="poligonos")
parser.add_argument("--mixtas", action="store_true", help="incluir imágenes que además tienen otras clases")
parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
args = parser.parse_args()

if not (EXPORT / "data.yaml").exists():
    sys.exit(f"Falta {EXPORT}: descargar FabriD v3 según docs/DATASETS.md")
images, instances, class_names = fabrid_export_records(EXPORT)
by_key = {name.split()[0].lower(): k for k, name in enumerate(class_names)}
if unknown := [c for c in args.clases if c not in by_key]:
    sys.exit(f"Clases desconocidas {unknown}; opciones: {sorted(by_key)}")
chosen = [by_key[c] for c in args.clases]
new_id = {old: new for new, old in enumerate(chosen)}

with args.manifest.open(newline="") as handle:
    manifest = {row["ruta_relativa"]: row for row in csv.DictReader(handle)}
exported = {f"{im['split']}/images/{im['archivo']}": im for im in images}
if set(exported) != set(manifest):
    sys.exit(f"El manifiesto no coincide con el export ({len(manifest)} filas, {len(exported)} imágenes): "
             "¿es FabriD v3?")
rows_by_image = defaultdict(list)
for inst in instances:
    rows_by_image[f"{inst['split']}/images/{inst['archivo']}"].append(inst)


def clamp(value):
    return min(max(value, 0.0), 1.0)


selected, excluded = defaultdict(list), Counter()
for rel in sorted(manifest):
    row, insts = manifest[rel], rows_by_image[rel]
    if len(insts) != int(row["instancias"]):
        sys.exit(f"{rel}: {len(insts)} polígonos en el label y {row['instancias']} en el manifiesto")
    if any(len(i["points"]) < 3 for i in insts):
        sys.exit(f"{rel}: hay una fila que no es un polígono")
    classes = {i["class_id"] for i in insts}
    if row["usar_comparacion_comun"] != "True":
        excluded["ambigua"] += 1
    elif not insts:
        selected[row["split_propuesto"]].append((rel, []))
    elif classes <= set(chosen) or (classes & set(chosen) and args.mixtas):
        selected[row["split_propuesto"]].append((rel, [i for i in insts if i["class_id"] in new_id]))
    else:
        excluded["mixta" if classes & set(chosen) else "sólo otras clases"] += 1

tag = "todas" if set(args.clases) == set(by_key) else "_".join(args.clases)
out = DATA / "derived" / f"fabrid_{tag}{'_mixtas' if args.mixtas else ''}_{args.formato}"
for stale in ("images", "labels"):  # se regeneran enteras para que no queden restos de otra configuración
    shutil.rmtree(out / stale, ignore_errors=True)
summary = {"clases": [class_names[k] for k in chosen], "formato": args.formato, "mixtas": args.mixtas,
           "excluidas": dict(excluded), "splits": {}}
for split in ("train", "valid", "test"):
    (out / "images" / split).mkdir(parents=True)
    (out / "labels" / split).mkdir(parents=True)
    coco = {k: {"images": [], "annotations": [], "categories": [{"id": 1, "name": class_names[k]}]} for k in chosen}
    for image_id, (rel, insts) in enumerate(selected[split], start=1):
        im = exported[rel]
        link_or_copy(EXPORT / rel, out / "images" / split / im["archivo"])
        rows = []
        for inst in insts:
            xs, ys = [clamp(x) for x, _ in inst["points"]], [clamp(y) for _, y in inst["points"]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if args.formato == "cajas":
                values = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
            else:
                values = [v for point in zip(xs, ys) for v in point]
            rows.append(f"{new_id[inst['class_id']]} " + " ".join(f"{v:.6f}" for v in values) + "\n")
            w, h = (x2 - x1) * im["img_ancho"], (y2 - y1) * im["img_alto"]
            coco[inst["class_id"]]["annotations"].append({
                "id": len(coco[inst["class_id"]]["annotations"]) + 1, "image_id": image_id, "category_id": 1,
                "bbox": [round(x1 * im["img_ancho"], 2), round(y1 * im["img_alto"], 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2), "iscrowd": 0})
        (out / "labels" / split / f"{Path(im['archivo']).stem}.txt").write_text("".join(rows))
        for k in chosen:
            coco[k]["images"].append({"id": image_id, "file_name": rel, "width": im["img_ancho"],
                                      "height": im["img_alto"], "source_id": im["source_id"]})
    for key, k in zip(args.clases, chosen):
        (out / f"coco_{split}_{key}.json").write_text(json.dumps(coco[k], ensure_ascii=False))
    summary["splits"][split] = {"imagenes": len(selected[split]),
                                "normales": sum(not insts for _, insts in selected[split]),
                                **{f"instancias {class_names[k]}": len(coco[k]["annotations"]) for k in chosen}}
    print(f"{split}: {summary['splits'][split]}")
(out / "data.yaml").write_text(
    f"# Generado por scripts/prepare_fabrid_yolo.py a partir de {args.manifest.name}; no editar a mano.\n"
    f"path: {json.dumps(str(out))}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n"
    + "".join(f"  {new_id[k]}: {json.dumps(class_names[k])}\n" for k in chosen))
(out / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Excluidas: {dict(excluded)}")
print(f"Escrito en {out}")
