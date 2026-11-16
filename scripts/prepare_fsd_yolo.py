"""Prepara los originales de FabricSpotDefect (clase spot) para Ultralytics, sin copiar ni modificar los datos.

Es la fuente del anexo exploratorio del E4 (FabricSpotDefect → CISUTAC Spots). Corrige dos problemas del export:

- Orientación: en 228 originales el archivo está acostado pero las cajas COCO corresponden a la foto girada
  90° en sentido horario (ancho y alto invertidos, sin etiqueta EXIF). Se verificó a ojo. En vez de rotar las
  fotos, las cajas se pasan a la orientación del archivo, que es la que leen Ultralytics y PIL.
- Duplicados entre splits: 7 fotos numeradas de valid son copias (algunas giradas) de fotos de train del lote de
  WhatsApp del 08/03/2024, y otras dos son la misma pieza con otro encuadre. Se encontraron comparando miniaturas
  con las cuatro rotaciones y se revisaron a ojo (reports/transferencia/e4_transferencia.md). Como en FAB-4, la
  evaluación tiene prioridad: esas fotos de train pasan a valid.

Además se excluyen las 36 fotos de train sin cajas (no están verificadas como sanas) y no se usan los
aumentados: sólo originales.

Escribe en data/derived/fsd_spot/ (no versionado): images/<split>/ (enlaces), labels/<split>/ (una caja por
mancha), data.yaml, coco_<split>_spot.json (px de la foto tal como está guardada) y resumen.json.

Uso: python scripts/prepare_fsd_yolo.py
"""

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eda_utils import link_or_copy, roboflow_source_id

DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
OUT = DATA / "derived" / "fsd_spot"

# Fotos de train relacionadas con una foto de valid (copia, copia girada o misma pieza): pasan a valid.
MOVE_TO_VALID = {
    "IMG-20240308-WA0065",  # = valid 60
    "IMG-20240308-WA0069",  # = valid 47
    "IMG-20240308-WA0078",  # = valid 49 (y probablemente la misma tela que valid 48)
    "IMG-20240308-WA0095",  # = valid 2
    "IMG-20240308-WA0102",  # = valid 9
    "IMG-20240308-WA0070",  # = valid 46 girada; misma pieza que valid 44 (escritura «SPD-421»)
    "IMG-20240308-WA0117",  # = valid 50 girada
    "IMG-20240308-WA0061",  # misma pieza que valid 73, con otro encuadre
}

candidates = sorted((DATA / "fabric_spot_defect").glob("*/FabricSpotDefect/Orginal"))
if not candidates:
    sys.exit(f"Falta FabricSpotDefect en {DATA / 'fabric_spot_defect'} (ver docs/DATASETS.md)")
ORIGINAL = candidates[0]


def clamp(value, top):
    return min(max(value, 0.0), top)


records, counts = [], Counter()
for split in ("train", "valid", "test"):
    coco = json.loads((ORIGINAL / split / "_annotations.coco.json").read_text())
    spot = next(c["id"] for c in coco["categories"] if c["name"] == "spot")
    boxes = {}
    for a in coco["annotations"]:
        if a["category_id"] == spot:
            boxes.setdefault(a["image_id"], []).append(a["bbox"])
    for im in coco["images"]:
        with Image.open(ORIGINAL / split / im["file_name"]) as raw:
            width, height = raw.size
        if (width, height) == (im["width"], im["height"]):
            converted = boxes.get(im["id"], [])
        elif (width, height) == (im["height"], im["width"]):
            # COCO = archivo girado 90° horario: (x, y) de COCO -> (y, ancho_COCO - x) en el archivo.
            converted = [[y, im["width"] - x - w, h, w] for x, y, w, h in boxes.get(im["id"], [])]
            counts["fotos con cajas giradas"] += 1
        else:
            sys.exit(f"{split}/{im['file_name']}: tamaño {width}×{height} incompatible con COCO")
        source = roboflow_source_id(im["file_name"])
        if split == "train" and not converted:
            counts["train sin cajas (excluidas)"] += 1
            continue
        destination = "valid" if split == "train" and source in MOVE_TO_VALID else split
        counts["movidas de train a valid"] += destination != split
        clipped = []
        for x, y, w, h in converted:
            x1, y1 = clamp(x, width), clamp(y, height)
            x2, y2 = clamp(x + w, width), clamp(y + h, height)
            if x2 > x1 and y2 > y1:
                clipped.append((x1, y1, x2, y2))
        records.append({"split": destination, "rel": f"{split}/{im['file_name']}", "archivo": im["file_name"],
                        "source_id": source, "width": width, "height": height, "boxes": clipped})
if missing := MOVE_TO_VALID - {r["source_id"] for r in records}:
    sys.exit(f"No se encontraron en train: {sorted(missing)}")

for stale in ("images", "labels"):  # se regeneran enteras para que no queden restos de otra versión
    shutil.rmtree(OUT / stale, ignore_errors=True)
summary = {"correcciones": dict(counts), "splits": {}}
for split in ("train", "valid", "test"):
    (OUT / "images" / split).mkdir(parents=True)
    (OUT / "labels" / split).mkdir(parents=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "spot"}]}
    chosen = [r for r in records if r["split"] == split]
    for image_id, r in enumerate(chosen, start=1):
        link_or_copy(ORIGINAL / r["rel"], OUT / "images" / split / r["archivo"])
        W, H = r["width"], r["height"]
        (OUT / "labels" / split / f"{Path(r['archivo']).stem}.txt").write_text("".join(
            f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
            for x1, y1, x2, y2 in r["boxes"]))
        coco["images"].append({"id": image_id, "file_name": r["rel"], "width": W, "height": H,
                               "source_id": r["source_id"]})
        for x1, y1, x2, y2 in r["boxes"]:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id, "category_id": 1,
                                        "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                        "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0})
    (OUT / f"coco_{split}_spot.json").write_text(json.dumps(coco, ensure_ascii=False))
    summary["splits"][split] = {"imagenes": len(chosen), "manchas": len(coco["annotations"])}
    print(f"{split}: {summary['splits'][split]}")
(OUT / "data.yaml").write_text(
    "# Generado por scripts/prepare_fsd_yolo.py; no editar a mano.\n"
    f"path: {json.dumps(str(OUT))}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n  0: spot\n")
(OUT / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Correcciones: {dict(counts)}")
print(f"Escrito en {OUT}")
