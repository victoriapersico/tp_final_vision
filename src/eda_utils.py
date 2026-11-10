"""Utilidades reproducibles para auditar los datasets de ReWear AI.

Este módulo sólo lee imágenes y anotaciones; nunca modifica los originales.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, quantiles

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def image_inventory(root: Path):
    """Resume cantidad, formatos, modos y resoluciones sin cargar píxeles."""
    files = image_files(root)
    resolutions, modes, formats, unreadable = Counter(), Counter(), Counter(), []
    for path in files:
        try:
            with Image.open(path) as im:
                resolutions[f"{im.width}x{im.height}"] += 1
                modes[im.mode] += 1
                formats[im.format or path.suffix.lower()] += 1
        except Exception as exc:  # deja trazabilidad de archivos problemáticos
            unreadable.append({"path": str(path), "error": str(exc)})
    return {
        "images": len(files),
        "formats": dict(formats),
        "modes": dict(modes),
        "resolutions": dict(resolutions),
        "unreadable": unreadable,
    }


def sha256(path: Path, block_size: int = 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_duplicate_groups(paths):
    """Agrupa duplicados exactos por tamaño y SHA-256."""
    sizes = defaultdict(list)
    for path in paths:
        sizes[path.stat().st_size].append(path)
    hashes = defaultdict(list)
    for candidates in sizes.values():
        if len(candidates) > 1:
            for path in candidates:
                hashes[sha256(path)].append(str(path))
    return [items for items in hashes.values() if len(items) > 1]


def polygon_area(points):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2


def analyze_yolo_segmentation(root: Path, class_names):
    """Analiza un export YOLO-seg organizado por train/valid/test."""
    result = {"classes": list(class_names), "splits": {}, "instances": Counter(), "areas": []}
    stem_splits = defaultdict(set)
    for split in ("train", "valid", "test"):
        labels = sorted((root / split / "labels").glob("*.txt"))
        images = image_files(root / split / "images")
        empty, malformed, instances = 0, 0, 0
        for image in images:
            # Roboflow añade .rf.<hash>; la parte anterior aproxima el original.
            stem_splits[image.stem.split(".rf.")[0]].add(split)
        for label in labels:
            rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
            if not rows:
                empty += 1
            for row in rows:
                try:
                    cls = int(float(row[0])); coords = list(map(float, row[1:]))
                    points = list(zip(coords[::2], coords[1::2]))
                    area = polygon_area(points)
                    result["instances"][class_names[cls]] += 1
                    result["areas"].append(area)
                    instances += 1
                except Exception:
                    malformed += 1
        result["splits"][split] = {"images": len(images), "labels": len(labels), "empty_labels": empty,
                                           "instances": instances, "malformed_rows": malformed}
    areas = sorted(result.pop("areas"))
    result["instances"] = dict(result["instances"])
    result["area_fraction"] = {
        "count": len(areas), "median": median(areas) if areas else None,
        "p10": areas[int(.1 * (len(areas)-1))] if areas else None,
        "p90": areas[int(.9 * (len(areas)-1))] if areas else None,
        "under_1pct": sum(a < .01 for a in areas),
    }
    result["possible_source_leakage"] = {k: sorted(v) for k, v in stem_splits.items() if len(v) > 1}
    return result


def analyze_raw_fabrid(root: Path):
    """Resume metadatos high-res y parches de RAW-FABRID."""
    with (root / "RAW_FABRID_HighRes_Metadata.csv").open(newline="") as f:
        high = list(csv.DictReader(f))
    with (root / "RAW_FABRID_Patches_Metadata.csv").open(newline="") as f:
        patches = list(csv.DictReader(f))
    with (root / "RAW_FABRID_HighRes_COCO.json").open() as f:
        coco = json.load(f)
    area_fracs = sorted(float(a["area"]) / (1792 * 1024) for a in coco["annotations"])
    parents_by_type = defaultdict(set)
    counts = Counter()
    for row in patches:
        counts[row["type"]] += 1
        parents_by_type[row["type"]].add(row["parent_raw"])
    overlap = {}
    keys = sorted(parents_by_type)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = parents_by_type[a] & parents_by_type[b]
            if shared:
                overlap[f"{a}__{b}"] = len(shared)
    return {
        "high_res": {"images": len(high), "defective": sum(r["has_defect"].lower() == "true" for r in high),
                     "good": sum(r["has_defect"].lower() == "false" for r in high),
                     "rolls": dict(Counter(r["roll_id"] for r in high))},
        "annotations": {"instances": len(coco["annotations"]), "with_bbox": sum(bool(a.get("bbox")) for a in coco["annotations"]),
                        "with_polygon": sum(bool(a.get("segmentation")) for a in coco["annotations"]),
                        "median_area_fraction": median(area_fracs), "under_1pct": sum(a < .01 for a in area_fracs)},
        "patches": {"counts": dict(counts), "unique_parents": {k: len(v) for k, v in parents_by_type.items()},
                    "parent_overlap_between_splits": overlap},
    }


# Nombre CISUTAC: <fecha>_<hora>_<id prenda: 8 hex>_<vista>_nobg_png.rf.<hash Roboflow>
CISUTAC_GARMENT = re.compile(r"_([0-9a-f]{8})_(front|back)")


def yolo_class_names(data_yaml: Path):
    """Lee `names: [...]` de un data.yaml de Roboflow sin depender de PyYAML."""
    for line in data_yaml.read_text().splitlines():
        if line.startswith("names:"):
            return ast.literal_eval(line.split(":", 1)[1].strip())
    return []


def yolo_instance(row, class_names, width, height):
    """Convierte una fila YOLO (caja o polígono normalizados) a medidas en píxeles de su imagen.

    Los polígonos se resumen con su caja envolvente; su área poligonal queda aparte
    para no mezclar áreas de cajas y de máscaras.
    """
    cls = class_names[int(float(row[0]))]
    coords = list(map(float, row[1:]))
    if len(coords) == 4:
        kind, (xc, yc, w, h), poly_area = "caja", coords, None
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs, ys = coords[::2], coords[1::2]
        kind = "poligono"
        xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        poly_area = round(polygon_area(list(zip(xs, ys))) * width * height, 1)
    else:
        raise ValueError(f"fila YOLO con {len(coords)} coordenadas")
    return {
        "tipo": kind, "clase": cls,
        "x_centro": round(xc, 6), "y_centro": round(yc, 6), "ancho_rel": round(w, 6), "alto_rel": round(h, 6),
        # Sin redondear: los umbrales de tamaño (p. ej. 32 px) se evalúan con precisión completa.
        "ancho_px": w * width, "alto_px": h * height,
        "area_caja_px": w * h * width * height, "area_caja_rel": w * h,
        "area_poligono_px": poly_area,
        "coords_validas": all(0 <= c <= 1 for c in coords) and w > 0 and h > 0,
    }


def cisutac_records(root: Path, class_names, splits=("train", "valid")):
    """Lee un export Roboflow de CISUTAC: una fila por imagen y otra por instancia anotada.

    Roboflow exporta cada captura (nombre antes de `.rf.`) en versión `original` y, casi
    siempre, en una copia `640` (640×640). El código de 8 hex del nombre identifica la prenda.
    """
    images, instances, orphan_labels = [], [], []
    for split in splits:
        image_paths = image_files(root / split / "images")
        stems = {p.stem for p in image_paths}
        orphan_labels += [{"split": split, "archivo": p.name}
                          for p in sorted((root / split / "labels").glob("*.txt")) if p.stem not in stems]
        for path in image_paths:
            capture = path.stem.split(".rf.")[0]
            match = CISUTAC_GARMENT.search(capture)
            with Image.open(path) as im:
                width, height = im.size
                orientation = im.getexif().get(0x0112, 1)
            label = root / split / "labels" / f"{path.stem}.txt"
            rows = [line.split() for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
            image = {"split": split, "archivo": path.name, "captura": capture, "prenda": match.group(1) if match else None,
                     "vista": match.group(2) if match else None,
                     "version": "640" if (width, height) == (640, 640) else "original",
                     "img_ancho": width, "img_alto": height, "exif_orientacion": orientation,
                     "tiene_label": label.exists(), "instancias": 0, "filas_malformadas": 0}
            for row in rows:
                try:
                    instance = yolo_instance(row, class_names, width, height)
                except (ValueError, IndexError):
                    image["filas_malformadas"] += 1
                    continue
                image["instancias"] += 1
                keys = ("split", "captura", "prenda", "version", "img_ancho", "img_alto")
                instances.append({**{k: image[k] for k in keys}, **instance})
            images.append(image)
    return {"images": images, "instances": instances, "orphan_labels": orphan_labels}


def cisutac_inventory(records):
    """Tabla CIS-1 por split y total, separando unidades: archivo, captura, prenda e instancia."""
    rows = []
    for split in [*dict.fromkeys(im["split"] for im in records["images"]), "total"]:
        ims = [im for im in records["images"] if split in ("total", im["split"])]
        ins = [i for i in records["instances"] if split in ("total", i["split"])]
        originals = [im for im in ims if im["version"] == "original"]
        copies = defaultdict(Counter)
        for im in ims:
            copies[im["captura"]][im["version"]] += 1
        resolutions = Counter(f"{im['img_ancho']}x{im['img_alto']}" for im in originals)
        rows.append({
            "split": split,
            "archivos": len(ims),
            "archivos_original": len(originals),
            "archivos_640": len(ims) - len(originals),
            "capturas": len(copies),
            "capturas_sin_640": sum(c["640"] == 0 for c in copies.values()),
            "capturas_irregulares": sum(c["original"] != 1 or c["640"] > 1 for c in copies.values()),
            "prendas": len({im["prenda"] for im in ims if im["prenda"]}),
            "capturas_sin_id_prenda": len({im["captura"] for im in ims if not im["prenda"]}),
            "instancias_todas_las_copias": len(ins),
            "instancias_original": sum(i["version"] == "original" for i in ins),
            "poligonos_original": sum(i["version"] == "original" and i["tipo"] == "poligono" for i in ins),
            "imagenes_sin_instancias": sum(im["instancias"] == 0 for im in ims),
            "labels_faltantes": sum(not im["tiene_label"] for im in ims),
            "labels_sin_imagen": sum(split in ("total", o["split"]) for o in records["orphan_labels"]),
            "filas_malformadas": sum(im["filas_malformadas"] for im in ims),
            "instancias_coords_invalidas": sum(not i["coords_validas"] for i in ins),
            "imagenes_exif_rotadas": sum(im["exif_orientacion"] != 1 for im in ims),
            "resoluciones_original": "; ".join(f"{k}: {n}" for k, n in resolutions.most_common()),
        })
    return rows


def dhash(path: Path, size: int = 8):
    """Hash perceptual por diferencias: signo del gradiente horizontal de una miniatura en grises.

    Fotos casi idénticas (recompresión, recortes o retoques leves) quedan a pocos bits de distancia.
    """
    with Image.open(path) as im:
        im.draft("L", (size * 8, size * 8))  # decodifica el JPEG a escala reducida
        small = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = small.tobytes()
    bits = (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] for r in range(size) for c in range(size))
    return sum(1 << i for i, bit in enumerate(bits) if bit)


def near_duplicate_pairs(hashes, max_distance):
    """Pares (distancia, a, b) cuyos hashes difieren en a lo sumo `max_distance` bits."""
    keys = list(hashes)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            distance = (hashes[a] ^ hashes[b]).bit_count()
            if distance <= max_distance:
                pairs.append((distance, a, b))
    return sorted(pairs)


def deciles(values):
    """p10, p50 y p90 con interpolación lineal (igual que numpy.percentile por defecto)."""
    q = quantiles(values, n=10, method="inclusive")
    return q[0], q[4], q[8]


def equivalent_side(instance):
    """Lado del cuadrado con la misma área que la caja: √(ancho·alto) en px."""
    return (instance["ancho_px"] * instance["alto_px"]) ** 0.5


def cisutac_size_quantiles(records):
    """Tabla CIS-3 de una medida por fila (n, p10, p50, p90, mín., máx.), sólo versiones originales.

    Las áreas de polígonos se reportan aparte de las de cajas.
    """
    ins = [i for i in records["instances"] if i["version"] == "original"]
    polygons = [i for i in ins if i["tipo"] == "poligono"]
    measures = {
        ("defectos_por_imagen", "instancias"):
            [im["instancias"] for im in records["images"] if im["version"] == "original"],
        ("ancho_caja", "px"): [i["ancho_px"] for i in ins],
        ("alto_caja", "px"): [i["alto_px"] for i in ins],
        ("lado_equivalente", "px"): [equivalent_side(i) for i in ins],
        ("area_caja", "px²"): [i["area_caja_px"] for i in ins],
        ("area_caja_rel", "% de la imagen"): [100 * i["area_caja_rel"] for i in ins],
        ("relacion_aspecto", "ancho/alto"): [i["ancho_px"] / i["alto_px"] for i in ins],
        ("area_poligono", "px²"): [i["area_poligono_px"] for i in polygons],
        ("area_poligono_sobre_caja", "fracción"): [i["area_poligono_px"] / i["area_caja_px"] for i in polygons],
    }
    rows = []
    for (measure, unit), values in measures.items():
        p10, p50, p90 = deciles(values)
        rows.append({"medida": measure, "unidad": unit, "n": len(values),
                     **{k: float(f"{v:.4g}") for k, v in
                        (("p10", p10), ("p50", p50), ("p90", p90), ("min", min(values)), ("max", max(values)))}})
    return rows


def cisutac_size_thresholds(records):
    """Porcentaje de defectos bajo umbrales de tamaño, medidos en la imagen original (CIS-3)."""
    ins = [i for i in records["instances"] if i["version"] == "original"]
    checks = {
        "lado equivalente < 8 px": lambda i: equivalent_side(i) < 8,
        "lado equivalente < 16 px": lambda i: equivalent_side(i) < 16,
        "COCO small: área < 32² px": lambda i: equivalent_side(i) < 32,
        "COCO medium: 32² ≤ área < 96² px": lambda i: 32 <= equivalent_side(i) < 96,
        "COCO large: área ≥ 96² px": lambda i: equivalent_side(i) >= 96,
        "lado menor (mín. de ancho y alto) < 4 px": lambda i: min(i["ancho_px"], i["alto_px"]) < 4,
        "lado menor (mín. de ancho y alto) < 8 px": lambda i: min(i["ancho_px"], i["alto_px"]) < 8,
        "área relativa < 0,01 %": lambda i: i["area_caja_rel"] < 1e-4,
        "área relativa < 0,1 %": lambda i: i["area_caja_rel"] < 1e-3,
        "área relativa < 1 %": lambda i: i["area_caja_rel"] < 1e-2,
    }
    return [{"umbral": name, "defectos": sum(map(check, ins)), "total": len(ins),
             "porcentaje": round(100 * sum(map(check, ins)) / len(ins), 1)}
            for name, check in checks.items()]


def box_xyxy(instance):
    """Caja en formato Pascal VOC (x1, y1, x2, y2), en px de su imagen."""
    width, height = instance["img_ancho"], instance["img_alto"]
    return ((instance["x_centro"] - instance["ancho_rel"] / 2) * width,
            (instance["y_centro"] - instance["alto_rel"] / 2) * height,
            (instance["x_centro"] + instance["ancho_rel"] / 2) * width,
            (instance["y_centro"] + instance["alto_rel"] / 2) * height)


def box_iou(a, b):
    """Intersection over Union entre dos cajas (x1, y1, x2, y2)."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def cisutac_label_checks(records):
    """CIS-2: controles automáticos de las cajas originales (superposiciones y cajas muy finas)."""
    by_capture = defaultdict(list)
    for i in records["instances"]:
        if i["version"] == "original":
            by_capture[i["captura"]].append(i)
    duplicates, overlapping, nested = [], 0, 0
    for capture, ins in by_capture.items():
        boxes = [box_xyxy(i) for i in ins]
        for k, a in enumerate(boxes):
            for b in boxes[k + 1:]:
                iou = box_iou(a, b)
                if iou >= 0.5:
                    duplicates.append(capture)
                elif iou > 0:
                    overlapping += 1
                if all(a[j] <= b[j] for j in (0, 1)) and all(a[j] >= b[j] for j in (2, 3)) \
                        or all(b[j] <= a[j] for j in (0, 1)) and all(b[j] >= a[j] for j in (2, 3)):
                    nested += 1
    thin = [i for ins in by_capture.values() for i in ins if min(i["ancho_px"], i["alto_px"]) < 4]
    return [
        {"control": "pares de cajas con IoU >= 0,5 (posible duplicado)", "valor": len(duplicates),
         "detalle": "; ".join(duplicates)},
        {"control": "pares de cajas superpuestas con 0 < IoU < 0,5", "valor": overlapping, "detalle": ""},
        {"control": "pares de cajas anidadas (una dentro de otra)", "valor": nested, "detalle": ""},
        {"control": "cajas con algún lado < 4 px", "valor": len(thin), "detalle": ""},
        {"control": "capturas con alguna caja de lado < 4 px", "valor": len({i["captura"] for i in thin}),
         "detalle": ""},
    ]


def cisutac_coco_ground_truth(records, captures, class_name):
    """Cajas reales de las capturas indicadas en formato COCO, en px de la imagen original.

    `area` es el área de la caja (misma definición que el EDA), que pycocotools usa para AP_small.
    """
    images_by_capture = {im["captura"]: im for im in records["images"] if im["version"] == "original"}
    by_capture = defaultdict(list)
    for inst in records["instances"]:
        if inst["version"] == "original":
            by_capture[inst["captura"]].append(inst)
    images, annotations = [], []
    for image_id, capture in enumerate(captures, start=1):
        im = images_by_capture[capture]
        images.append({"id": image_id, "file_name": f"{im['split']}/images/{im['archivo']}",
                       "width": im["img_ancho"], "height": im["img_alto"], "captura": capture})
        for inst in by_capture[capture]:
            x1, y1, x2, y2 = box_xyxy(inst)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0})
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": class_name}]}


def shifted_iou(side, shift):
    """IoU entre un cuadrado de lado `side` y la misma caja corrida `shift` px en x y en y."""
    inter = max(side - shift, 0) ** 2
    return inter / (2 * side ** 2 - inter)


def coco_size_counts(manifest, instances):
    """CIS-6: defectos small/medium/large de COCO por split propuesto, medidos en la imagen original."""
    split_of = {row["captura"]: row["split_propuesto"] for row in manifest}
    rows = []
    for split in ("train", "valid", "test"):
        sides = [equivalent_side(i) for i in instances if split_of.get(i["captura"]) == split]
        rows.append({"split": split, "instancias": len(sides),
                     "small_lado_menor_32": sum(s < 32 for s in sides),
                     "medium_32_a_96": sum(32 <= s < 96 for s in sides),
                     "large_96_o_mas": sum(s >= 96 for s in sides),
                     "diminutos_lado_menor_16": sum(s < 16 for s in sides)})
    return rows


def scaled_sizes(instances, mode=None):
    """Ancho y alto en px de cada caja tras redimensionar la imagen completa.

    `mode`: None (original), un entero (lado mayor tras un resize que conserva la proporción)
    o "640x640" (estirada a cuadrado, como la copia de Roboflow).
    """
    sizes = []
    for i in instances:
        width, height = i["img_ancho"], i["img_alto"]
        if mode is None:
            sx = sy = 1
        elif mode == "640x640":
            sx, sy = 640 / width, 640 / height
        else:
            sx = sy = mode / max(width, height)
        sizes.append((i["ancho_px"] * sx, i["alto_px"] * sy))
    return sizes


def size_summary(sizes):
    """Percentiles del lado equivalente y porcentaje de cajas que quedan con pocos píxeles."""
    sides = [(w * h) ** 0.5 for w, h in sizes]
    p10, p50, p90 = deciles(sides)

    def pct(condition):
        return round(100 * sum(map(condition, sizes)) / len(sizes), 1)

    return {"n": len(sizes), "lado_p10": round(p10, 1), "lado_p50": round(p50, 1), "lado_p90": round(p90, 1),
            "ancho_p50": round(median(w for w, _ in sizes), 1), "alto_p50": round(median(h for _, h in sizes), 1),
            "pct_lado_menor_32": pct(lambda s: (s[0] * s[1]) ** 0.5 < 32),
            "pct_lado_menor_8": pct(lambda s: (s[0] * s[1]) ** 0.5 < 8),
            "pct_lado_menor_4": pct(lambda s: (s[0] * s[1]) ** 0.5 < 4),
            "pct_algun_lado_menor_2": pct(lambda s: min(s) < 2)}


def match_copy_instances(records, tolerance=2 / 640):
    """Empareja cada instancia original con la de su copia 640×640.

    Sólo conserva capturas cuyas cajas coinciden todas (mismo número y cada caja a `tolerance`
    en coordenadas normalizadas, dos píxeles de la grilla de 640); las demás se descartan enteras.
    """
    by_capture = defaultdict(lambda: defaultdict(list))
    for i in records["instances"]:
        by_capture[i["captura"]][i["version"]].append(i)
    pairs = []
    for versions in by_capture.values():
        originals, copies = versions["original"], list(versions["640"])
        if not copies or len(copies) != len(originals):
            continue
        matched = []
        for inst in originals:
            def gap(c):
                return max(abs(inst[k] - c[k]) for k in ("x_centro", "y_centro", "ancho_rel", "alto_rel"))
            best = min(copies, key=gap)
            if gap(best) > tolerance:
                break
            copies.remove(best)
            matched.append((inst, best))
        else:
            pairs += matched
    return pairs


def cisutac_resize_table(records):
    """CIS-4: tamaño de los defectos tras redimensionar la foto completa, sin entrenar nada."""
    originals = [i for i in records["instances"] if i["version"] == "original"]
    pairs = match_copy_instances(records)
    rows = [{"resolucion": label, **size_summary(scaled_sizes(originals, mode))}
            for label, mode in (("original (2160×3840)", None), ("lado mayor 1280 (720×1280)", 1280),
                                ("lado mayor 1024 (576×1024)", 1024), ("lado mayor 640 (360×640)", 640),
                                ("640×640 estirada (simulada)", "640x640"))]
    rows.append({"resolucion": "640×640 copia real de Roboflow",
                 **size_summary([(copy["ancho_px"], copy["alto_px"]) for _, copy in pairs])})
    return rows


def tile_starts(length, tile, overlap):
    """Inicio de cada tile sobre un eje, como en SAHI: paso tile·(1 − overlap); el último toca el borde."""
    if length <= tile:
        return [0]
    return list(range(0, length - tile, int(tile * (1 - overlap)))) + [length - tile]


def cisutac_tiling_table(records, tiles=(512, 640, 1024), overlaps=(0.0, 0.2)):
    """CIS-4: costo de recorrer cada original con tiles y cuántos defectos entran enteros en algún tile.

    Supone que cada tile entra al modelo a su tamaño nativo (sin reescalar), así que los defectos
    conservan los píxeles de la foto original.
    """
    images = [im for im in records["images"] if im["version"] == "original"]
    instances = [i for i in records["instances"] if i["version"] == "original"]
    rows = []
    for tile in tiles:
        for overlap in overlaps:
            counts = [len(tile_starts(im["img_ancho"], tile, overlap)) * len(tile_starts(im["img_alto"], tile, overlap))
                      for im in images]

            def inside(i):
                width, height = i["img_ancho"], i["img_alto"]
                x1, x2 = (i["x_centro"] - i["ancho_rel"] / 2) * width, (i["x_centro"] + i["ancho_rel"] / 2) * width
                y1, y2 = (i["y_centro"] - i["alto_rel"] / 2) * height, (i["y_centro"] + i["alto_rel"] / 2) * height
                return (any(s - 1e-6 <= x1 and x2 <= s + tile + 1e-6 for s in tile_starts(width, tile, overlap))
                        and any(s - 1e-6 <= y1 and y2 <= s + tile + 1e-6 for s in tile_starts(height, tile, overlap)))

            rows.append({
                "tile_px": tile, "solapamiento": overlap, "tiles_por_imagen_mediana": float(median(counts)),
                "pixeles_procesados_sobre_imagen": round(median(
                    n * tile ** 2 / (im["img_ancho"] * im["img_alto"]) for n, im in zip(counts, images)), 2),
                "costo_relativo_a_una_pasada_640": round(median(counts) * tile ** 2 / 640 ** 2, 1),
                "pct_defectos_enteros_en_algun_tile": round(100 * sum(map(inside, instances)) / len(instances), 1),
            })
    return rows


# Bins de tamaño por lado equivalente √(ancho·alto) en px de la imagen original: <8, 8-16, 16-32, 32-64, ≥64.
SIZE_EDGES = (8, 16, 32, 64)


def size_histogram(instances):
    counts = [0] * (len(SIZE_EDGES) + 1)
    for i in instances:
        counts[sum(equivalent_side(i) >= edge for edge in SIZE_EDGES)] += 1
    return counts


def size_distance(counts, reference):
    """Distancia de variación total entre dos histogramas de tamaños (0 = iguales, 1 = disjuntos)."""
    return 0.5 * sum(abs(c / sum(counts) - r / sum(reference)) for c, r in zip(counts, reference))


def cisutac_split_proposal(records, test_range=(0.12, 0.18), valid_range=(0.10, 0.15)):
    """Manifiesto provisional CIS-5: train/valid/test agrupados por día de captura (sesión).

    Prueba todas las combinaciones de días y elige para test y valid, con la fracción de capturas
    pedida, las que más se parecen al dataset completo en distribución de tamaños de defecto
    (minimiza la peor de las dos distancias). Usa sólo etiquetas, nunca resultados de modelos.
    """
    originals = [im for im in records["images"] if im["version"] == "original"]
    days = sorted({im["captura"][:8] for im in originals})
    captures = Counter(im["captura"][:8] for im in originals)
    by_day = defaultdict(list)
    for i in records["instances"]:
        if i["version"] == "original":
            by_day[i["captura"][:8]].append(i)
    hist = {day: size_histogram(by_day[day]) for day in days}
    total = [sum(column) for column in zip(*hist.values())]

    def stats(mask):
        chosen = [day for k, day in enumerate(days) if mask >> k & 1]
        counts = [sum(column) for column in zip(*(hist[day] for day in chosen))]
        return sum(captures[day] for day in chosen) / len(originals), size_distance(counts, total)

    subsets = {mask: stats(mask) for mask in range(1, 1 << len(days)) if mask.bit_count() >= 2}
    tests = [(m, d) for m, (f, d) in subsets.items() if test_range[0] <= f <= test_range[1]]
    valids = [(m, d) for m, (f, d) in subsets.items() if valid_range[0] <= f <= valid_range[1]]
    *_, test_mask, valid_mask = min((max(dt, dv), dt + dv, mt, mv)
                                    for mt, dt in tests for mv, dv in valids if not mt & mv)

    def split(day):
        k = days.index(day)
        return "test" if test_mask >> k & 1 else "valid" if valid_mask >> k & 1 else "train"

    return [{"captura": im["captura"], "prenda": im["prenda"], "fecha": im["captura"][:8],
             "split_oficial": im["split"], "split_propuesto": split(im["captura"][:8]),
             "instancias": im["instancias"]}
            for im in originals]


def cisutac_split_summary(manifest, instances):
    """Por split: tamaño, sesiones y parecido de la distribución de tamaños con el dataset completo."""
    split_of = {row["captura"]: row["split_propuesto"] for row in manifest}
    reference = size_histogram([i for i in instances if i["captura"] in split_of])
    days = {split: {row["fecha"] for row in manifest if row["split_propuesto"] == split}
            for split in ("train", "valid", "test")}
    rows = []
    for split in ("train", "valid", "test"):
        caps = [row for row in manifest if row["split_propuesto"] == split]
        ins = [i for i in instances if split_of.get(i["captura"]) == split]
        if not caps:
            continue
        hist = size_histogram(ins)
        rows.append({
            "split": split, "capturas": len(caps), "prendas": len({row["prenda"] for row in caps}),
            "dias": len(days[split]), "instancias": len(ins),
            "dias_compartidos_con_otros_splits": len(days[split] & set().union(*(v for k, v in days.items() if k != split))),
            "pct_lado_menor_16px": round(100 * (hist[0] + hist[1]) / len(ins), 1),
            "lado_px_mediana": round(median(equivalent_side(i) for i in ins), 1),
            "dist_tamanos_vs_total": round(size_distance(hist, reference), 3),
        })
    return rows
