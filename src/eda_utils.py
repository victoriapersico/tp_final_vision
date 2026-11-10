"""Utilidades reproducibles para auditar los datasets de ReWear AI.

Este módulo sólo lee imágenes y anotaciones; nunca modifica los originales.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

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
