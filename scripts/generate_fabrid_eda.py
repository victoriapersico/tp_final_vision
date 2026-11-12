"""Genera FAB-1 a FAB-6 y el control FSD-1 del EDA de ReWear AI.

El script sólo lee los datasets originales y escribe derivados compactos en
``reports/fabrid``. No modifica imágenes, anotaciones ni splits oficiales.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_utils import (  # noqa: E402
    FABRID_CLASSES,
    fabrid_cross_split_sources,
    fabrid_export_records,
    fabrid_grouped_manifest,
    fabrid_normal_reconciliation,
    fabrid_size_by_class,
    image_files,
    roboflow_source_id,
)


DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
FABRID = DATA / "fabrid_v3"
EXPORT = FABRID / "annotations_export"
ORIGINAL = FABRID / "original"
OUT = ROOT / "reports" / "fabrid"


def locate_fsd():
    candidates = list((DATA / "fabric_spot_defect").glob("*/FabricSpotDefect"))
    if len(candidates) != 1:
        raise FileNotFoundError("No se pudo resolver de forma unívoca la carpeta FabricSpotDefect")
    return candidates[0]


def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        return
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def compact_image(path, maximum=1200):
    with Image.open(path) as image:
        result = image.convert("RGB")
        result.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
        return result.copy()


def original_inventory():
    inventory, context = [], []
    for category_dir in sorted(path for path in ORIGINAL.iterdir() if path.is_dir()):
        paths = image_files(category_dir)
        resolutions, modes, dates = Counter(), Counter(), Counter()
        for path in paths:
            with Image.open(path) as image:
                resolutions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
            parts = roboflow_source_id(path).split("_")
            dates[parts[1] if len(parts) > 1 and parts[0] == "IMG" else parts[0]] += 1
        inventory.append({
            "categoria_original": category_dir.name, "imagenes": len(paths),
            "modo": "; ".join(f"{key}:{value}" for key, value in modes.items()),
            "resoluciones": len(resolutions),
            "resolucion_principal": resolutions.most_common(1)[0][0],
            "cantidad_resolucion_principal": resolutions.most_common(1)[0][1],
        })
        context.append({
            "categoria_original": category_dir.name, "imagenes": len(paths),
            "imagenes_512x512": resolutions["512x512"],
            "fechas_o_lotes": "; ".join(f"{key}:{value}" for key, value in dates.most_common()),
            "resoluciones": "; ".join(f"{key}:{value}" for key, value in resolutions.most_common()),
        })
    return inventory, context


def export_summary(images, instances):
    rows = []
    for split in ("train", "valid", "test", "TOTAL"):
        subset = images if split == "TOTAL" else [row for row in images if row["split"] == split]
        ins = instances if split == "TOTAL" else [row for row in instances if row["split"] == split]
        rows.append({
            "split": split, "imagenes": len(subset),
            "etiquetas_vacias": sum(row["etiqueta_vacia"] for row in subset),
            "imagenes_con_defecto": sum(not row["etiqueta_vacia"] for row in subset),
            "instancias": len(ins), "ancho_max": max(row["img_ancho"] for row in subset),
            "alto_max": max(row["img_alto"] for row in subset),
        })
    return rows


COLORS = {
    "Needle Mark Defect": "#e41a1c", "hole Defect": "#377eb8",
    "missing yarn Defect": "#4daf4a", "stain Defect": "#984ea3",
}


def annotated_image(instance):
    image = Image.open(instance["path"]).convert("RGB")
    points = [(round(x * image.width), round(y * image.height)) for x, y in instance["points"]]
    draw = ImageDraw.Draw(image, "RGBA")
    draw.polygon(points, fill=(255, 40, 40, 55), outline=(255, 20, 20, 255), width=3)
    return image, points


def plot_classes(instances):
    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    for row, class_name in enumerate(FABRID_CLASSES):
        candidates = sorted((item for item in instances if item["clase"] == class_name),
                            key=lambda item: item["area_poligono_rel"])
        item = candidates[len(candidates) // 2]
        image, points = annotated_image(item)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{class_name} — vista completa")
        xs, ys = zip(*points)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 32)
        margin = max(20, round(span * 0.6))
        box = (max(0, min(xs) - margin), max(0, min(ys) - margin),
               min(image.width, max(xs) + margin), min(image.height, max(ys) + margin))
        axes[row, 1].imshow(image.crop(box))
        axes[row, 1].set_title(f"Zoom — área {100 * item['area_poligono_rel']:.3f} %")
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle("FabriD v3: ejemplos reproducibles de las cuatro clases", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "fab2_clases_poligonos.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_capture_context():
    categories = ["defect free", "Needle Mark Defect", "hole Defect", "missing yarn Defect", "stain Defect"]
    normal_paths = image_files(ORIGINAL / "defect free")
    selections = [
        ("Normal A", normal_paths[len(normal_paths) // 3]),
        ("Normal B", normal_paths[2 * len(normal_paths) // 3]),
    ]
    for category in categories[1:]:
        paths = image_files(ORIGINAL / category)
        selections.append((category, paths[len(paths) // 2]))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for axis, (label, path) in zip(axes.ravel(), selections):
        with Image.open(path) as source:
            resolution = f"{source.width}×{source.height}"
        axis.imshow(compact_image(path))
        axis.set_title(f"{label}\nOriginal {resolution}")
        axis.axis("off")
    fig.suptitle("FabriD v3: variación de captura en normales y clases defectuosas", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "fab2_normales_defectos.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sizes(instances):
    labels = [name.replace(" Defect", "") for name in FABRID_CLASSES]
    areas = [[100 * row["area_poligono_rel"] for row in instances if row["clase"] == name]
             for name in FABRID_CLASSES]
    sides = [[(row["bbox_ancho_px"] * row["bbox_alto_px"]) ** 0.5
              for row in instances if row["clase"] == name] for name in FABRID_CLASSES]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].boxplot(areas, tick_labels=labels, showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Área poligonal (% de imagen)")
    axes[0].set_title("Tamaño relativo")
    axes[1].boxplot(sides, tick_labels=labels, showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Lado equivalente de bbox (px del export)")
    axes[1].set_title("Escala en el export ≤640 px")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.2, axis="y")
    fig.suptitle("FabriD v3: distribución del tamaño de defectos por clase")
    fig.tight_layout()
    fig.savefig(OUT / "fab3_tamanos_por_clase.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cross_split(images, conflict_rows):
    by_source = defaultdict(list)
    for image in images:
        by_source[image["source_id"]].append(image)
    fig, axes = plt.subplots(7, 6, figsize=(15, 18))
    for group_index, conflict in enumerate(conflict_rows):
        block_row, block_col = divmod(group_index, 2)
        items = sorted(by_source[conflict["source_id"]], key=lambda row: (row["split"], row["archivo"]))
        for offset in range(3):
            axis = axes[block_row, block_col * 3 + offset]
            if offset < len(items):
                item = items[offset]
                axis.imshow(compact_image(item["path"], 500))
                axis.set_title(f"{conflict['source_id'].replace('IMG_', '')}\n{item['split']}", fontsize=8)
            axis.axis("off")
    fig.suptitle("FabriD v3: 14 capturas fuente con derivados en más de un split", fontsize=15, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "fab4_fuentes_repetidas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def split_counts(manifest):
    rows = []
    for split in ("train", "valid", "test"):
        subset = [row for row in manifest if row["split_propuesto"] == split]
        rows.append({
            "split_propuesto": split, "imagenes_totales": len(subset),
            "normales_concordantes": sum(row["estado_binario"] == "normal" for row in subset),
            "defectuosas_anotadas": sum(row["estado_binario"] == "defectuoso" for row in subset),
            "ambiguas_excluidas": sum(row["estado_binario"] == "ambiguo" for row in subset),
            "imagenes_movidas_desde_split_oficial": sum(row["movido"] for row in subset),
            "fuentes_unicas": len({row["source_id"] for row in subset}),
        })
    return rows


def fsd_analysis(fsd_root):
    original = fsd_root / "Orginal"
    category_counts, rows, all_images, declared_categories = Counter(), [], [], set()
    total_segmentations = 0
    for split in ("train", "valid", "test"):
        payload = json.loads((original / split / "_annotations.coco.json").read_text())
        categories = {row["id"]: row["name"] for row in payload["categories"]}
        declared_categories.update(categories.values())
        annotated_ids = {row["image_id"] for row in payload["annotations"]}
        total_segmentations += sum(bool(annotation.get("segmentation")) for annotation in payload["annotations"])
        for annotation in payload["annotations"]:
            category_counts[categories[annotation["category_id"]]] += 1
        rows.append({
            "unidad": f"original_{split}", "cantidad": len(payload["images"]),
            "detalle": (f"{len(payload['annotations'])} anotaciones; "
                        f"{sum(bool(a.get('segmentation')) for a in payload['annotations'])} con segmentación; "
                        f"{len(payload['images']) - len(annotated_ids)} sin anotaciones"),
        })
        all_images.append((split, payload, categories))
    rows += [
        {"unidad": "originales_total", "cantidad": 1014, "detalle": "643 train, 219 valid, 152 test"},
        {"unidad": "anotaciones_originales", "cantidad": sum(category_counts.values()),
         "detalle": "; ".join(f"{key}:{category_counts[key]}" for key in sorted(declared_categories))},
        {"unidad": "categorias_COCO_declaradas", "cantidad": len(declared_categories),
         "detalle": "; ".join(sorted(declared_categories))},
        {"unidad": "categorias_con_anotaciones", "cantidad": sum(category_counts[name] > 0 for name in declared_categories),
         "detalle": "; ".join(name for name in sorted(declared_categories) if category_counts[name] > 0)},
        {"unidad": "anotaciones_con_segmentacion", "cantidad": total_segmentations,
         "detalle": "las restantes contienen bounding box sin polígono útil"},
    ]
    augmented_sources = {}
    for representation in ("COCO", "YOLOv8"):
        paths = []
        for split in ("train", "valid", "test"):
            folder = fsd_root / "Augmented" / representation / split
            if representation == "YOLOv8":
                folder /= "images"
            paths += image_files(folder)
        augmented_sources[representation] = Counter(roboflow_source_id(path) for path in paths)
        rows.append({"unidad": f"augmentadas_{representation}", "cantidad": len(paths),
                     "detalle": "416x416; representación alternativa del mismo export"})
    rows.append({
        "unidad": "multiconjunto_fuentes_COCO_igual_YOLO", "cantidad": int(
            augmented_sources["COCO"] == augmented_sources["YOLOv8"]),
        "detalle": "1 significa mismos 2.300 ejemplos fuente; no sumarlos como observaciones",
    })
    return rows, all_images


def plot_fsd_examples(fsd_root, datasets):
    candidates = []
    for split, payload, categories in datasets:
        by_image = defaultdict(list)
        for annotation in payload["annotations"]:
            by_image[annotation["image_id"]].append(annotation)
        image_map = {row["id"]: row for row in payload["images"]}
        for image_id, annotations in by_image.items():
            names = {categories[row["category_id"]] for row in annotations}
            candidates.append((split, image_map[image_id], annotations, categories, names))
    selected = []
    selected.append(next(row for row in candidates if len(row[2]) == 1))
    selected.append(next(row for row in candidates if row[1]["file_name"] != selected[0][1]["file_name"]
                         and any(annotation.get("segmentation") for annotation in row[2])))
    selected.append(max((row for row in candidates if row[1]["file_name"] not in
                         {item[1]["file_name"] for item in selected}), key=lambda row: len(row[2])))
    colors = {"spot": (230, 40, 40), "Hole-EnBF": (30, 120, 230)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (split, image_row, annotations, categories, _) in zip(axes, selected):
        path = fsd_root / "Orginal" / split / image_row["file_name"]
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for annotation in annotations:
            name = categories[annotation["category_id"]]
            x, y, width, height = annotation["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline=colors[name], width=max(2, image.width // 300))
        axis.imshow(image)
        axis.set_title(f"{split} — {len(annotations)} cajas\n" + ", ".join(sorted({categories[a['category_id']] for a in annotations})))
        axis.axis("off")
    fig.suptitle("FabricSpotDefect: ejemplos originales con bounding boxes", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "fsd1_ejemplos_cajas.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    if not EXPORT.is_dir() or not ORIGINAL.is_dir():
        raise FileNotFoundError(f"No se encontró FabriD v3 bajo {FABRID}")
    fsd_root = locate_fsd()
    OUT.mkdir(parents=True, exist_ok=True)

    images, instances, class_names = fabrid_export_records(EXPORT)
    inventory, context = original_inventory()
    export_rows = export_summary(images, instances)
    sizes = fabrid_size_by_class(instances)
    conflicts = fabrid_cross_split_sources(EXPORT)
    manifest = fabrid_grouped_manifest(EXPORT, ORIGINAL)
    normal_cases, normal_summary = fabrid_normal_reconciliation(EXPORT, ORIGINAL)
    protocol_counts = split_counts(manifest)
    fsd_rows, fsd_datasets = fsd_analysis(fsd_root)

    write_csv("fab1_inventario_original.csv", inventory)
    write_csv("fab1_export_splits.csv", export_rows)
    write_csv("fab2_contexto_captura.csv", context)
    write_csv("fab3_tamanos_clase.csv", sizes)
    write_csv("fab4_fuentes_repetidas.csv", conflicts)
    write_csv("fab4_split_agrupado.csv", manifest)
    write_csv("fab4_split_resumen.csv", protocol_counts)
    write_csv("fab5_conciliacion_normales.csv", normal_cases)
    write_csv("fab5_normales_resumen.csv", normal_summary)
    write_csv("fab6_protocolo_conteos.csv", protocol_counts)
    write_csv("fsd1_resumen.csv", fsd_rows)

    plot_classes(instances)
    plot_capture_context()
    plot_sizes(instances)
    plot_cross_split(images, conflicts)
    plot_fsd_examples(fsd_root, fsd_datasets)

    summary = {
        "clases": class_names, "inventario_original": inventory, "export_splits": export_rows,
        "tamanos": sizes, "fuentes_cross_split": conflicts,
        "normales": normal_summary, "protocolo": protocol_counts, "fabric_spot_defect": fsd_rows,
    }
    (OUT / "fab_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
