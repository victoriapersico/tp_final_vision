"""Genera el bloque RAW-1 a RAW-6 del EDA de RAW-FABRID.

Sólo lee el dataset original. Todas las tablas y figuras derivadas se escriben en
``reports/raw_fabrid``.
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
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda_utils import (  # noqa: E402
    raw_fabrid_channel_check,
    raw_fabrid_data,
    raw_fabrid_orphan_masks,
    raw_fabrid_size_quantiles,
    raw_fabrid_traceability,
)


DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
RAW = DATA / "raw_fabrid" / "extracted" / "RAW_FABRID"
OUT = ROOT / "reports" / "raw_fabrid"


def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        return
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mask_overlay(image, mask, alpha=120):
    base = image.convert("RGBA")
    red = Image.new("RGBA", base.size, (255, 30, 30, 0))
    red.putalpha(mask.convert("L").point(lambda value: alpha if value else 0))
    return Image.alpha_composite(base, red)


def plot_originals(high, high_coco):
    images = {item["id"]: item for item in high_coco["images"]}
    annotations = sorted(high_coco["annotations"], key=lambda item: item["area"])
    chosen = [annotations[round((len(annotations) - 1) * q)] for q in (0.1, 0.5, 0.9)]
    normal_names = sorted(row["filename"] for row in high if row["has_defect"].lower() == "false")
    normal = Image.open(RAW / "images" / normal_names[len(normal_names) // 2])
    empty_mask = Image.new("L", normal.size, 0)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes[0, 0].imshow(normal, cmap="gray")
    axes[0, 0].set_title("Original normal")
    axes[0, 1].imshow(empty_mask, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Sin máscara (normal según CSV)")
    axes[0, 2].imshow(normal, cmap="gray")
    axes[0, 2].set_title("Vista completa")
    for axis in axes[0]:
        axis.axis("off")
    for row, annotation in enumerate(chosen, start=1):
        name = images[annotation["image_id"]]["file_name"]
        image = Image.open(RAW / "images" / name)
        mask = Image.open(RAW / "masks" / name)
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"Original defectuoso — p{(10, 50, 90)[row - 1]} por área")
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"Máscara — {int(sum(mask.histogram()[1:])):,} px")
        axes[row, 2].imshow(mask_overlay(image, mask))
        axes[row, 2].set_title("Superposición")
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle("RAW-FABRID: original normal, defectuosos y máscaras", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "raw2_originales_mascaras.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_patches(high, patches):
    by_parent = defaultdict(lambda: defaultdict(list))
    for row in patches:
        by_parent[row["parent_raw"]][row["type"]].append(row["patch_name"])
    candidates = sorted(parent for parent, groups in by_parent.items()
                        if groups["test_good"] and groups["test_defect"])
    selected = [candidates[round((len(candidates) - 1) * q)] for q in (0.15, 0.5, 0.85)]
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for row, parent in enumerate(selected):
        good = sorted(by_parent[parent]["test_good"])[0]
        defect = sorted(by_parent[parent]["test_defect"])[0]
        image = Image.open(RAW / "images" / parent)
        good_image = Image.open(RAW / "MVTec" / "test" / "good" / good)
        defect_image = Image.open(RAW / "MVTec" / "test" / "defect" / defect)
        mask = Image.open(RAW / "MVTec" / "ground_truth" / "defect" / defect)
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title("Imagen madre")
        axes[row, 1].imshow(good_image, cmap="gray")
        axes[row, 1].set_title("Patch test/good")
        axes[row, 2].imshow(defect_image, cmap="gray")
        axes[row, 2].set_title("Patch test/defect")
        axes[row, 3].imshow(mask_overlay(defect_image, mask))
        axes[row, 3].set_title("Defecto + máscara")
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle("RAW-FABRID: patches y trazabilidad a la imagen madre", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "raw2_parches.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sizes(high_coco, patches):
    high_areas = [100 * float(item["area"]) / (1792 * 1024) for item in high_coco["annotations"]]
    patch_areas = [100 * float(row["defect_area"]) / (256 * 256)
                   for row in patches if row["type"] == "test_defect"]
    bins = [10 ** (-5 + index * 0.35) for index in range(22)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes[0].hist(high_areas, bins=bins, color="#355C7D", edgecolor="white")
    axes[0].set_title("Original 1792×1024")
    axes[1].hist(patch_areas, bins=bins, color="#C06C84", edgecolor="white")
    axes[1].set_title("Patch defectuoso 256×256")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Área de máscara/polígono (% de la imagen)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Cantidad")
    fig.suptitle("Escala relativa de las anomalías (denominadores distintos)")
    fig.tight_layout()
    fig.savefig(OUT / "raw3_distribucion_tamanos.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    if not RAW.is_dir():
        raise FileNotFoundError(f"No se encontró RAW-FABRID en {RAW}")
    OUT.mkdir(parents=True, exist_ok=True)
    high, patches, high_coco, patch_coco = raw_fabrid_data(RAW)

    images = sorted((RAW / "images").glob("*.png"))
    masks = sorted((RAW / "masks").glob("*.png"))
    mvtec_counts = {
        "train_good": len(list((RAW / "MVTec" / "train" / "good").glob("*.png"))),
        "test_good": len(list((RAW / "MVTec" / "test" / "good").glob("*.png"))),
        "test_defect": len(list((RAW / "MVTec" / "test" / "defect").glob("*.png"))),
        "ground_truth_defect": len(list((RAW / "MVTec" / "ground_truth" / "defect").glob("*.png"))),
    }
    inventory = [
        {"unidad": "originales", "cantidad": len(images), "detalle": "PNG 1792x1024"},
        {"unidad": "originales_normales_csv", "cantidad": sum(r["has_defect"].lower() == "false" for r in high),
         "detalle": "sin defectos según CSV"},
        {"unidad": "originales_defectuosas_csv", "cantidad": sum(r["has_defect"].lower() == "true" for r in high),
         "detalle": "con defectos según CSV"},
        {"unidad": "mascaras_highres", "cantidad": len(masks), "detalle": "incluye 2 huérfanas"},
        {"unidad": "anotaciones_highres_coco", "cantidad": len(high_coco["annotations"]),
         "detalle": "bbox + polígono"},
        {"unidad": "patches_totales", "cantidad": len(patches), "detalle": "todos 256x256"},
        *({"unidad": key, "cantidad": value, "detalle": "estructura MVTec"}
          for key, value in mvtec_counts.items()),
        {"unidad": "anotaciones_patches_coco", "cantidad": len(patch_coco["annotations"]),
         "detalle": "un patch puede contener varias instancias"},
    ]
    write_csv("raw1_inventario.csv", inventory)

    roll_rows = []
    for roll in sorted({row["roll_id"] for row in high}):
        subset = [row for row in high if row["roll_id"] == roll]
        roll_rows.append({"roll_id": roll, "originales": len(subset),
                          "normales": sum(row["has_defect"].lower() == "false" for row in subset),
                          "defectuosas": sum(row["has_defect"].lower() == "true" for row in subset),
                          "instancias": sum(int(row["num_defects"]) for row in subset)})
    write_csv("raw1_rollos.csv", roll_rows)
    channel_rows = raw_fabrid_channel_check(RAW)
    write_csv("raw1_canales_muestra.csv", channel_rows)

    quantiles = raw_fabrid_size_quantiles(RAW)
    write_csv("raw3_cuantiles.csv", quantiles)
    orphans = raw_fabrid_orphan_masks(RAW)
    write_csv("raw4_discrepancia_202_204.csv", orphans)
    trace, trace_rolls, intersections = raw_fabrid_traceability(RAW)
    write_csv("raw5_trazabilidad.csv", trace)
    write_csv("raw5_trazabilidad_rollos.csv", trace_rolls)
    write_csv("raw5_intersecciones.csv", intersections)

    plot_originals(high, high_coco)
    plot_patches(high, patches)
    plot_sizes(high_coco, patches)

    summary = {
        "inventario": inventory,
        "rollos": roll_rows,
        "canales_muestra": {"n": len(channel_rows),
                            "rgb": sum(row["modo_pil"] == "RGB" for row in channel_rows),
                            "canales_identicos": sum(row["canales_identicos"] for row in channel_rows)},
        "cuantiles": quantiles,
        "mascaras_huerfanas": orphans,
        "trazabilidad": trace,
        "intersecciones": intersections,
    }
    (OUT / "raw_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
