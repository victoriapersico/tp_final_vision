"""Genera la comparación visual compacta de los tres dominios del EDA."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data"))
OUTPUT = ROOT / "reports" / "integracion"


DOMAIN_ROWS = [
    {
        "dominio": "Prendas usadas",
        "dataset": "CISUTAC Holes / Spots",
        "captura": "Prenda completa, fondo removido",
        "anotacion": "Cajas YOLO",
        "normales_verificadas": "No",
        "uso": "Detección supervisada y prueba de tiling",
        "riesgo_principal": "Defecto diminuto al reducir la foto",
    },
    {
        "dominio": "Tejidos RGB variables",
        "dataset": "FabriD v3",
        "captura": "Primer plano con cámara/teléfono",
        "anotacion": "Polígonos YOLO-seg",
        "normales_verificadas": "607 concordantes",
        "uso": "Segmentación y comparación con PatchCore",
        "riesgo_principal": "Atajos de resolución/lote y derivados entre splits",
    },
    {
        "dominio": "Inspección industrial",
        "dataset": "RAW-FABRID",
        "captura": "Line-scan controlado; apariencia monocroma",
        "anotacion": "Máscaras y metadatos",
        "normales_verificadas": "Sí",
        "uso": "Benchmark principal de PatchCore",
        "riesgo_principal": "Patches dependientes de la misma captura/rollo",
    },
]


def image_files(folder: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in suffixes)


def choose_cisutac_original() -> Path:
    candidates = image_files(DATA / "cisutac_holes" / "train" / "images")
    for path in candidates:
        with Image.open(path) as image:
            if max(image.size) > 900:
                return path
    raise FileNotFoundError("No se encontró una captura original de CISUTAC Holes")


def choose_fabrid() -> Path:
    candidates = image_files(DATA / "fabrid_v3" / "original" / "stain Defect")
    if not candidates:
        raise FileNotFoundError("No se encontró la clase stain de FabriD v3")
    return candidates[len(candidates) // 2]


def choose_raw_fabrid() -> Path:
    candidates = image_files(
        DATA / "raw_fabrid" / "extracted" / "RAW_FABRID" / "MVTec" / "test" / "defect"
    )
    if not candidates:
        raise FileNotFoundError("No se encontró test/defect de RAW-FABRID")
    return candidates[len(candidates) // 2]


def center_crop(image: Image.Image, ratio: float = 1.25) -> Image.Image:
    width, height = image.size
    current = width / height
    if current > ratio:
        new_width = int(height * ratio)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    new_height = int(width / ratio)
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "comparacion_dominios.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DOMAIN_ROWS[0].keys())
        writer.writeheader()
        writer.writerows(DOMAIN_ROWS)

    examples = [
        (choose_cisutac_original(), "Prenda usada\nCISUTAC"),
        (choose_fabrid(), "Tejido RGB\nFabriD"),
        (choose_raw_fabrid(), "Inspección industrial\nRAW-FABRID"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), constrained_layout=True)
    for axis, (path, title) in zip(axes, examples):
        with Image.open(path) as source:
            image = center_crop(source.convert("RGB"))
        axis.imshow(image)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.axis("off")

    fig.suptitle(
        "El mismo problema cambia mucho según cómo se toma la imagen",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / "comparacion_dominios.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Tabla: {csv_path.relative_to(ROOT)}")
    print(f"Figura: {(OUTPUT / 'comparacion_dominios.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
