"""Figura de aciertos y falsos positivos de una o más corridas de scripts/evaluate_cisutac.py.

Para cada corrida, una fila con los aciertos y los falsos positivos de mayor confianza (a lo sumo uno por
imagen), sobre las predicciones que superan el umbral elegido en esa corrida (`umbral_confianza` de
metricas.json).
Caja llena: defecto real; caja punteada: predicción. Recortes de la imagen original.

Ejemplo:
  python scripts/plot_cisutac_predictions.py --dataset holes \
      --corrida results/cisutac/baseline/holes_valid_completa640 --etiqueta "Foto completa 640" \
      --corrida results/cisutac/baseline/holes_valid_tiles640 --etiqueta "Tiles 640" \
      --salida reports/cisutac/baseline_ejemplos_holes.jpg
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection_eval import xywh_to_xyxy  # noqa: E402
from eda_utils import box_iou  # noqa: E402

DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
COLORS = {"holes": "#2a78d6", "spots": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "text.color": INK, "figure.facecolor": "#fcfcfb"})

parser = argparse.ArgumentParser(description="Aciertos y falsos positivos de corridas de evaluate_cisutac.py.")
parser.add_argument("--dataset", choices=("holes", "spots"), required=True)
parser.add_argument("--corrida", type=Path, action="append", required=True, help="carpeta con metricas.json")
parser.add_argument("--etiqueta", action="append", help="nombre de cada corrida en la figura (mismo orden)")
parser.add_argument("--aciertos", type=int, default=2)
parser.add_argument("--errores", type=int, default=4)
parser.add_argument("--salida", type=Path, required=True)
args = parser.parse_args()
labels = args.etiqueta or [run.name for run in args.corrida]

gt = json.loads((DATA / "derived" / f"cisutac_{args.dataset}" / "coco_valid.json").read_text())
files = {im["id"]: im["file_name"] for im in gt["images"]}
truths = {}
for a in gt["annotations"]:
    truths.setdefault(a["image_id"], []).append(xywh_to_xyxy(a["bbox"]))
images = {}


def load(image_id):
    if image_id not in images:
        path = DATA / f"cisutac_{args.dataset}" / files[image_id]
        images[image_id] = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return images[image_id]


def examples(run):
    """Aciertos y falsos positivos de mayor confianza, con el mejor IoU contra las cajas reales."""
    threshold = json.loads((run / "metricas.json").read_text())["umbral_confianza"]
    kept = sorted((p for p in json.loads((run / "predicciones.json").read_text()) if p["score"] >= threshold),
                  key=lambda p: -p["score"])
    scored = [(p, max((box_iou(xywh_to_xyxy(p["bbox"]), t) for t in truths.get(p["image_id"], [])), default=0.0))
              for p in kept]
    def first_per_image(items, n):  # un ejemplo por imagen, para que no se repita la misma prenda
        seen, chosen = set(), []
        for p, iou in items:
            if p["image_id"] not in seen:
                seen.add(p["image_id"])
                chosen.append((p, iou))
        return chosen[:n]

    return (first_per_image([s for s in scored if s[1] >= 0.5], args.aciertos),
            first_per_image([s for s in scored if s[1] < 0.5], args.errores))


halo = [patheffects.withStroke(linewidth=3, foreground="white")]
columns = args.aciertos + args.errores
fig, axes = plt.subplots(len(args.corrida), columns, figsize=(2.3 * columns, 3.1 * len(args.corrida)), squeeze=False)
for row_axes, run, label in zip(axes, args.corrida, labels):
    hits, errors = examples(run)
    cells = [("acierto", s) for s in hits] + [None] * (args.aciertos - len(hits)) + [("falso positivo", s) for s in errors]
    for ax, cell in zip(row_axes, cells):
        ax.axis("off")
        if cell is None:
            continue
        kind, (p, iou) = cell
        image = load(p["image_id"])
        x1, y1, x2, y2 = xywh_to_xyxy(p["bbox"])
        window = int(max(160, 1.6 * max(x2 - x1, y2 - y1)))
        x0 = int(min(max((x1 + x2 - window) / 2, 0), image.width - window))
        y0 = int(min(max((y1 + y2 - window) / 2, 0), image.height - window))
        ax.imshow(image.crop((x0, y0, x0 + window, y0 + window)))
        for t in truths.get(p["image_id"], []):
            ax.add_patch(plt.Rectangle((t[0] - x0, t[1] - y0), t[2] - t[0], t[3] - t[1], fill=False,
                                       ec=COLORS[args.dataset], lw=1.2, path_effects=halo))
        ax.add_patch(plt.Rectangle((x1 - x0, y1 - y0), x2 - x1, y2 - y1, fill=False, ec=INK, lw=1.2, ls="--",
                                   path_effects=halo))
        side = ((x2 - x1) * (y2 - y1)) ** 0.5
        note = f"\nIoU {iou:.2f} con un defecto real" if 0.1 <= iou < 0.5 else ""
        ax.set_title(f"{kind}\nconfianza {p['score']:.2f} · lado {side:.0f} px{note}".replace(".", ","), fontsize=8,
                     color=INK if kind == "acierto" else MUTED)
    row_axes[0].text(-0.08, 0.5, label, transform=row_axes[0].transAxes, rotation=90, ha="right", va="center",
                     fontsize=9, fontweight="bold")
fig.suptitle(f"CISUTAC {args.dataset.capitalize()}, valid: caja llena = defecto real, punteada = predicción",
             fontsize=10)
fig.tight_layout(rect=(0.02, 0, 1, 0.97), h_pad=2.5)
args.salida.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.salida, dpi=110, pil_kwargs={"quality": 88})
print(f"Figura en {args.salida}")
