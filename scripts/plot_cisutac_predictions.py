"""Figuras con ejemplos de las predicciones de corridas de evaluación en CISUTAC (valid).

Siempre sobre las predicciones que superan el umbral elegido en cada corrida (`umbral_confianza` de
metricas.json). Caja llena: defecto real; caja punteada: predicción. Recortes de la imagen original.

- Modo por defecto: para cada corrida, una fila con los aciertos y los falsos positivos de mayor confianza
  (a lo sumo uno por imagen).
- --por-error: para una corrida, una fila por tipo de error de src/detection_eval.py (alucinaciones, errores
  de localización y duplicados, los de mayor confianza; y defectos no detectados, repartidos por tamaño).
  Muestra qué confunde el modelo con un defecto y cómo son los que no encuentra.

Ejemplos:
  python scripts/plot_cisutac_predictions.py --dataset holes \
      --corrida results/cisutac/baseline/holes_valid_completa640 --etiqueta "Foto completa 640" \
      --corrida results/cisutac/baseline/holes_valid_tiles640 --etiqueta "Tiles 640" \
      --salida reports/cisutac/baseline_ejemplos_holes.jpg
  python scripts/plot_cisutac_predictions.py --dataset holes --por-error \
      --corrida results/cisutac/baseline/holes_valid_completa640 --salida errores_ejemplos.jpg
"""

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection_eval import box_states, match_detections, prediction_categories, xywh_to_xyxy
from eda_utils import box_iou

DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
COLORS = {"holes": "#2a78d6", "spots": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
# Mismos colores por tipo de error que scripts/analyze_detection_errors.py.
ERROR_ROWS = (("alucinaciones", "alucinaciones", "#eda100"), ("localizacion", "errores de localización", "#eb6834"),
              ("duplicados", "duplicados", "#1baf7a"), ("no_detectados", "no detectados", "#6250d6"))
HALO = [patheffects.withStroke(linewidth=3, foreground="white")]
STYLE = {"font.size": 9, "text.color": INK, "figure.facecolor": "#fcfcfb"}


def styled(style):
    """Decorador: aplica `style` sólo mientras se dibuja y después restaura esas claves.

    Así importar o usar estas funciones (p. ej. desde un notebook) no cambia el estilo de quien las llama. No
    usa plt.rc_context: en Jupyter, con matplotlib 3.11, deja de mostrar los gráficos de las celdas siguientes.
    """
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            saved = {key: plt.rcParams[key] for key in style}
            plt.rcParams.update(style)
            try:
                return function(*args, **kwargs)
            finally:
                plt.rcParams.update(saved)
        return wrapper
    return decorator


class Valid:
    """Cajas reales de valid (las primeras `n_images` fotos, como recortan las evaluaciones) y sus fotos."""

    def __init__(self, dataset, n_images=None):
        self.dataset = dataset
        self.gt = json.loads((DATA / "derived" / f"cisutac_{dataset}" / "coco_valid.json").read_text())
        self.gt["images"] = self.gt["images"][:n_images]
        kept = {im["id"] for im in self.gt["images"]}
        self.gt["annotations"] = [a for a in self.gt["annotations"] if a["image_id"] in kept]
        self.files = {im["id"]: im["file_name"] for im in self.gt["images"]}
        self.truths = {}
        for a in self.gt["annotations"]:  # mismo orden que match_detections
            self.truths.setdefault(a["image_id"], []).append(xywh_to_xyxy(a["bbox"]))
        self._images = {}

    def image(self, image_id):
        if image_id not in self._images:
            path = DATA / f"cisutac_{self.dataset}" / self.files[image_id]
            self._images[image_id] = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return self._images[image_id]


def load_run(run):
    metrics = json.loads((run / "metricas.json").read_text())
    if metrics.get("configuracion", metrics)["split"] != "valid":
        raise ValueError(f"{run.name}: sólo se grafican corridas de valid")
    return metrics, json.loads((run / "predicciones.json").read_text())


def first_per_image(items, n, image_of):
    """Los primeros `n` elementos con imágenes distintas, para que no se repita la misma prenda."""
    seen, chosen = set(), []
    for item in items:
        if image_of(item) not in seen:
            seen.add(image_of(item))
            chosen.append(item)
    return chosen[:n]


def draw_crop(ax, valid, image_id, box, others=(), highlight=True):
    """Recorte alrededor de `box` (xyxy) con las cajas reales (llenas) y predicciones (punteadas).

    Con `highlight`, `box` es una predicción y se dibuja gruesa; si no, es un defecto real y se remarca.
    """
    image = valid.image(image_id)
    x1, y1, x2, y2 = box
    window = int(max(160, 1.6 * max(x2 - x1, y2 - y1)))
    x0 = int(min(max((x1 + x2 - window) / 2, 0), image.width - window))
    y0 = int(min(max((y1 + y2 - window) / 2, 0), image.height - window))
    ax.imshow(image.crop((x0, y0, x0 + window, y0 + window)))
    for t in valid.truths.get(image_id, []):
        ax.add_patch(plt.Rectangle((t[0] - x0, t[1] - y0), t[2] - t[0], t[3] - t[1], fill=False,
                                   ec=COLORS[valid.dataset], lw=1.2, path_effects=HALO))
    for o in others:
        ax.add_patch(plt.Rectangle((o[0] - x0, o[1] - y0), o[2] - o[0], o[3] - o[1], fill=False, ec=INK, lw=0.7,
                                   ls="--", path_effects=HALO))
    if highlight:
        ax.add_patch(plt.Rectangle((x1 - x0, y1 - y0), x2 - x1, y2 - y1, fill=False, ec=INK, lw=1.2, ls="--",
                                   path_effects=HALO))
    else:
        ax.add_patch(plt.Rectangle((x1 - x0, y1 - y0), x2 - x1, y2 - y1, fill=False, ec=COLORS[valid.dataset],
                                   lw=2.4, path_effects=HALO))
    ax.axis("off")


@styled(STYLE)
def plot_hits_and_fps(dataset, runs, labels, n_hits, n_errors, out):
    """Modo por defecto: aciertos y falsos positivos de mayor confianza, una fila por corrida."""
    valid = Valid(dataset)
    columns = n_hits + n_errors
    fig, axes = plt.subplots(len(runs), columns, figsize=(2.3 * columns, 3.1 * len(runs)), squeeze=False)
    for row_axes, run, label in zip(axes, runs, labels):
        metrics, predictions = load_run(run)
        kept = sorted((p for p in predictions if p["score"] >= metrics["umbral_confianza"]), key=lambda p: -p["score"])
        scored = [(p, max((box_iou(xywh_to_xyxy(p["bbox"]), t) for t in valid.truths.get(p["image_id"], [])),
                          default=0.0)) for p in kept]
        hits = first_per_image([s for s in scored if s[1] >= 0.5], n_hits, lambda s: s[0]["image_id"])
        errors = first_per_image([s for s in scored if s[1] < 0.5], n_errors, lambda s: s[0]["image_id"])
        cells = [("acierto", s) for s in hits] + [None] * (n_hits - len(hits)) + [("falso positivo", s) for s in errors]
        for ax, cell in zip(row_axes, cells):
            ax.axis("off")
            if cell is None:
                continue
            kind, (p, iou) = cell
            box = xywh_to_xyxy(p["bbox"])
            draw_crop(ax, valid, p["image_id"], box)
            side = ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5
            note = f"\nIoU {iou:.2f} con un defecto real" if 0.1 <= iou < 0.5 else ""
            ax.set_title(f"{kind}\nconfianza {p['score']:.2f} · lado {side:.0f} px{note}".replace(".", ","),
                         fontsize=8, color=INK if kind == "acierto" else MUTED)
        row_axes[0].text(-0.08, 0.5, label, transform=row_axes[0].transAxes, rotation=90, ha="right", va="center",
                         fontsize=9, fontweight="bold")
    fig.suptitle(f"CISUTAC {dataset.capitalize()}, valid: caja llena = defecto real, punteada = predicción",
                 fontsize=10)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97), h_pad=2.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)


@styled(STYLE)
def plot_error_examples(dataset, run, out, per_row=5):
    """--por-error: una fila por tipo de error de una corrida de valid, con las categorías de detection_eval."""
    metrics, predictions = load_run(run)
    valid = Valid(dataset, metrics["imagenes"])
    threshold = metrics["umbral_confianza"]
    strict = match_detections(valid.gt, predictions, 0.5, threshold)
    loose = match_detections(valid.gt, predictions, 0.1, threshold)
    categories = prediction_categories(strict)
    details = {d["indice"]: d for image in strict.values() for d in image["detalle"]}
    by_confidence = sorted(categories, key=lambda k: -predictions[k]["score"])
    kept_boxes = {}
    for k in by_confidence:
        kept_boxes.setdefault(predictions[k]["image_id"], []).append(xywh_to_xyxy(predictions[k]["bbox"]))
    rows, totals = {}, {}
    for kind, _, _ in ERROR_ROWS[:3]:
        chosen = [k for k in by_confidence if categories[k] == kind]
        totals[kind] = len(chosen)
        rows[kind] = first_per_image(chosen, per_row, lambda k: predictions[k]["image_id"])
    missed = sorted(((image_id, box) for image_id, states in box_states(strict, loose).items()
                     for box, state in zip(valid.truths.get(image_id, []), states) if state == "no_detectados"),
                    key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))
    totals["no_detectados"] = len(missed)
    # Repartidos por tamaño: posiciones equiespaciadas en la lista ordenada de menor a mayor.
    steps = sorted({round(i * (len(missed) - 1) / max(per_row - 1, 1)) for i in range(per_row)}) if missed else []
    rows["no_detectados"] = [missed[i] for i in steps]

    fig, axes = plt.subplots(len(ERROR_ROWS), per_row, figsize=(2.3 * per_row, 2.9 * len(ERROR_ROWS)), squeeze=False)
    for row_axes, (kind, name, color) in zip(axes, ERROR_ROWS):
        for ax in row_axes:
            ax.axis("off")
        for ax, item in zip(row_axes, rows[kind]):
            if kind == "no_detectados":
                image_id, box = item
                draw_crop(ax, valid, image_id, box, others=kept_boxes.get(image_id, []), highlight=False)
                side = ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5
                ax.set_title(f"lado {side:.0f} px\nninguna predicción con IoU ≥ 0,1", fontsize=8, color=MUTED)
                continue
            p, d = predictions[item], details[item]
            box = xywh_to_xyxy(p["bbox"])
            draw_crop(ax, valid, p["image_id"], box)
            side = ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5
            note = {"localizacion": f"\nIoU {d['iou_max']:.2f} con un defecto",
                    "duplicados": f"\nIoU {d['iou_max']:.2f} con un defecto ya detectado"}.get(kind, "")
            ax.set_title(f"confianza {p['score']:.2f} · lado {side:.0f} px{note}".replace(".", ","), fontsize=8,
                         color=MUTED)
        if not rows[kind]:
            row_axes[0].text(0.5, 0.5, "sin casos", ha="center", va="center", color=MUTED,
                             transform=row_axes[0].transAxes)
        total = len(categories) if kind != "no_detectados" else len(valid.gt["annotations"])
        unit = "predicciones" if kind != "no_detectados" else "defectos"
        row_axes[0].text(-0.12, 0.5, f"{name}\n{totals[kind]} de {total} {unit}", transform=row_axes[0].transAxes,
                         rotation=90, ha="right", va="center", fontsize=9, fontweight="bold")
        row_axes[0].add_patch(plt.Rectangle((-0.1, 0.3), 0.04, 0.4, transform=row_axes[0].transAxes, color=color,
                                            clip_on=False))
    fig.suptitle(f"{run.name}: errores con el umbral de valid ({threshold:.2f})".replace(".", ",") +
                 "\ncaja llena = defecto real · punteada = predicción · la más gruesa es el caso de cada recuadro", fontsize=10)
    fig.tight_layout(rect=(0.03, 0, 1, 0.95), h_pad=2.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Ejemplos de predicciones de corridas de evaluación (valid).")
    parser.add_argument("--dataset", choices=("holes", "spots"), required=True)
    parser.add_argument("--corrida", type=Path, action="append", required=True, help="carpeta con metricas.json")
    parser.add_argument("--etiqueta", action="append", help="nombre de cada corrida en la figura (mismo orden)")
    parser.add_argument("--aciertos", type=int, default=2)
    parser.add_argument("--errores", type=int, default=4)
    parser.add_argument("--por-error", action="store_true", help="una fila por tipo de error (una sola corrida)")
    parser.add_argument("--por-fila", type=int, default=5, help="ejemplos por fila con --por-error")
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    if args.por_error:
        if len(args.corrida) != 1:
            sys.exit("--por-error grafica una sola corrida")
        plot_error_examples(args.dataset, args.corrida[0], args.salida, args.por_fila)
    else:
        plot_hits_and_fps(args.dataset, args.corrida, args.etiqueta or [run.name for run in args.corrida],
                          args.aciertos, args.errores, args.salida)
    print(f"Figura en {args.salida}")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
