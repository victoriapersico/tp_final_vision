"""Controles adicionales del análisis de errores de detección en CISUTAC. Sólo valid.

Complementa a scripts/analyze_detection_errors.py con lo que sus CSV no guardan. Lee las mismas carpetas de
corridas (predicciones.json + metricas.json) y usa las funciones de src/detection_eval.py:

- sin umbral, con las 100 predicciones más confiables por foto (las mismas que usa el mAP): recall a IoU 0,1,
  0,3 y 0,5, estado de cada defecto y cuántos de los no detectados con el umbral sí tienen una predicción cerca;
- el Δ de localización de error_deltas partido en sus dos correcciones, aplicadas por separado: quitar las
  cajas de más sobre defectos ya acertados y mover a su defecto la caja de los que sólo se encontraron con
  una caja mala (cuántas son, su confianza mediana y cuántas superan el umbral);
- con el umbral de valid: confianza mediana de aciertos y alucinaciones, qué son las 20 predicciones más
  confiables y, de los errores de localización, cuántos tocan un borde de tile, cuántos caen sobre un defecto
  ya acertado (en cuántos de esos la caja es más grande o tiene más confianza que el acierto) y su área
  relativa al defecto.

Las corridas de test se saltean sin leer sus predicciones. Escribe errores_controles.csv en --salida (por
defecto results/analisis_errores/, como analyze_detection_errors.py).

Ejemplo:
  python scripts/detection_error_checks.py --carpeta reports/cisutac/predicciones_bloque_a \
      --patron "*/*__valid_*" --salida reports/cisutac/errores
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_detection_errors import load_run, write_csv
from cisutac_baseline import tile_grid
from detection_eval import box_states, coco_ap, match_detections, prediction_categories, summarize, xywh_to_xyxy
from eda_utils import box_iou

IOUS = (0.1, 0.3, 0.5)
KINDS = ("aciertos", "localizacion", "duplicados", "alucinaciones")
BORDER_PX = 3  # una caja toca un borde de tile si alguno de sus lados queda a esta distancia o menos


def top_per_image(predictions, max_dets=100):
    """Las `max_dets` predicciones de mayor confianza de cada foto, como las usa el mAP de COCO."""
    by_image = defaultdict(list)
    for p in predictions:
        by_image[p["image_id"]].append(p)
    return [p for preds in by_image.values() for p in sorted(preds, key=lambda p: -p["score"])[:max_dets]]


def area(box):
    return box[2] * box[3]


def without_threshold(gt, predictions, threshold):
    """Recall y estado de los defectos con las 100 predicciones más confiables por foto, sin umbral."""
    top = top_per_image(predictions)
    runs = {iou: match_detections(gt, top, iou) for iou in IOUS}
    row = {f"top100_recall_iou{iou:g}": summarize(runs[iou])["recall"] for iou in IOUS}
    states = box_states(runs[0.5], runs[0.1])
    flat = [s for image_states in states.values() for s in image_states]
    row.update({f"top100_{s}": flat.count(s) for s in ("detectados", "caja_mala", "no_detectados")})
    at_threshold = box_states(match_detections(gt, predictions, 0.5, threshold),
                              match_detections(gt, predictions, 0.1, threshold))
    row["no_detectados_umbral_con_prediccion_cerca"] = sum(
        before == "no_detectados" and after != "no_detectados"
        for image_id in states for before, after in zip(at_threshold[image_id], states[image_id]))
    return row


def localization_split(gt, predictions, threshold):
    """El Δ de localización de error_deltas, con sus dos correcciones aplicadas por separado (misma lógica)."""
    top = top_per_image(predictions)
    truths = defaultdict(list)
    for a in gt["annotations"]:
        truths[a["image_id"]].append(a)
    strict = match_detections(gt, top, 0.5)
    categories = prediction_categories(strict)
    extra, moved = set(), {}
    for image_id, image in strict.items():
        taken = {k for k, (_, hit) in enumerate(image["cajas"]) if hit}
        for d in image["detalle"]:  # de mayor a menor confianza
            if categories[d["indice"]] != "localizacion":
                continue
            if d["caja_cercana"] in taken:
                extra.add(d["indice"])
            else:
                taken.add(d["caja_cercana"])
                moved[d["indice"]] = truths[image_id][d["caja_cercana"]]["bbox"]
    base = coco_ap(gt, top)["mAP@0.5"]
    removed = coco_ap(gt, [p for k, p in enumerate(top) if k not in extra])["mAP@0.5"]
    fixed = coco_ap(gt, [dict(p, bbox=moved.get(k, p["bbox"])) for k, p in enumerate(top)])["mAP@0.5"]
    scores = [top[k]["score"] for k in moved]
    return {"delta_loc_quitar_extra": removed - base, "cajas_extra": len(extra),
            "delta_loc_mover": fixed - base, "cajas_movidas": len(moved),
            "conf_mediana_movidas": median(scores) if scores else None,
            "movidas_sobre_umbral": sum(s >= threshold for s in scores)}


def touches_tile_border(box, grid, tile, width, height):
    """Si la caja (xyxy) tiene un lado sobre un borde interno de algún tile de la grilla de SAHI."""
    xs = {e for x, _ in grid for e in (x, x + tile) if 0 < e < width}
    ys = {e for _, y in grid for e in (y, y + tile) if 0 < e < height}
    x1, y1, x2, y2 = box
    return (any(abs(v - e) <= BORDER_PX for v in (x1, x2) for e in xs) or
            any(abs(v - e) <= BORDER_PX for v in (y1, y2) for e in ys))


def at_threshold(gt, predictions, threshold, metrics):
    """Confianzas, las 20 predicciones más confiables y detalle de los errores de localización, con umbral."""
    strict = match_detections(gt, predictions, 0.5, threshold)
    categories = prediction_categories(strict)
    ranked = sorted(categories, key=lambda k: -predictions[k]["score"])
    row = {}
    for kind in ("aciertos", "alucinaciones"):
        scores = [predictions[k]["score"] for k in ranked if categories[k] == kind]
        row[f"conf_mediana_{kind}"] = median(scores) if scores else None
    top20 = [categories[k] for k in ranked[:20]]
    row.update({f"top20_{kind}": top20.count(kind) for kind in KINDS})

    truths = defaultdict(list)
    for a in gt["annotations"]:
        truths[a["image_id"]].append(a["bbox"])
    sizes = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    tiles = metrics["modo"] == "tiles"
    loc, border, on_hit, bigger, more_confident, ratios = 0, 0, 0, 0, 0, []
    for image_id, image in strict.items():
        hits = [d["indice"] for d in image["detalle"] if d["emparejada"]]
        hit_boxes = {k for k, (_, hit) in enumerate(image["cajas"]) if hit}
        grid = tile_grid(*sizes[image_id], metrics["tile"], metrics["solapamiento"]) if tiles else []
        for d in image["detalle"]:
            if categories[d["indice"]] != "localizacion":
                continue
            p, truth = predictions[d["indice"]], truths[image_id][d["caja_cercana"]]
            loc += 1
            ratios.append(area(p["bbox"]) / area(truth))
            border += tiles and touches_tile_border(xywh_to_xyxy(p["bbox"]), grid, metrics["tile"], *sizes[image_id])
            if d["caja_cercana"] in hit_boxes:
                # El acierto de ese defecto: la predicción emparejada con mayor IoU con él.
                hit = predictions[max(hits, key=lambda k: box_iou(xywh_to_xyxy(predictions[k]["bbox"]),
                                                                  xywh_to_xyxy(truth)))]
                on_hit += 1
                bigger += area(p["bbox"]) > area(hit["bbox"])
                more_confident += p["score"] > hit["score"]
    row.update({"localizacion_umbral": loc, "localizacion_borde_tile": border if tiles else None,
                "localizacion_sobre_acierto": on_hit, "sobre_acierto_mas_grande": bigger,
                "sobre_acierto_mas_confianza": more_confident,
                "localizacion_area_mediana": median(ratios) if ratios else None})
    return row


def main():
    parser = argparse.ArgumentParser(description="Controles adicionales del análisis de errores (sólo valid).")
    parser.add_argument("--carpeta", type=Path, required=True,
                        help="carpeta con una subcarpeta por corrida (predicciones.json + metricas.json)")
    parser.add_argument("--patron", default="*", help="patrón de las subcarpetas a analizar, p. ej. '*/*__valid_*'")
    parser.add_argument("--salida", type=Path, default=ROOT / "results" / "analisis_errores")
    args = parser.parse_args()

    folders = sorted(f for f in args.carpeta.glob(args.patron)
                     if (f / "metricas.json").exists() and (f / "predicciones.json").exists())
    rows = []
    for folder in folders:
        run = load_run(folder)
        if run is None:
            print(f"  {folder.name}: no es de valid; se saltea sin leer sus predicciones")
            continue
        gt, predictions, metrics = run["gt"], run["predictions"], run["metrics"]
        config = metrics.get("configuracion", metrics)
        if len(gt["annotations"]) != metrics["cajas_reales"]:
            print(f"  {folder.name}: las cajas reales no coinciden con las de su evaluación; se saltea")
            continue
        print(f"  {folder.name}: analizando")
        threshold = metrics["umbral_confianza"]
        rows.append({"dataset": run["dataset"], "corrida": run["corrida"], "umbral": threshold,
                     "defectos": len(gt["annotations"]), **without_threshold(gt, predictions, threshold),
                     **localization_split(gt, predictions, threshold),
                     **at_threshold(gt, predictions, threshold, config)})
    if not rows:
        sys.exit(f"No hay corridas de valid en {args.carpeta} con el patrón {args.patron!r}")

    args.salida.mkdir(parents=True, exist_ok=True)
    write_csv(args.salida / "errores_controles.csv", rows)

    import pandas as pd

    pd.set_option("display.width", 200, "display.max_rows", 100)
    print("\n", pd.DataFrame(rows).set_index("corrida").drop(columns="dataset").T.round(3).to_string())
    print(f"\nResultados en {args.salida / 'errores_controles.csv'}")


if __name__ == "__main__":
    main()
