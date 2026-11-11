"""Evaluación de detecciones de CISUTAC en coordenadas de la imagen original (propuesta de CIS-6).

Cajas reales: el JSON COCO de scripts/prepare_cisutac_yolo.py. Predicciones: formato de resultados
de COCO, [{"image_id": int, "bbox": [x, y, ancho, alto], "score": float}, ...], en px de la original.

mAP@0.5, mAP@0.5:0.95 y AP_small se calculan con pycocotools (implementación de referencia, `coco_ap`).
El resto agrega lo que pycocotools no reporta: precisión/recall/F1 con un umbral de confianza fijo,
recall por rango de tamaño e intervalos por bootstrap de imágenes.
"""

from __future__ import annotations

import contextlib
import io
import random
from collections import defaultdict

from eda_utils import SIZE_EDGES, box_iou

SIZE_LABELS = ("< 8 px", "8–16 px", "16–32 px", "32–64 px", "≥ 64 px")


def xywh_to_xyxy(box):
    return box[0], box[1], box[0] + box[2], box[1] + box[3]


def coco_ap(gt, predictions, max_dets=100):
    """mAP@0.5:0.95, mAP@0.5 y AP por tamaño con pycocotools, en las coordenadas de `gt` (la original).

    Rangos de COCO sobre el `area` de la caja real: small < 32², medium 32²–96², large ≥ 96² px.
    Devuelve None en un rango sin cajas reales. `max_dets` es el máximo de predicciones por imagen.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    keys = ("mAP@0.5:0.95", "mAP@0.5", "mAP@0.75", "AP_small", "AP_medium", "AP_large")
    if not predictions:
        return dict.fromkeys(keys, 0.0)
    truth = COCO()
    truth.dataset = {"info": {}, **gt}
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools imprime índices y su resumen completo
        truth.createIndex()
        detections = truth.loadRes([dict(p, category_id=1) for p in predictions])
        evaluator = COCOeval(truth, detections, "bbox")
        evaluator.params.maxDets = [1, 10, max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {k: (None if v < 0 else float(v)) for k, v in zip(keys, evaluator.stats[:6])}


def match_detections(gt, predictions, iou_threshold=0.5, score_threshold=0.0):
    """Empareja, imagen por imagen, predicciones y cajas reales como COCO.

    Las predicciones se recorren de mayor a menor confianza y cada una toma la caja real libre con
    mayor IoU (si supera `iou_threshold`). Devuelve por imagen: verdaderos positivos, cantidad de
    predicciones y, para cada caja real, si fue encontrada.
    """
    truths, preds = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        truths[a["image_id"]].append(a)
    for p in predictions:
        if p["score"] >= score_threshold:
            preds[p["image_id"]].append(p)
    matched = {}
    for image in gt["images"]:
        image_id = image["id"]
        found = [False] * len(truths[image_id])
        for p in sorted(preds[image_id], key=lambda p: -p["score"]):
            best, best_iou = None, iou_threshold
            for k, a in enumerate(truths[image_id]):
                iou = 0.0 if found[k] else box_iou(xywh_to_xyxy(p["bbox"]), xywh_to_xyxy(a["bbox"]))
                if iou >= best_iou:
                    best, best_iou = k, iou
            if best is not None:
                found[best] = True
        matched[image_id] = {"predicciones": len(preds[image_id]),
                             "verdaderos_positivos": sum(found),
                             "cajas": [(a["area"] ** 0.5, hit) for a, hit in zip(truths[image_id], found)]}
    return matched


def summarize(matched, image_ids=None):
    """Precisión, recall, F1 y recall por rango de tamaño sobre una lista de imágenes (admite repetidas)."""
    image_ids = list(matched) if image_ids is None else image_ids
    tp = sum(matched[i]["verdaderos_positivos"] for i in image_ids)
    n_pred = sum(matched[i]["predicciones"] for i in image_ids)
    boxes = [box for i in image_ids for box in matched[i]["cajas"]]
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / len(boxes) if boxes else 0.0
    result = {"precision": precision, "recall": recall,
              "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    for k, label in enumerate(SIZE_LABELS):
        in_bin = [hit for side, hit in boxes if sum(side >= edge for edge in SIZE_EDGES) == k]
        result[f"recall {label}"] = sum(in_bin) / len(in_bin) if in_bin else None
    return result


def best_f1_threshold(gt, predictions, iou_threshold=0.5, steps=100):
    """Umbral de confianza que maximiza F1. Se elige en valid y después se fija para test."""
    thresholds = [k / steps for k in range(steps)]
    return max(thresholds, key=lambda t: summarize(match_detections(gt, predictions, iou_threshold, t))["f1"])


def bootstrap_interval(matched, metric, n=1000, seed=0, level=0.95):
    """Intervalo percentil de `metric` remuestreando imágenes con reposición."""
    rng = random.Random(seed)
    ids = list(matched)
    values = sorted(v for v in (summarize(matched, rng.choices(ids, k=len(ids)))[metric] for _ in range(n))
                    if v is not None)
    lower = values[int((1 - level) / 2 * len(values))]
    upper = values[min(int((1 + level) / 2 * len(values)), len(values) - 1)]
    return lower, upper
