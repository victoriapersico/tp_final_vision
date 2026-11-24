"""Evaluación de detecciones de CISUTAC en coordenadas de la imagen original (propuesta de CIS-6).

Cajas reales: el JSON COCO de scripts/prepare_cisutac_yolo.py. Predicciones: formato de resultados
de COCO, [{"image_id": int, "bbox": [x, y, ancho, alto], "score": float}, ...], en px de la original.

mAP@0.5, mAP@0.5:0.95 y AP_small se calculan con pycocotools (implementación de referencia, `coco_ap`).
El resto agrega lo que pycocotools no reporta: precisión/recall/F1 con un umbral de confianza fijo,
recall por rango de tamaño, intervalos por bootstrap de imágenes y la descomposición de errores
(`error_breakdown`, `error_deltas`).
"""

from __future__ import annotations

import contextlib
import io
import random
from collections import Counter, defaultdict

from eda_utils import SIZE_EDGES, box_iou

SIZE_LABELS = ("< 8 px", "8–16 px", "16–32 px", "32–64 px", "≥ 64 px")
# Grupos de lado equivalente fijados en el protocolo (reports/protocolo_experimental.md, sección 4).
PROTOCOL_EDGES = (16, 32)
PROTOCOL_LABELS = ("< 16 px", "16–32 px", "≥ 32 px")


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
    predicciones y, para cada caja real, si fue encontrada. En `detalle` queda una fila por predicción,
    en el orden en que se emparejaron: su índice en `predictions`, su mejor IoU con cualquier caja real
    (aunque ya esté tomada), el índice de esa caja en la imagen y si quedó emparejada (`error_breakdown`).
    """
    truths, preds = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        truths[a["image_id"]].append(a)
    for index, p in enumerate(predictions):
        if p["score"] >= score_threshold:
            preds[p["image_id"]].append((index, p))
    matched = {}
    for image in gt["images"]:
        image_id = image["id"]
        found = [False] * len(truths[image_id])
        detail = []
        for index, p in sorted(preds[image_id], key=lambda item: -item[1]["score"]):
            ious = [box_iou(xywh_to_xyxy(p["bbox"]), xywh_to_xyxy(a["bbox"])) for a in truths[image_id]]
            best, best_iou = None, iou_threshold
            for k, iou in enumerate(ious):
                if not found[k] and iou >= best_iou:
                    best, best_iou = k, iou
            if best is not None:
                found[best] = True
            nearest = max(range(len(ious)), key=ious.__getitem__, default=None)
            detail.append({"indice": index, "iou_max": ious[nearest] if ious else 0.0, "caja_cercana": nearest,
                           "emparejada": best is not None})
        matched[image_id] = {"predicciones": len(preds[image_id]),
                             "verdaderos_positivos": sum(found),
                             "cajas": [(a["area"] ** 0.5, hit) for a, hit in zip(truths[image_id], found)],
                             "detalle": detail}
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


def _recall_by_size(matched, edges, labels):
    boxes = [box for image in matched.values() for box in image["cajas"]]
    result = {}
    for k, label in enumerate(labels):
        in_bin = [hit for side, hit in boxes if sum(side >= edge for edge in edges) == k]
        result[label] = sum(in_bin) / len(in_bin) if in_bin else None
    return result


def prediction_categories(matched, iou_threshold=0.5, background_iou=0.1):
    """Categoría de cada predicción (por su índice) según un emparejamiento de `match_detections`.

    acierto: emparejada a `iou_threshold`; duplicado: IoU ≥ `iou_threshold` con una caja que ya tomó una
    predicción de mayor confianza; localización: mejor IoU entre `background_iou` y `iou_threshold`;
    alucinación: mejor IoU menor a `background_iou` con todas las cajas reales.
    """
    categories = {}
    for image in matched.values():
        for d in image["detalle"]:
            categories[d["indice"]] = ("aciertos" if d["emparejada"] else
                                       "duplicados" if d["iou_max"] >= iou_threshold else
                                       "localizacion" if d["iou_max"] >= background_iou else "alucinaciones")
    return categories


def box_states(strict, loose):
    """Estado de cada caja real (por imagen, en el orden de `gt`), con emparejamientos estricto y laxo.

    detectados: emparejada a IoU estricto (0,5); caja_mala: sólo al laxo (0,1); no_detectados: a ninguno.
    """
    return {image_id: ["detectados" if hit else "caja_mala" if loose_hit else "no_detectados"
                       for (_, hit), (_, loose_hit) in zip(image["cajas"], loose[image_id]["cajas"])]
            for image_id, image in strict.items()}


def error_breakdown(gt, predictions, score_threshold, ious=(0.1, 0.3, 0.5), iou_threshold=0.5,
                    background_iou=0.1, edges=PROTOCOL_EDGES, labels=PROTOCOL_LABELS):
    """Descompone los errores con un umbral de confianza fijo, como TIDE (Bolya et al., ECCV 2020).

    Cada predicción sobre el umbral cae en una sola categoría (`prediction_categories`) y cada caja real,
    por rango de tamaño, en un estado (`box_states`). Además, el recall con emparejamiento uno a uno a cada
    IoU de `ious`: total, por tamaño y con intervalo del 95 %. `alucinaciones_iou0` cuenta las alucinaciones
    que no tocan ninguna caja real.
    """
    runs = {iou: match_detections(gt, predictions, iou, score_threshold)
            for iou in sorted({*ious, iou_threshold, background_iou})}
    strict, loose = runs[iou_threshold], runs[background_iou]
    categories = prediction_categories(strict, iou_threshold, background_iou)
    kinds = Counter(aciertos=0, duplicados=0, localizacion=0, alucinaciones=0)
    kinds.update(categories.values())
    kinds["alucinaciones_iou0"] = sum(categories[d["indice"]] == "alucinaciones" and d["iou_max"] == 0
                                      for image in strict.values() for d in image["detalle"])
    states = {label: Counter(detectados=0, caja_mala=0, no_detectados=0) for label in labels}
    for image_id, image_states in box_states(strict, loose).items():
        for (side, _), state in zip(strict[image_id]["cajas"], image_states):
            states[labels[sum(side >= edge for edge in edges)]][state] += 1
    recall = {iou: {"recall": summarize(runs[iou])["recall"], "ic95": bootstrap_interval(runs[iou], "recall"),
                    "por_tamano": _recall_by_size(runs[iou], edges, labels)} for iou in ious}
    return {"predicciones": dict(kinds), "cajas": {label: dict(c) for label, c in states.items()}, "recall": recall}


def error_deltas(gt, predictions, iou_threshold=0.5, background_iou=0.1, max_dets=100):
    """Cuánto subiría mAP@`iou_threshold` si se corrigiera cada tipo de error (oráculos de TIDE).

    Como el mAP, usa todas las predicciones (sin umbral de confianza), limitadas a las `max_dets` de mayor
    confianza por imagen como hace COCO. Cada corrección se aplica por separado sobre el original:
    - localización: la caja pasa a ser la de su defecto más cercano si nadie lo detectó; si ya estaba
      detectado (o lo corrigió otra predicción de mayor confianza), se elimina porque sería un duplicado;
    - duplicados y alucinaciones: se eliminan;
    - no detectados: se quitan de las cajas reales, porque no hay ninguna predicción que corregir.
    Las mejoras no suman lo que le falta al mAP para llegar a 1: los errores se superponen.
    """
    by_image = defaultdict(list)
    for p in predictions:
        by_image[p["image_id"]].append(p)
    top = [p for preds in by_image.values() for p in sorted(preds, key=lambda p: -p["score"])[:max_dets]]
    truths = defaultdict(list)
    for a in gt["annotations"]:
        truths[a["image_id"]].append(a)
    strict = match_detections(gt, top, iou_threshold)
    loose = match_detections(gt, top, background_iou)
    categories = prediction_categories(strict, iou_threshold, background_iou)
    drop = {"duplicados": set(), "alucinaciones": set(), "localizacion": set()}
    moved = {}
    for image_id, image in strict.items():
        taken = {k for k, (_, hit) in enumerate(image["cajas"]) if hit}
        for d in image["detalle"]:  # de mayor a menor confianza
            kind = categories[d["indice"]]
            if kind in ("duplicados", "alucinaciones"):
                drop[kind].add(d["indice"])
            elif kind == "localizacion" and d["caja_cercana"] in taken:
                drop["localizacion"].add(d["indice"])
            elif kind == "localizacion":
                taken.add(d["caja_cercana"])
                moved[d["indice"]] = truths[image_id][d["caja_cercana"]]["bbox"]
    missed = {id(a) for image_id, image_states in box_states(strict, loose).items()
              for a, state in zip(truths[image_id], image_states) if state == "no_detectados"}
    base = coco_ap(gt, top, max_dets)[f"mAP@{iou_threshold:g}"]
    variants = {
        "localizacion": ([dict(p, bbox=moved.get(k, p["bbox"])) for k, p in enumerate(top)
                          if k not in drop["localizacion"]], gt),
        "duplicados": ([p for k, p in enumerate(top) if k not in drop["duplicados"]], gt),
        "alucinaciones": ([p for k, p in enumerate(top) if k not in drop["alucinaciones"]], gt),
        "no_detectados": (top, dict(gt, annotations=[a for a in gt["annotations"] if id(a) not in missed])),
    }
    result = {f"mAP@{iou_threshold:g}": base}
    for kind, (preds, truth) in variants.items():
        fixed = coco_ap(truth, preds, max_dets)[f"mAP@{iou_threshold:g}"]
        # Sin cajas reales (p. ej. si no se detectó ninguna) el mAP no está definido.
        result[f"delta_{kind}"] = None if fixed is None or base is None else fixed - base
    return result
