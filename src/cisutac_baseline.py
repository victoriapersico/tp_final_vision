"""Funciones del bloque A (paradigma supervisado): detección de agujeros y manchas en CISUTAC.

Las usa `notebooks/02_baseline_supervisado_cisutac.ipynb`. Reúne en un solo lugar:
- armado del dataset de tiles para entrenar a la misma escala con la que se infiere;
- entrenamiento con Ultralytics, con caché de corridas (no se reentrena si ya existen los pesos);
- inferencia con la foto completa o con tiles (SAHI) y evaluación en coordenadas de la original
  con `detection_eval.py` (misma métrica para todas las variantes);
- tablas y figuras para comparar corridas.

Requiere los datos de `scripts/prepare_cisutac_yolo.py` en data/derived/. Nada de esto modifica los originales.
Dependencias de entrenamiento: ultralytics, sahi, pycocotools (se importan sólo cuando hacen falta).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from detection_eval import (SIZE_LABELS, best_f1_threshold, bootstrap_interval, coco_ap, match_detections,
                            summarize, xywh_to_xyxy)
from eda_utils import tile_starts

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
RESULTS = ROOT / "results" / "cisutac" / "bloque_a"


# --------------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------------

def derived_dir(dataset):
    return DATA / "derived" / f"cisutac_{dataset}"


def original_dir(dataset):
    return DATA / f"cisutac_{dataset}"


def load_ground_truth(dataset, split):
    """Cajas reales COCO (px de la original) escritas por prepare_cisutac_yolo.py."""
    path = derived_dir(dataset) / f"coco_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}: correr antes python scripts/prepare_cisutac_yolo.py")
    return json.loads(path.read_text())


def pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tile_grid(width, height, tile, overlap):
    """Esquinas superiores izquierdas de los tiles que recorren la imagen (mismo esquema que SAHI)."""
    return [(x, y) for y in tile_starts(height, tile, overlap) for x in tile_starts(width, tile, overlap)]


def boxes_in_tile(boxes, x0, y0, tile, min_visible):
    """Recorta las cajas (xyxy, px de la original) al tile.

    Devuelve (cajas_que_quedan, fragmentos): una caja queda si conserva al menos `min_visible` de su
    área dentro del tile; si conserva algo menos, es un fragmento ambiguo.
    """
    kept, fragments = [], 0
    for x1, y1, x2, y2 in boxes:
        ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + tile), min(y2, y0 + tile)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            continue
        area = (x2 - x1) * (y2 - y1)
        if inter / area >= min_visible:
            kept.append((ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0))
        else:
            fragments += 1
    return kept, fragments


def classify_tiles(gt, tile, overlap, min_visible=0.5):
    """Clasifica cada tile de cada imagen en positivo (tiene defectos), negativo (no toca ninguno) o
    descartado (sólo contiene fragmentos de defectos cortados, que serían negativos falsos)."""
    by_image = {}
    for a in gt["annotations"]:
        by_image.setdefault(a["image_id"], []).append(xywh_to_xyxy(a["bbox"]))
    positives, negatives, discarded = [], [], 0
    for image in gt["images"]:
        boxes = by_image.get(image["id"], [])
        for x0, y0 in tile_grid(image["width"], image["height"], tile, overlap):
            kept, fragments = boxes_in_tile(boxes, x0, y0, tile, min_visible)
            entry = {"image": image, "x0": x0, "y0": y0, "boxes": kept}
            if kept:
                positives.append(entry)
            elif fragments:
                discarded += 1
            else:
                negatives.append(entry)
    return positives, negatives, discarded


def build_tile_dataset(dataset, tile=640, overlap=0.2, neg_per_pos=1.0, seed=0, min_visible=0.5,
                       max_images=None):
    """Arma (o reutiliza) un dataset de tiles para Ultralytics y devuelve (data.yaml, estadísticas).

    - Train y valid salen de los mismos splits que la foto completa (valid sólo se usa para elegir la época).
    - `neg_per_pos`: tiles sin defectos por cada tile con defectos (None = todos). Se muestrean con `seed`.
    - Los recortes se guardan una sola vez en data/derived/cisutac_<dataset>_tiles/t<tile>_o<overlap>/ y
      cada configuración sólo escribe una lista de archivos, así distintas proporciones comparten recortes.
    """
    from PIL import Image

    root = DATA / "derived" / f"cisutac_{dataset}_tiles" / f"t{tile}_o{round(overlap * 100)}"
    tag = f"neg{'todos' if neg_per_pos is None else f'{neg_per_pos:g}'}_s{seed}"
    stats = {}
    lists = {}
    for split in ("train", "valid"):
        gt = load_ground_truth(dataset, split)
        if max_images:
            gt = {**gt, "images": gt["images"][:max_images]}
        positives, negatives, discarded = classify_tiles(gt, tile, overlap, min_visible)
        rng = random.Random(seed)
        if neg_per_pos is None:
            chosen_neg = negatives
        else:
            chosen_neg = rng.sample(negatives, min(len(negatives), round(neg_per_pos * len(positives))))
        selected = positives + chosen_neg
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        paths = []
        pending = {}
        for entry in selected:
            stem = Path(entry["image"]["file_name"]).stem
            name = f"{stem}__x{entry['x0']}_y{entry['y0']}"
            image_path = root / "images" / split / f"{name}.jpg"
            paths.append(image_path)
            rows = []
            for x1, y1, x2, y2 in entry["boxes"]:
                rows.append(f"0 {(x1 + x2) / 2 / tile:.6f} {(y1 + y2) / 2 / tile:.6f} "
                            f"{(x2 - x1) / tile:.6f} {(y2 - y1) / tile:.6f}\n")
            (root / "labels" / split / f"{name}.txt").write_text("".join(rows))
            if not image_path.exists():
                pending.setdefault(entry["image"]["file_name"], []).append((entry["x0"], entry["y0"], image_path))
        for file_name, crops in pending.items():  # cada foto original se abre una sola vez
            with Image.open(original_dir(dataset) / file_name) as im:
                im = im.convert("RGB")
                for x0, y0, image_path in crops:
                    im.crop((x0, y0, x0 + tile, y0 + tile)).save(image_path, quality=95)
        list_path = root / "listas" / f"{split}_{tag}.txt"
        list_path.parent.mkdir(exist_ok=True)
        list_path.write_text("".join(f"{p}\n" for p in sorted(paths)))
        lists[split] = list_path
        stats[split] = {"imagenes_originales": len(gt["images"]), "tiles_posibles": len(positives) + len(negatives)
                        + discarded, "tiles_con_defecto": len(positives), "tiles_sin_defecto_disponibles": len(negatives),
                        "tiles_sin_defecto_usados": len(chosen_neg), "tiles_descartados_por_fragmentos": discarded,
                        "tiles_totales": len(selected), "cajas": sum(len(e["boxes"]) for e in positives)}
    yaml = root / f"data_{tag}.yaml"
    yaml.write_text(f"# Generado por src/cisutac_baseline.py; no editar a mano.\n"
                    f"path: {json.dumps(str(root))}\ntrain: {json.dumps(str(lists['train']))}\n"
                    f"val: {json.dumps(str(lists['valid']))}\nnames:\n  0: defecto\n")
    return yaml, stats


def equivalent_epochs(epochs_full, n_full_images, n_training_images, minimum=10):
    """Épocas para que una corrida vea aproximadamente tantas imágenes como `epochs_full` sobre la foto
    completa. Así las variantes se comparan con el mismo presupuesto de pasos de entrenamiento."""
    return max(minimum, round(epochs_full * n_full_images / n_training_images))


# --------------------------------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------------------------------

def train(name, data_yaml, imgsz, epochs, patience, batch, seed=0, model="yolo11n.pt", retrain=False,
          fraction=1.0, workers=2, **extra):
    """Entrena un YOLO de Ultralytics y devuelve la carpeta de la corrida.

    Si ya existe `weights/best.pt` y `retrain` es False, no vuelve a entrenar: así el notebook se puede
    re-ejecutar para ver resultados sin gastar GPU. La configuración y el tiempo quedan en corrida.json.
    """
    run_dir = RESULTS / "entrenamientos" / name
    if (run_dir / "weights" / "best.pt").exists() and not retrain:
        print(f"{name}: ya entrenado, se reutiliza {run_dir / 'weights' / 'best.pt'}")
        return run_dir
    from ultralytics import YOLO

    weights_dir = ROOT / "results" / "pesos"
    weights_dir.mkdir(parents=True, exist_ok=True)
    # Pesos preentrenados (.pt) se descargan a results/pesos/; un .yaml de arquitectura se pasa tal cual.
    initial = str(weights_dir / model) if model.endswith(".pt") and not Path(model).exists() else model
    start = time.perf_counter()
    YOLO(initial).train(
        data=str(data_yaml), imgsz=imgsz, epochs=epochs, patience=patience, batch=batch, seed=seed,
        deterministic=True, device=pick_device(), workers=workers, fraction=fraction,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True, verbose=False, plots=True, **extra)
    config = {"nombre": name, "data": str(data_yaml), "imgsz": imgsz, "epocas": epochs, "paciencia": patience,
              "batch": batch, "semilla": seed, "modelo": model, "fraction": fraction, **extra,
              "minutos_entrenamiento": round((time.perf_counter() - start) / 60, 1),
              "dispositivo": str(pick_device())}
    (run_dir / "corrida.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    return run_dir


def training_curves(run_dir):
    """results.csv de Ultralytics como DataFrame, con nombres de columnas sin espacios."""
    import pandas as pd

    df = pd.read_csv(Path(run_dir) / "results.csv")
    df.columns = df.columns.str.strip()
    return df


# --------------------------------------------------------------------------------------------------
# Inferencia y evaluación
# --------------------------------------------------------------------------------------------------

def make_predictor(weights, mode, imgsz=640, tile=640, overlap=0.2, also_full=False, conf=0.001):
    """Función path -> [(caja xywh en px de la original, score)] para la foto completa o para tiles."""
    device = pick_device()
    if mode == "completa":
        from ultralytics import YOLO

        model = YOLO(str(weights))

        def predict(path):
            result = model.predict(str(path), imgsz=imgsz, conf=conf, device=device, max_det=300, verbose=False)[0]
            return [([x1, y1, x2 - x1, y2 - y1], float(s))
                    for (x1, y1, x2, y2), s in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
        return predict

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    sahi_device = f"cuda:{device}" if isinstance(device, int) else device
    model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                               confidence_threshold=conf, device=sahi_device, image_size=tile)

    def predict(path):
        # Mismo criterio de fusión que scripts/evaluate_cisutac.py: NMS agnóstico a IoU 0,5.
        result = get_sliced_prediction(str(path), model, slice_height=tile, slice_width=tile,
                                       overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                       perform_standard_pred=also_full, postprocess_type="NMS",
                                       postprocess_match_metric="IOU", postprocess_match_threshold=0.5,
                                       postprocess_class_agnostic=True, verbose=0)
        return [([float(v) for v in p.bbox.to_xywh()], float(p.score.value)) for p in result.object_prediction_list]
    return predict


def evaluate(name, dataset, weights, split="valid", mode="completa", imgsz=640, tile=640, overlap=0.2,
             also_full=False, conf=0.001, threshold=None, limit=None, reevaluate=False, confirm_test=False):
    """Infiere sobre un split y calcula todas las métricas en coordenadas de la original.

    Guarda predicciones.json y metricas.json en results/cisutac/bloque_a/evaluaciones/<name>/ y, si ya
    existen y `reevaluate` es False, los lee. En valid el umbral de confianza es el de máximo F1; en test
    tiene que venir de valid (`threshold`) y hay que pasar `confirm_test=True`.
    """
    out = RESULTS / "evaluaciones" / name
    if (out / "metricas.json").exists() and not reevaluate:
        return json.loads((out / "metricas.json").read_text())
    if split == "test" and (not confirm_test or threshold is None):
        raise ValueError("El test se usa una sola vez: pasar confirm_test=True y el umbral elegido en valid.")
    gt = load_ground_truth(dataset, split)
    if limit:
        gt["images"] = gt["images"][:limit]
        kept = {im["id"] for im in gt["images"]}
        gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in kept]
    predict = make_predictor(weights, mode, imgsz, tile, overlap, also_full, conf)
    folder = original_dir(dataset)
    predict(folder / gt["images"][0]["file_name"])  # calentamiento: no se cuenta en el tiempo
    memory = _reset_peak_memory()
    predictions, seconds = [], []
    for image in gt["images"]:
        start = time.perf_counter()
        boxes = predict(folder / image["file_name"])
        seconds.append(time.perf_counter() - start)
        predictions += [{"image_id": image["id"], "bbox": [round(v, 2) for v in box], "score": round(s, 5)}
                        for box, s in boxes]
    if threshold is None:
        threshold = best_f1_threshold(gt, predictions)
    matched = match_detections(gt, predictions, score_threshold=threshold)
    n_tiles = [len(tile_grid(im["width"], im["height"], tile, overlap)) + also_full for im in gt["images"]] \
        if mode == "tiles" else [1] * len(gt["images"])
    metrics = {
        "corrida": name, "dataset": dataset, "split": split, "modo": mode, "imgsz": imgsz, "tile": tile,
        "solapamiento": overlap, "tambien_completa": also_full, "pesos": str(weights),
        "imagenes": len(gt["images"]), "cajas_reales": len(gt["annotations"]), "predicciones": len(predictions),
        **coco_ap(gt, predictions), "umbral_confianza": threshold, **summarize(matched),
        "ic95_recall": bootstrap_interval(matched, "recall"), "ic95_f1": bootstrap_interval(matched, "f1"),
        "segundos_por_imagen": sum(seconds) / len(seconds), "tiles_por_imagen": sum(n_tiles) / len(n_tiles),
        "memoria_max_gb": memory(),
        "encontrados_por_tamano": _found_by_size(matched),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "predicciones.json").write_text(json.dumps(predictions))
    (out / "metricas.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def _reset_peak_memory():
    """Devuelve una función que informa la memoria máxima de GPU (CUDA) desde ahora, o None."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return lambda: round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    return lambda: None


def _found_by_size(matched):
    """Texto «encontrados/total» por rango de lado equivalente, como en el EDA."""
    from eda_utils import SIZE_EDGES

    counts = {label: [0, 0] for label in SIZE_LABELS}
    for image in matched.values():
        for side, hit in image["cajas"]:
            label = SIZE_LABELS[sum(side >= edge for edge in SIZE_EDGES)]
            counts[label][0] += hit
            counts[label][1] += 1
    return {label: f"{found}/{total}" for label, (found, total) in counts.items()}


# --------------------------------------------------------------------------------------------------
# Tablas y figuras
# --------------------------------------------------------------------------------------------------

MAIN_COLUMNS = ["corrida", "modo", "mAP@0.5", "mAP@0.5:0.95", "AP_small", "AP_medium", "AP_large",
                "umbral_confianza", "precision", "recall", "f1", "segundos_por_imagen", "tiles_por_imagen"]


def metrics_table(results, columns=MAIN_COLUMNS):
    """Tabla con una fila por evaluación (lista de dicts de `evaluate`)."""
    import pandas as pd

    df = pd.DataFrame(results)
    df["recall (IC 95 %)"] = [f"{r:.3f} ({lo:.3f}–{hi:.3f})" for r, (lo, hi) in zip(df["recall"], df["ic95_recall"])]
    return df[[c for c in columns if c in df] + ["recall (IC 95 %)"]].set_index("corrida").round(3)


def size_table(results):
    """Defectos encontrados / total por rango de tamaño, una fila por evaluación."""
    import pandas as pd

    return pd.DataFrame({r["corrida"]: r["encontrados_por_tamano"] for r in results}).T


def plot_curves(run_dirs, metric="metrics/mAP50(B)", ax=None):
    """Curvas de entrenamiento: pérdida de cajas en train y valid, y `metric` en valid."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5)) if ax is None else (None, ax)
    for label, run_dir in run_dirs.items():
        df = training_curves(run_dir)
        axes[0].plot(df["epoch"], df["train/box_loss"], label=label)
        axes[1].plot(df["epoch"], df["val/box_loss"], label=label)
        axes[2].plot(df["epoch"], df[metric], label=label)
    for a, title in zip(axes, ("Pérdida de cajas (train)", "Pérdida de cajas (valid)", f"{metric} (valid de Ultralytics)")):
        a.set_title(title)
        a.set_xlabel("época")
        a.grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_recall_by_size(results, ax=None):
    """Barras de recall por rango de tamaño (con el umbral elegido en valid) para varias evaluaciones."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(SIZE_LABELS))
    width = 0.8 / len(results)
    for k, r in enumerate(results):
        values = [r[f"recall {label}"] or 0 for label in SIZE_LABELS]
        ax.bar(x + k * width - 0.4 + width / 2, values, width, label=r["corrida"])
    totals = [results[0]["encontrados_por_tamano"][label].split("/")[1] for label in SIZE_LABELS]
    ax.set_xticks(x, [f"{label}\n(n={total})" for label, total in zip(SIZE_LABELS, totals)])
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1)
    ax.set_title("Recall por lado equivalente del defecto (px de la original)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_cost_vs_quality(results, metric="AP_small", ax=None):
    """Dispersión costo (segundos por imagen, escala log) vs. calidad para elegir un compromiso."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for r in results:
        ax.scatter(r["segundos_por_imagen"], r[metric] or 0)
        ax.annotate(r["corrida"], (r["segundos_por_imagen"], r[metric] or 0), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("segundos por imagen (log)")
    ax.set_ylabel(metric)
    ax.set_title(f"Costo vs. {metric}")
    ax.grid(alpha=0.3)
    return ax


def plot_resolution_zoom(dataset, split="valid", sides=(640, 1280), n=3, crop=160, seed=0):
    """Cómo ve el modelo un mismo defecto a resolución nativa y con la foto reducida a cada lado mayor.

    Cada columna es un defecto real: arriba el recorte nativo, abajo el mismo recorte después de reducir la
    foto completa (y volver a ampliarlo para mostrarlo al mismo tamaño).
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    gt = load_ground_truth(dataset, split)
    files = {im["id"]: im["file_name"] for im in gt["images"]}
    small = [a for a in gt["annotations"] if a["area"] ** 0.5 < 32]
    chosen = random.Random(seed).sample(small or gt["annotations"], min(n, len(small or gt["annotations"])))
    fig, axes = plt.subplots(1 + len(sides), len(chosen), figsize=(3 * len(chosen), 3 * (1 + len(sides))),
                             squeeze=False)
    for col, a in enumerate(chosen):
        x, y, w, h = a["bbox"]
        box = (x + w / 2 - crop / 2, y + h / 2 - crop / 2, x + w / 2 + crop / 2, y + h / 2 + crop / 2)
        with Image.open(original_dir(dataset) / files[a["image_id"]]) as im:
            im = im.convert("RGB")
            axes[0][col].imshow(im.crop(box))
            axes[0][col].set_title(f"nativa · lado {(w * h) ** 0.5:.0f} px", fontsize=9)
            for row, side in enumerate(sides, start=1):
                scale = side / max(im.size)
                reduced = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
                patch = reduced.crop(tuple(v * scale for v in box)).resize((crop, crop), Image.NEAREST)
                axes[row][col].imshow(patch)
                axes[row][col].set_title(f"lado mayor {side} · lado {(w * h) ** 0.5 * scale:.1f} px", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_tile_samples(data_yaml, n=4, seed=0):
    """Tiles de entrenamiento con sus cajas: fila de arriba con defectos, fila de abajo sin defectos."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    train_list = next(line.split(":", 1)[1].strip() for line in Path(data_yaml).read_text().splitlines()
                      if line.startswith("train:"))
    paths = [Path(p) for p in Path(json.loads(train_list)).read_text().splitlines() if p.strip()]
    labels = {p: (p.parent.parent.parent / "labels" / p.parent.name / f"{p.stem}.txt").read_text() for p in paths}
    rng = random.Random(seed)
    groups = [[p for p in paths if labels[p].strip()], [p for p in paths if not labels[p].strip()]]
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4), squeeze=False)
    for row, (group, title) in enumerate(zip(groups, ("con defecto", "sin defecto"))):
        for col, path in enumerate(rng.sample(group, min(n, len(group)))):
            with Image.open(path) as im:
                axes[row][col].imshow(im)
                size = im.width
            for line in labels[path].splitlines():
                _, xc, yc, w, h = map(float, line.split())
                axes[row][col].add_patch(Rectangle(((xc - w / 2) * size, (yc - h / 2) * size), w * size, h * size,
                                                   fill=False, ec="lime", lw=1.5))
            axes[row][col].set_title(title, fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_examples(dataset, evaluation_name, n=4, split="valid", crop=320, seed=0):
    """Recortes alrededor de defectos reales: caja verde = real, roja punteada = predicción sobre el umbral."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    folder = RESULTS / "evaluaciones" / evaluation_name
    metrics = json.loads((folder / "metricas.json").read_text())
    predictions = [p for p in json.loads((folder / "predicciones.json").read_text())
                   if p["score"] >= metrics["umbral_confianza"]]
    gt = load_ground_truth(dataset, split)
    files = {im["id"]: im["file_name"] for im in gt["images"]}
    annotations = random.Random(seed).sample(gt["annotations"], min(n, len(gt["annotations"])))
    fig, axes = plt.subplots(1, len(annotations), figsize=(3.2 * len(annotations), 3.4))
    for ax, a in zip(axes if len(annotations) > 1 else [axes], annotations):
        x, y, w, h = a["bbox"]
        cx, cy = x + w / 2, y + h / 2
        side = max(crop, 2 * max(w, h))
        x0, y0 = cx - side / 2, cy - side / 2
        with Image.open(original_dir(dataset) / files[a["image_id"]]) as im:
            ax.imshow(im.convert("RGB").crop((x0, y0, x0 + side, y0 + side)))
        for other in gt["annotations"]:
            if other["image_id"] == a["image_id"]:
                bx, by, bw, bh = other["bbox"]
                ax.add_patch(Rectangle((bx - x0, by - y0), bw, bh, fill=False, ec="lime", lw=1.5))
        for p in predictions:
            if p["image_id"] == a["image_id"]:
                bx, by, bw, bh = p["bbox"]
                ax.add_patch(Rectangle((bx - x0, by - y0), bw, bh, fill=False, ec="red", lw=1.2, ls="--"))
        ax.set_xlim(0, side)
        ax.set_ylim(side, 0)
        ax.set_title(f"lado {(w * h) ** 0.5:.0f} px", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{evaluation_name}: verde = real, rojo = predicción (conf ≥ {metrics['umbral_confianza']:.2f})",
                 fontsize=10)
    plt.tight_layout()
    return fig
