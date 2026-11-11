"""Inferencia y evaluación de un detector sobre CISUTAC: foto completa redimensionada o tiles con SAHI.

No entrena. Sirve para evaluar un modelo ya entrenado o, con pesos preentrenados en COCO y --agnostico,
para probar el pipeline y medir tiempos. Las predicciones quedan en coordenadas de la imagen original y
se evalúan con src/detection_eval.py (pycocotools + recall por tamaño).

Requisitos: pip install ultralytics sahi pycocotools, y antes: python scripts/prepare_cisutac_yolo.py

Ejemplos:
  python scripts/evaluate_cisutac.py --dataset holes --pesos best.pt --modo completa
  python scripts/evaluate_cisutac.py --dataset spots --pesos best.pt --modo tiles --tile 640 --solapamiento 0.2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection_eval import best_f1_threshold, bootstrap_interval, coco_ap, match_detections, summarize
from eda_utils import tile_starts

DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()

parser = argparse.ArgumentParser(description="Evalúa un detector en CISUTAC (foto completa o tiles).")
parser.add_argument("--dataset", choices=("holes", "spots"), required=True)
parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
parser.add_argument("--pesos", required=True, help="pesos de Ultralytics (.pt)")
parser.add_argument("--modo", choices=("completa", "tiles"), required=True)
parser.add_argument("--imgsz", type=int, default=640, help="lado mayor de la entrada en modo completa")
parser.add_argument("--tile", type=int, default=640, help="lado del tile en modo tiles (entra al modelo sin reescalar)")
parser.add_argument("--solapamiento", type=float, default=0.2)
parser.add_argument("--tambien-completa", action="store_true",
                    help="en modo tiles, sumar la predicción de la foto completa (perform_standard_pred de SAHI)")
parser.add_argument("--conf", type=float, default=0.001, help="confianza mínima para guardar predicciones (para mAP)")
parser.add_argument("--umbral", type=float, help="umbral para precisión/recall/F1; si falta, se elige el de máximo F1")
parser.add_argument("--limite", type=int, help="evaluar sólo las primeras N imágenes (pruebas)")
parser.add_argument("--agnostico", action="store_true",
                    help="tomar todas las clases del modelo como defecto (sólo para probar con pesos de COCO)")
parser.add_argument("--confirmar-test", action="store_true", help="obligatorio para usar el split test")
parser.add_argument("--salida", type=Path, help="carpeta de resultados (por defecto, results/cisutac/...)")
args = parser.parse_args()

if args.split == "test" and not args.confirmar_test:
    sys.exit("El test se usa una sola vez al final: agregar --confirmar-test y un --umbral elegido en valid.")
if args.split == "test" and args.umbral is None:
    sys.exit("En test el umbral de confianza tiene que venir de valid (--umbral).")

import torch  # noqa: E402  (se importa después de validar argumentos: tarda en cargar)
from ultralytics import YOLO  # noqa: E402

device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
folder = f"cisutac_{args.dataset}"
gt = json.loads((DATA / "derived" / folder / f"coco_{args.split}.json").read_text())
if args.limite:
    gt["images"] = gt["images"][:args.limite]
    kept = {im["id"] for im in gt["images"]}
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in kept]

model = YOLO(args.pesos)
if len(model.names) > 1 and not args.agnostico:
    sys.exit(f"El modelo tiene {len(model.names)} clases; para probar el pipeline con pesos de COCO usar --agnostico.")

if args.modo == "completa":
    def predict(path):
        result = model.predict(str(path), imgsz=args.imgsz, conf=args.conf, device=device, max_det=300,
                               verbose=False)[0]  # Ultralytics devuelve las cajas en px de la original
        return [([x1, y1, x2 - x1, y2 - y1], float(score))
                for (x1, y1, x2, y2), score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
else:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    sliced_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=args.pesos,
                                                      confidence_threshold=args.conf, device=device,
                                                      image_size=args.tile)

    def predict(path):
        # Fusión de detecciones de tiles vecinos: NMS con IoU 0,5, fijada explícitamente (con confianza
        # baja SAHI la cambia a esto por su cuenta) para que no dependa de --conf.
        result = get_sliced_prediction(str(path), sliced_model, slice_height=args.tile, slice_width=args.tile,
                                       overlap_height_ratio=args.solapamiento, overlap_width_ratio=args.solapamiento,
                                       perform_standard_pred=args.tambien_completa, postprocess_type="NMS",
                                       postprocess_match_metric="IOU", postprocess_match_threshold=0.5,
                                       postprocess_class_agnostic=True, verbose=0)
        return [([float(v) for v in p.bbox.to_xywh()], float(p.score.value)) for p in result.object_prediction_list]

predictions, seconds = [], []
predict(DATA / folder / gt["images"][0]["file_name"])  # calentamiento: no se cuenta en el tiempo
for image in gt["images"]:
    start = time.perf_counter()
    boxes = predict(DATA / folder / image["file_name"])
    seconds.append(time.perf_counter() - start)
    predictions += [{"image_id": image["id"], "bbox": [round(v, 2) for v in box], "score": round(score, 5)}
                    for box, score in boxes]

threshold = args.umbral if args.umbral is not None else best_f1_threshold(gt, predictions)
matched = match_detections(gt, predictions, score_threshold=threshold)
tiles = [len(tile_starts(im["width"], args.tile, args.solapamiento)) * len(tile_starts(im["height"], args.tile, args.solapamiento))
         for im in gt["images"]] if args.modo == "tiles" else [1] * len(gt["images"])
metrics = {
    "configuracion": {k: str(v) for k, v in vars(args).items()} | {"dispositivo": device},
    "imagenes": len(gt["images"]), "cajas_reales": len(gt["annotations"]), "predicciones": len(predictions),
    **coco_ap(gt, predictions),
    "umbral_confianza": threshold, **summarize(matched),
    # Intervalos del 95 % remuestreando imágenes (1000 réplicas, semilla 0).
    "ic95_recall": bootstrap_interval(matched, "recall"), "ic95_f1": bootstrap_interval(matched, "f1"),
    "segundos_por_imagen": sum(seconds) / len(seconds), "tiles_por_imagen": sum(tiles) / len(tiles),
}

name = f"{args.dataset}_{args.split}_{args.modo}" + (f"{args.tile}_{args.solapamiento}" if args.modo == "tiles" else
                                                     f"{args.imgsz}") + f"_{Path(args.pesos).stem}"
out = args.salida or ROOT / "results" / "cisutac" / name
out.mkdir(parents=True, exist_ok=True)
(out / "predicciones.json").write_text(json.dumps(predictions))
(out / "metricas.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in metrics.items() if k != "configuracion"}, ensure_ascii=False, indent=2))
print(f"Resultados en {out}")
