"""Entrena el baseline supervisado de CISUTAC con Ultralytics: foto completa con lado mayor 640.

Usa los splits de data/derived (scripts/prepare_cisutac_yolo.py). Valid sólo elige el mejor epoch; el test
no se usa. Los resultados quedan en results/cisutac/entrenamientos/ (no versionado).

Requisitos: pip install ultralytics
Ejemplo: python scripts/train_cisutac.py --dataset holes
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()

parser = argparse.ArgumentParser(description="Entrena un detector YOLO en CISUTAC Holes o Spots.")
parser.add_argument("--dataset", choices=("holes", "spots"), required=True)
parser.add_argument("--modelo", default=str(ROOT / "results" / "pesos" / "yolo11n.pt"),
                    help="pesos iniciales preentrenados en COCO (se descargan si no existen)")
parser.add_argument("--epocas", type=int, default=100)
parser.add_argument("--paciencia", type=int, default=30, help="epochs sin mejora en valid antes de cortar")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--semilla", type=int, default=0)
parser.add_argument("--nombre", help="nombre de la corrida (por defecto, dataset_modelo_imgsz)")
args = parser.parse_args()

data_yaml = DATA / "derived" / f"cisutac_{args.dataset}" / "data.yaml"
if not data_yaml.exists():
    sys.exit(f"Falta {data_yaml}: correr antes python scripts/prepare_cisutac_yolo.py")

import torch  # noqa: E402  (se importa después de validar argumentos: tarda en cargar)
from ultralytics import YOLO  # noqa: E402

device = "mps" if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else "cpu"
model = YOLO(args.modelo)
model.train(data=str(data_yaml), epochs=args.epocas, patience=args.paciencia, imgsz=args.imgsz, batch=args.batch,
            seed=args.semilla, deterministic=True, device=device, cache="ram", workers=4,
            project=str(ROOT / "results" / "cisutac" / "entrenamientos"),
            name=args.nombre or f"{args.dataset}_{Path(args.modelo).stem}_{args.imgsz}", exist_ok=True)
