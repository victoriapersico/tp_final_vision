"""Genera resultados reproducibles del EDA sin alterar los datasets."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eda_utils import analyze_raw_fabrid, analyze_yolo_segmentation, image_inventory

DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

fabrid_annotations = DATA / "fabrid_v3" / "annotations_export"
raw_root = DATA / "raw_fabrid" / "extracted" / "RAW_FABRID"
fabric_root = next((DATA / "fabric_spot_defect").glob("*/FabricSpotDefect"))

summary = {
    "fabrid_v3": {
        "original_inventory": image_inventory(DATA / "fabrid_v3" / "original"),
        "annotation_export_inventory": image_inventory(fabrid_annotations),
        "annotations": analyze_yolo_segmentation(
            fabrid_annotations,
            ["Needle Mark Defect", "hole Defect", "missing yarn Defect", "stain Defect"],
        ),
    },
    "raw_fabrid": {
        "high_res_inventory": image_inventory(raw_root / "images"),
        "mask_inventory": image_inventory(raw_root / "masks"),
        "analysis": analyze_raw_fabrid(raw_root),
    },
    "fabric_spot_defect": {
        "original_inventory": image_inventory(fabric_root / "Orginal"),
        "augmented_inventory": image_inventory(fabric_root / "Augmented"),
    },
    "rise_samples": image_inventory(DATA / "rise_samples"),
}

(OUT / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
