"""Diagnóstico de la comparación YOLO11n-seg vs. PatchCore en FabriD v3 (notebook 05).

No entrena ni elige nada: lee los mapas que guardaron los notebooks 04 (valid) y 04b (test) y los desglosa
por sesión de captura, formato de imagen, clase y tamaño de defecto, para entender por qué PatchCore rinde
mucho peor en FabriD que en RAW-FABRID. Las funciones reciben los `items` de `fabrid_comparison.load_items`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

MODELOS = ("YOLO11n-seg", "PatchCore")
CLASES = ("Needle Mark", "hole", "missing yarn", "stain")  # orden de los ids del export (data.yaml del 04)
# El EDA (FAB-2) indica que las normales de mayo están fechadas 20260505 y los defectos equivalentes 20260506:
# fotos de la misma tanda tomadas a la noche y a la mañana siguiente. Se agrupan como una sola sesión.
SESIONES = {"20260505": "mayo 05-06", "20260506": "mayo 05-06", "20260714": "julio 14",
            "20260722": "julio 22", "20260803": "agosto 03"}
TAMANOS_BORDES = (0.1, 0.5, 2.0)  # % del área de la imagen
TAMANOS_ETIQUETAS = ("< 0,1 %", "0,1–0,5 %", "0,5–2 %", "≥ 2 %")


def sesion(source_id):
    match = re.search(r"20\d{6}", source_id)
    return SESIONES.get(match.group(0), match.group(0)) if match else "sin fecha"


def formato(ancho, alto):
    """Cuadrada (casi todas vienen de originales 512×512), horizontal o vertical."""
    if abs(ancho / alto - 1) < 0.02:
        return "cuadrada"
    return "horizontal" if ancho > alto else "vertical"


def tamano(area_pct):
    return TAMANOS_ETIQUETAS[int(np.searchsorted(TAMANOS_BORDES, area_pct, side="right"))]


def class_masks(item):
    """Una máscara por clase, rasterizada igual que `fabrid_comparison.binary_mask` (su unión es la máscara común)."""
    with Image.open(item["path"]) as im:
        w, h = im.size
    canvases = {c: Image.new("L", (w, h), 0) for c in CLASES}
    for line in item["label_path"].read_text().splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        coords = values[1:]
        points = [(min(w - 1, x * w), min(h - 1, y * h)) for x, y in zip(coords[::2], coords[1::2])]
        ImageDraw.Draw(canvases[CLASES[int(values[0])]]).polygon(points, fill=1)
    return {c: np.asarray(m, dtype=bool) for c, m in canvases.items()}


class Mapas:
    """Mapas guardados por el 04/04b (`mapas_<split>.npz` + `orden_imagenes.json`), leídos de a uno."""

    def __init__(self, folder, split):
        folder = Path(folder)
        self.order = json.loads((folder / "orden_imagenes.json").read_text())
        self.npz = np.load(folder / f"mapas_{split}.npz")
        self.index = {image_id: k for k, image_id in enumerate(self.order)}

    def __getitem__(self, image_id):
        return self.npz[f"map_{self.index[image_id]:04d}"]


def build_tables(items, mapas, umbrales):
    """Una fila por imagen y una por región defectuosa (componente conexa, conectividad 4 como en el AUPRO).

    `mapas[split][modelo]` es un `Mapas`. Por imagen: sesión, formato, clases, área de defecto y puntaje (máximo
    del mapa) de cada modelo. Por región: clase mayoritaria, área y fracción de sus píxeles que cada modelo marca
    con su umbral de valid (`cubierta_<modelo>`).
    """
    from scipy.ndimage import label

    imagenes, regiones = [], []
    for item in items:
        masks = class_masks(item)
        union = np.logical_or.reduce(list(masks.values()))
        h, w = union.shape
        maps = {m: mapas[item["split"]][m][item["id"]] for m in MODELOS}
        for m, score in maps.items():
            if score.shape != union.shape:
                raise ValueError(f"El mapa de {m} no tiene la geometría del export: {item['id']}")
        row = {"id": item["id"], "source_id": item["source_id"], "split": item["split"], "label": item["label"],
               "sesion": sesion(item["source_id"]), "formato": formato(w, h), "geometria": f"{w}x{h}",
               "clases": "; ".join(c for c in CLASES if masks[c].any()),
               "area_defecto_pct": 100 * union.mean()}
        row.update({f"score_{m}": float(maps[m].max()) for m in MODELOS})
        imagenes.append(row)
        components, n = label(union)
        for k in range(1, n + 1):
            region = components == k
            area = int(region.sum())
            clase = max(CLASES, key=lambda c: int((masks[c] & region).sum()))
            r = {"id": item["id"], "source_id": item["source_id"], "split": item["split"], "sesion": row["sesion"],
                 "formato": row["formato"], "clase": clase, "area_px": area, "area_pct": 100 * area / union.size}
            for m in MODELOS:
                r[f"cubierta_{m}"] = float((maps[m][region] >= umbrales[m]).mean())
                r[f"max_{m}"] = float(maps[m][region].max())
            regiones.append(r)
    imagenes, regiones = pd.DataFrame(imagenes), pd.DataFrame(regiones)
    imagenes["tamano"] = pd.Categorical(
        [tamano(a) if l else "normal" for a, l in zip(imagenes.area_defecto_pct, imagenes.label)],
        categories=("normal",) + TAMANOS_ETIQUETAS, ordered=True)
    regiones["tamano"] = pd.Categorical(regiones.area_pct.map(tamano), categories=TAMANOS_ETIQUETAS, ordered=True)
    return imagenes, regiones


def auroc_bootstrap(df, score, n=1000, seed=0):
    """AUROC imagen e IC 95 % por bootstrap de `source_id` (los derivados de una captura no son independientes)."""
    from sklearn.metrics import roc_auc_score

    y = df.label.to_numpy()
    if len(np.unique(y)) < 2:
        return np.nan, (np.nan, np.nan)
    s = df[score].to_numpy()
    point = float(roc_auc_score(y, s))
    if not n:
        return point, (np.nan, np.nan)
    groups = df.source_id.to_numpy()
    unique = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        ix = np.concatenate([rows[g] for g in rng.choice(unique, len(unique))])
        if len(np.unique(y[ix])) == 2:
            draws.append(roc_auc_score(y[ix], s[ix]))
    return point, tuple(np.percentile(draws, [2.5, 97.5])) if draws else (np.nan, np.nan)


def auroc_table(df, subsets, n=1000, seed=0):
    """AUROC imagen de cada modelo en cada subconjunto. `subsets`: {nombre: máscara booleana sobre `df`}."""
    rows = []
    for name, mask in subsets.items():
        part = df[mask]
        row = {"subconjunto": name, "normales": int((part.label == 0).sum()), "defectuosas": int(part.label.sum())}
        for m in MODELOS:
            point, (lo, hi) = auroc_bootstrap(part, f"score_{m}", n, seed)
            row[f"AUROC {m}"] = point
            row[f"IC95 {m}"] = "—" if np.isnan(lo) else f"{lo:.2f}–{hi:.2f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("subconjunto")


def cross_auroc(df, score):
    """AUROC de las defectuosas de cada sesión (columnas) contra las normales de cada sesión (filas).

    Si PatchCore separara por el defecto, todas las celdas serían parecidas. Si una fila es baja, las normales de
    esa sesión puntúan alto (tela "rara" para el banco); si una columna es alta, esa sesión se distingue por la
    captura, no necesariamente por el defecto.
    """
    from sklearn.metrics import roc_auc_score

    normal, defect = df[df.label == 0], df[df.label == 1]
    table = pd.DataFrame(index=sorted(normal.sesion.unique()), columns=sorted(defect.sesion.unique()), dtype=float)
    for a in table.index:
        for b in table.columns:
            n, d = normal[normal.sesion == a][score], defect[defect.sesion == b][score]
            table.loc[a, b] = roc_auc_score(np.r_[np.zeros(len(n)), np.ones(len(d))], np.r_[n, d])
    table.index.name, table.columns.name = "normales de", "defectuosas de"
    return table


def coverage_table(regiones, by):
    """Fracción media de cada región que cada modelo marca, y % de regiones que no toca en absoluto."""
    rows = []
    for key, part in regiones.groupby(by, observed=True):
        row = {by: key, "regiones": len(part)}
        for m in MODELOS:
            row[f"cubierta media {m}"] = part[f"cubierta_{m}"].mean()
            row[f"% sin tocar {m}"] = 100 * (part[f"cubierta_{m}"] == 0).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index(by)


def overlay(ax, item, score, mask=None, title="", vmax=None):
    """Imagen del export con el mapa de anomalía encima y el contorno del defecto real en verde."""
    with Image.open(item["path"]) as im:
        ax.imshow(im.convert("RGB"))
    ax.imshow(score, cmap="inferno", alpha=0.45, vmin=float(score.min()), vmax=vmax or float(score.max()))
    if mask is not None and mask.any():
        ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.8)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
