"""Funciones del bloque B (paradigma de detección de anomalías) sobre los patches de RAW-FABRID.

Las usa `notebooks/03_anomalias_raw_fabrid.ipynb`. Tres métodos que se entrenan sólo con tela normal:

- **B0 · Estadística local (clásico):** media y desvío local con filtros de caja, comparados con los de la tela
  normal. Sirve de piso para medir cuánto aportan las características aprendidas.
- **B1 · PaDiM** (Defard et al., 2021, arXiv:2011.08785): una gaussiana por posición del mapa de características
  de una CNN preentrenada; la anomalía es la distancia de Mahalanobis.
- **B2 · PatchCore** (Roth et al., CVPR 2022, arXiv:2106.08265): banco de memoria de características locales
  normales, reducido con un *coreset* greedy; la anomalía es la distancia al vecino más cercano.

PaDiM y PatchCore se implementan acá en forma compacta siguiendo los papers (backbones de torchvision
preentrenados en ImageNet), para poder explicar cada paso en el informe y correrlos en CPU.

Métricas: AUROC por patch, AUROC por píxel y AUPRO (Bergmann et al., 2021) con mapas a `MAP_SIZE`, más
intervalos por bootstrap agrupado por imagen madre (`parent_raw`) y resultados por rollo, como fijó el EDA.
Nada de esto modifica los datos originales.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
RAW = DATA / "raw_fabrid" / "extracted" / "RAW_FABRID"
MVTEC = RAW / "MVTec"
RESULTS = ROOT / "results" / "raw_fabrid" / "bloque_b"
SPLIT_MANIFEST = ROOT / "reports" / "raw_fabrid" / "rawB_split_valid_test.csv"
PATCH_SIZE = 256   # todos los patches de RAW-FABRID son 256×256
MAP_SIZE = 128     # resolución a la que se guardan mapas y máscaras para las métricas por píxel
FOLDERS = {"train_good": "train/good", "test_good": "test/good", "test_defect": "test/defect"}


# --------------------------------------------------------------------------------------------------
# Datos y particiones
# --------------------------------------------------------------------------------------------------

def load_patches():
    """Una fila por patch del CSV oficial, con ruta, máscara, madre, rollo y etiqueta (1 = defecto)."""
    with (RAW / "RAW_FABRID_Patches_Metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    items = []
    for row in rows:
        kind = row["type"]
        items.append({"name": row["patch_name"], "type": kind, "parent": row["parent_raw"],
                      "roll": row["parent_raw"].split("_")[0], "label": int(kind == "test_defect"),
                      "path": MVTEC / FOLDERS[kind] / row["patch_name"],
                      "mask": MVTEC / "ground_truth" / "defect" / row["patch_name"] if kind == "test_defect" else None})
    return items


def split_test_parents(items, valid_fraction=0.3, seed=2026, manifest=SPLIT_MANIFEST):
    """Parte las 202 madres de test en valid (elegir variantes y umbrales) y test (reporte final).

    El split se hace por madre (`parent_raw`) y estratificado por rollo, así ningún patch de una misma captura
    queda en los dos lados. Si el manifiesto ya existe se lee tal cual: el split no cambia entre corridas.
    """
    if not manifest.exists():
        by_roll = {}
        for parent, roll in sorted({(i["parent"], i["roll"]) for i in items if i["type"] != "train_good"}):
            by_roll.setdefault(roll, []).append(parent)
        rng = random.Random(seed)
        rows = []
        for roll, parents in sorted(by_roll.items()):
            rng.shuffle(parents)
            n_valid = round(valid_fraction * len(parents))
            rows += [{"parent_raw": p, "roll_id": roll, "split": "valid" if k < n_valid else "test"}
                     for k, p in enumerate(parents)]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["parent_raw", "roll_id", "split"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["parent_raw"]))
    with manifest.open(newline="") as handle:
        split_of = {row["parent_raw"]: row["split"] for row in csv.DictReader(handle)}
    splits = {"train": [i for i in items if i["type"] == "train_good"], "valid": [], "test": []}
    for item in items:
        if item["type"] != "train_good":
            splits[split_of[item["parent"]]].append(item)
    return splits


def sample_items(items, n, seed=0):
    """Muestra fija de `n` items (None = todos), para limitar el costo en CPU."""
    if n is None or n >= len(items):
        return list(items)
    return random.Random(seed).sample(list(items), n)


def stratified_subset(items, n_good, n_defect, seed=0):
    """Subconjunto con ambas clases (para el modo prueba)."""
    good = [i for i in items if i["label"] == 0]
    defect = [i for i in items if i["label"] == 1]
    return sample_items(good, n_good, seed) + sample_items(defect, n_defect, seed)


def load_image(path, size=PATCH_SIZE):
    """Patch como uint8 RGB (H, W, 3). Los PNG traen tres canales iguales (RAW-1); se conservan para la CNN."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im)


def load_mask(item, size=MAP_SIZE):
    """Máscara binaria a `size` (los patches normales tienen máscara vacía)."""
    from PIL import Image

    if item["mask"] is None:
        return np.zeros((size, size), bool)
    with Image.open(item["mask"]) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.NEAREST)) > 0


def batches(items, size, batch_size=32):
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        yield chunk, np.stack([load_image(i["path"], size) for i in chunk])


def _resize_maps(maps, size=MAP_SIZE):
    """Mapas (B, H, W) a (B, size, size) con interpolación bilineal."""
    import cv2

    return np.stack([cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR) for m in maps])


def _smooth(maps, sigma):
    from scipy.ndimage import gaussian_filter

    return np.stack([gaussian_filter(m, sigma) for m in maps]) if sigma else maps


# --------------------------------------------------------------------------------------------------
# B0 · Estadística local (procesamiento clásico)
# --------------------------------------------------------------------------------------------------

class LocalStatistics:
    """Para cada píxel, media y desvío de intensidad en una ventana (filtro de caja). En tela normal esos
    valores se mantienen en un rango; un agujero o una mancha cambia el brillo o la textura local.
    El mapa es el mayor de los dos desvíos estandarizados respecto de la tela normal, suavizado."""

    def __init__(self, window=15, sigma=4, size=PATCH_SIZE):
        self.window, self.sigma, self.size = window, sigma, size
        self.state = {}

    def _features(self, images):
        from scipy.ndimage import uniform_filter

        gray = images.mean(axis=-1).astype(np.float32) / 255
        mean = np.stack([uniform_filter(g, self.window) for g in gray])
        square = np.stack([uniform_filter(g * g, self.window) for g in gray])
        return mean, np.sqrt(np.maximum(square - mean ** 2, 0))

    def fit(self, items):
        sums = np.zeros((2, 2))  # [característica, (suma, suma de cuadrados)]
        count = 0
        for _, images in batches(items, self.size):
            for k, f in enumerate(self._features(images)):
                sums[k] += f.sum(), (f.astype(np.float64) ** 2).sum()
            count += images.shape[0] * self.size ** 2
        mu = sums[:, 0] / count
        self.state = {"mu": mu, "sd": np.sqrt(sums[:, 1] / count - mu ** 2)}

    def predict_batch(self, images):
        mean, std = self._features(images)
        z = np.maximum(np.abs(mean - self.state["mu"][0]) / self.state["sd"][0],
                       np.abs(std - self.state["mu"][1]) / self.state["sd"][1])
        return _smooth(z, self.sigma)


# --------------------------------------------------------------------------------------------------
# Características de una CNN preentrenada
# --------------------------------------------------------------------------------------------------

BACKBONES = {"resnet18": "ResNet18_Weights", "wide_resnet50_2": "Wide_ResNet50_2_Weights"}


class FeatureExtractor:
    """Salidas intermedias (`layer1`…`layer3`) de una ResNet de torchvision preentrenada en ImageNet."""

    def __init__(self, backbone, layers):
        import torch
        import torchvision.models as models
        from torchvision.models.feature_extraction import create_feature_extractor

        weights = getattr(models, BACKBONES[backbone]).IMAGENET1K_V1
        model = getattr(models, backbone)(weights=weights).eval()
        self.extractor = create_feature_extractor(model, return_nodes={layer: layer for layer in layers})
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, images):
        import torch

        x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
        with torch.inference_mode():
            return self.extractor((x - self.mean) / self.std)


def _upsample_to(feature, reference):
    import torch.nn.functional as F

    return F.interpolate(feature, size=reference.shape[-2:], mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------------------------------
# B1 · PaDiM
# --------------------------------------------------------------------------------------------------

class PaDiM:
    """Concatena layer1–layer3, elige `dims` canales al azar y ajusta una gaussiana (media y covarianza)
    en cada posición del mapa. Supone que la misma posición significa lo mismo en todas las imágenes:
    es razonable en objetos alineados, no necesariamente en una textura."""

    def __init__(self, backbone="resnet18", dims=100, size=PATCH_SIZE, sigma=4, seed=0):
        self.backbone, self.dims, self.size, self.sigma, self.seed = backbone, dims, size, sigma, seed
        self.extractor = FeatureExtractor(backbone, ["layer1", "layer2", "layer3"])
        self.state = {}

    def _embed(self, images):
        import torch

        f = self.extractor(images)
        e = torch.cat([f["layer1"], _upsample_to(f["layer2"], f["layer1"]), _upsample_to(f["layer3"], f["layer1"])], 1)
        if "idx" not in self.state:
            generator = torch.Generator().manual_seed(self.seed)
            self.state["idx"] = torch.randperm(e.shape[1], generator=generator)[:self.dims].numpy()
        e = e[:, torch.from_numpy(self.state["idx"])]
        b, d, h, w = e.shape
        return e.permute(0, 2, 3, 1).reshape(b, h * w, d).double(), (h, w)

    def fit(self, items):
        import torch

        total, outer, n = 0, 0, 0
        for _, images in batches(items, self.size):
            e, self.grid = self._embed(images)
            total = total + e.sum(0)
            outer = outer + torch.einsum("bpi,bpj->pij", e, e)
            n += e.shape[0]
        mean = total / n
        cov = (outer - n * torch.einsum("pi,pj->pij", mean, mean)) / (n - 1)
        cov = cov + 0.01 * torch.eye(self.dims, dtype=cov.dtype)  # regularización del paper
        self.state.update(mean=mean.float().numpy(), inv=torch.linalg.inv(cov).float().numpy(), grid=np.array(self.grid))

    def predict_batch(self, images):
        import torch

        e, (h, w) = self._embed(images)
        diff = e.float() - torch.from_numpy(self.state["mean"])
        tmp = torch.einsum("bpi,pij->bpj", diff, torch.from_numpy(self.state["inv"]))
        distance = (tmp * diff).sum(-1).clamp(min=0).sqrt().reshape(-1, 1, h, w)
        maps = _upsample_to(distance, torch.zeros(1, 1, self.size, self.size))[:, 0].numpy()
        return _smooth(maps, self.sigma)


# --------------------------------------------------------------------------------------------------
# B2 · PatchCore
# --------------------------------------------------------------------------------------------------

def greedy_coreset(features, ratio, projection_dim=128, seed=0):
    """Índices de un *coreset* k-center greedy (PatchCore, sección 3.2): en cada paso agrega el punto más lejano
    a los ya elegidos, así el subconjunto cubre todo el espacio de características normales.
    Las distancias se calculan en una proyección aleatoria de baja dimensión (Johnson-Lindenstrauss)."""
    import torch

    n = features.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return torch.arange(n)
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(features.shape[1], projection_dim, generator=generator) / projection_dim ** 0.5
    z = features @ projection
    selected = [int(torch.randint(n, (1,), generator=generator))]
    min_distance = ((z - z[selected[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        index = int(torch.argmax(min_distance))
        selected.append(index)
        min_distance = torch.minimum(min_distance, ((z - z[index]) ** 2).sum(1))
    return torch.tensor(selected)


class PatchCore:
    """Características locales de layer2 y layer3 (promediadas en vecindarios de 3×3), guardadas en un banco
    de memoria. El puntaje de cada posición es la distancia a la característica normal más cercana; el del
    patch es el máximo. No depende de la posición, por eso encaja con texturas."""

    def __init__(self, backbone="resnet18", coreset=0.01, size=PATCH_SIZE, sigma=4, max_dims=1024, seed=0):
        self.backbone, self.coreset, self.size, self.sigma = backbone, coreset, size, sigma
        self.max_dims, self.seed = max_dims, seed
        self.extractor = FeatureExtractor(backbone, ["layer2", "layer3"])
        self.state = {}

    def _embed(self, images):
        import torch
        import torch.nn.functional as F

        f = self.extractor(images)
        l2 = F.avg_pool2d(f["layer2"], 3, stride=1, padding=1)
        l3 = _upsample_to(F.avg_pool2d(f["layer3"], 3, stride=1, padding=1), l2)
        e = torch.cat([l2, l3], 1)
        b, d, h, w = e.shape
        e = e.permute(0, 2, 3, 1).reshape(-1, d)
        if d > self.max_dims:  # reducción de dimensión del paper: promedio adaptativo sobre canales
            e = F.adaptive_avg_pool1d(e.unsqueeze(1), self.max_dims).squeeze(1)
        return e, (b, h, w)

    def fit(self, items):
        import torch

        bank = torch.cat([self._embed(images)[0] for _, images in batches(items, self.size)])
        index = greedy_coreset(bank, self.coreset, seed=self.seed)
        self.state = {"memory": bank[index].numpy(), "bank_size": np.array(bank.shape[0])}

    def predict_batch(self, images, chunk=4096):
        import torch

        e, (b, h, w) = self._embed(images)
        memory = torch.from_numpy(self.state["memory"])
        distance = torch.cat([torch.cdist(e[s:s + chunk], memory).min(1).values for s in range(0, len(e), chunk)])
        maps = _upsample_to(distance.reshape(b, 1, h, w), torch.zeros(1, 1, self.size, self.size))[:, 0].numpy()
        return _smooth(maps, self.sigma)


# --------------------------------------------------------------------------------------------------
# Corrida: ajuste, inferencia y caché
# --------------------------------------------------------------------------------------------------

def run(name, make_method, train_items, eval_sets, recompute=False, config=None):
    """Ajusta el método con `train_items` (sólo normales) e infiere sobre cada conjunto de `eval_sets`.

    Guarda en results/raw_fabrid/bloque_b/<name>/ el estado del modelo, los puntajes por patch y los mapas
    (a MAP_SIZE, float16). Si ya existen, los lee: el notebook se puede re-ejecutar sin recalcular.
    Devuelve {split: {"scores", "maps", "items"}} y la info de costo en `corrida.json`.
    """
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    info_path = out / "corrida.json"
    info = json.loads(info_path.read_text()) if info_path.exists() and not recompute else {}
    method = None
    results = {}
    for split, items in eval_sets.items():
        scores_path, maps_path = out / f"puntajes_{split}.csv", out / f"mapas_{split}.npy"
        if scores_path.exists() and maps_path.exists() and not recompute:
            with scores_path.open(newline="") as handle:
                saved = {row["patch"]: float(row["puntaje"]) for row in csv.DictReader(handle)}
            if set(saved) == {i["name"] for i in items}:
                results[split] = {"items": items, "scores": np.array([saved[i["name"]] for i in items]),
                                  "maps": np.load(maps_path).astype(np.float32)}
                continue
        if method is None:
            method = make_method()
            state_path = out / "modelo.npz"
            if state_path.exists() and not recompute:
                method.state = dict(np.load(state_path, allow_pickle=False))
            else:
                start = time.perf_counter()
                method.fit(train_items)
                info.update(config or {}, segundos_ajuste=round(time.perf_counter() - start, 1),
                            normales_entrenamiento=len(train_items))
                np.savez(state_path, **method.state)
        start = time.perf_counter()
        scores, maps = [], []
        for _, images in batches(items, method.size):
            full = method.predict_batch(images)
            scores.append(full.reshape(len(full), -1).max(1))
            maps.append(_resize_maps(full).astype(np.float16))
        info[f"segundos_por_patch_{split}"] = round((time.perf_counter() - start) / len(items), 4)
        scores, maps = np.concatenate(scores), np.concatenate(maps)
        with scores_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patch", "madre", "rollo", "etiqueta", "puntaje"])
            writer.writerows([i["name"], i["parent"], i["roll"], i["label"], f"{s:.6g}"] for i, s in zip(items, scores))
        np.save(maps_path, maps)
        results[split] = {"items": items, "scores": scores, "maps": maps.astype(np.float32)}
    if method is not None and "memory" in method.state:
        info["tamano_memoria"] = int(method.state["memory"].shape[0])
        info["banco_completo"] = int(method.state["bank_size"])
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    for split, result in results.items():
        result["info"] = info
        result["segundos_por_patch"] = info.get(f"segundos_por_patch_{split}")
    return results


# --------------------------------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------------------------------

def patch_auroc(labels, scores):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def grouped_bootstrap_auroc(labels, scores, groups, n=1000, seed=0, level=0.95):
    """Intervalo de AUROC remuestreando madres (no patches) con reposición."""
    labels, scores, groups = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = list(members)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        index = np.concatenate([members[keys[k]] for k in rng.integers(len(keys), size=len(keys))])
        if 0 < labels[index].sum() < len(index):
            values.append(patch_auroc(labels[index], scores[index]))
    return float(np.quantile(values, (1 - level) / 2)), float(np.quantile(values, (1 + level) / 2))


_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz pasó a llamarse trapezoid en NumPy 2


def pixel_metrics(maps, masks, bins=1000, fpr_limit=0.3):
    """AUROC por píxel y AUPRO con histogramas de umbrales (exactos salvo la discretización en `bins`).

    AUPRO (Bergmann et al., 2021): para cada umbral, PRO es el promedio, sobre cada región conexa defectuosa,
    de la fracción de la región marcada; se integra PRO en función de la tasa de falsos positivos por píxel
    hasta `fpr_limit` (0,3) y se normaliza para que el máximo sea 1. A diferencia del AUROC por píxel, cada
    defecto pesa lo mismo sin importar su tamaño.
    """
    from scipy.ndimage import label

    edges = np.linspace(float(maps.min()), float(maps.max()) + 1e-6, bins + 1)
    neg, pos, regions = np.zeros(bins), np.zeros(bins), []
    for m, mask in zip(maps, masks):
        neg += np.histogram(m[~mask], edges)[0]
        if mask.any():
            pos += np.histogram(m[mask], edges)[0]
            components, count = label(mask)
            for k in range(1, count + 1):
                regions.append(np.histogram(m[components == k], edges)[0])
    # Recorrer los umbrales de mayor a menor: acumular desde el último bin.
    fpr = np.concatenate([[0], np.cumsum(neg[::-1]) / neg.sum()])
    tpr = np.concatenate([[0], np.cumsum(pos[::-1]) / max(pos.sum(), 1)])
    result = {"AUROC_pixel": float(_trapezoid(tpr, fpr))}
    if regions:
        regions = np.array(regions)
        pro = np.concatenate([[0], (np.cumsum(regions[:, ::-1], 1) / regions.sum(1, keepdims=True)).mean(0)])
        keep = fpr <= fpr_limit
        f, p = fpr[keep], pro[keep]
        if f[-1] < fpr_limit:  # interpolar el punto exacto en fpr_limit
            f, p = np.append(f, fpr_limit), np.append(p, np.interp(fpr_limit, fpr, pro))
        result["AUPRO@0.3"] = float(_trapezoid(p, f) / fpr_limit)
        result["regiones_defecto"] = int(len(regions))
    return result


def best_f1_threshold(labels, scores):
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(name, result, threshold=None):
    """Todas las métricas de un conjunto (salida de `run`). En valid el umbral es el de máximo F1; en test
    tiene que venir de valid."""
    items, scores = result["items"], result["scores"]
    labels = np.array([i["label"] for i in items])
    masks = np.stack([load_mask(i) for i in items])
    threshold = best_f1_threshold(labels, scores) if threshold is None else threshold
    predicted = scores >= threshold
    tp = int((predicted & (labels == 1)).sum())
    precision = tp / max(int(predicted.sum()), 1)
    recall = tp / max(int(labels.sum()), 1)
    by_roll = {}
    for roll in sorted({i["roll"] for i in items}):
        index = [k for k, i in enumerate(items) if i["roll"] == roll]
        if 0 < labels[index].sum() < len(index):
            by_roll[roll] = round(patch_auroc(labels[index], scores[index]), 4)
    info = result.get("info", {})
    return {"corrida": name, "patches": len(items), "defectuosos": int(labels.sum()),
            "madres": len({i["parent"] for i in items}),
            "AUROC_patch": patch_auroc(labels, scores),
            "ic95_AUROC_patch": grouped_bootstrap_auroc(labels, scores, [i["parent"] for i in items]),
            **pixel_metrics(result["maps"], masks),
            "umbral": threshold, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "AUROC_por_rollo": by_roll,
            "segundos_ajuste": info.get("segundos_ajuste"),
            "segundos_por_patch": result.get("segundos_por_patch"),
            "tamano_memoria": info.get("tamano_memoria")}


def metrics_table(metrics):
    import pandas as pd

    df = pd.DataFrame(metrics).set_index("corrida")
    df["AUROC_patch (IC 95 %)"] = [f"{a:.3f} ({lo:.3f}–{hi:.3f})"
                                   for a, (lo, hi) in zip(df["AUROC_patch"], df["ic95_AUROC_patch"])]
    columns = ["AUROC_patch (IC 95 %)", "AUROC_pixel", "AUPRO@0.3", "precision", "recall", "f1",
               "segundos_ajuste", "segundos_por_patch", "tamano_memoria"]
    return df[[c for c in columns if c in df]].round(3)


def roll_table(metrics):
    import pandas as pd

    return pd.DataFrame({m["corrida"]: m["AUROC_por_rollo"] for m in metrics}).T.round(3)


# --------------------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------------------

def plot_samples(items, n=4, seed=0):
    """Patches normales (arriba) y defectuosos con su máscara en rojo (abajo)."""
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    good = rng.sample([i for i in items if i["label"] == 0], n)
    defect = rng.sample([i for i in items if i["label"] == 1], n)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4), squeeze=False)
    for row, group in enumerate((good, defect)):
        for ax, item in zip(axes[row], group):
            ax.imshow(load_image(item["path"]))
            if item["label"]:
                ax.contour(load_mask(item, PATCH_SIZE), levels=[0.5], colors="red", linewidths=1)
            ax.set_title(f"{'defecto' if item['label'] else 'normal'} · {item['roll']}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    return fig


def plot_maps(results, names, split="valid", n=4, seed=0, worst=False):
    """Para patches defectuosos: imagen con máscara real, y el mapa de anomalía de cada método.

    Con `worst=True` muestra los defectos peor puntuados por el primer método (análisis de errores).
    """
    import matplotlib.pyplot as plt

    first = results[names[0]][split]
    defect_index = [k for k, i in enumerate(first["items"]) if i["label"] == 1]
    if worst:
        chosen = sorted(defect_index, key=lambda k: first["scores"][k])[:n]
    else:
        chosen = random.Random(seed).sample(defect_index, min(n, len(defect_index)))
    fig, axes = plt.subplots(1 + len(names), len(chosen), figsize=(2.6 * len(chosen), 2.6 * (1 + len(names))),
                             squeeze=False)
    for col, k in enumerate(chosen):
        item = first["items"][k]
        axes[0][col].imshow(load_image(item["path"]))
        axes[0][col].contour(load_mask(item, PATCH_SIZE), levels=[0.5], colors="red", linewidths=1)
        axes[0][col].set_title("real", fontsize=8)
        for row, name in enumerate(names, start=1):
            r = results[name][split]
            axes[row][col].imshow(r["maps"][k], cmap="inferno")
            axes[row][col].set_title(f"{name} · {r['scores'][k]:.2f}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_roc(results, names, split="valid", ax=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    for name in names:
        r = results[name][split]
        fpr, tpr, _ = roc_curve([i["label"] for i in r["items"]], r["scores"])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], ls=":", c="gray")
    ax.set_xlabel("tasa de falsos positivos")
    ax.set_ylabel("tasa de verdaderos positivos")
    ax.set_title(f"ROC por patch ({split})")
    ax.legend(fontsize=8)
    return ax


def plot_cost_vs_quality(metrics, metric="AUROC_patch", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for m in metrics:
        if m["segundos_por_patch"]:
            ax.scatter(m["segundos_por_patch"], m[metric])
            ax.annotate(m["corrida"], (m["segundos_por_patch"], m[metric]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("segundos por patch (log)")
    ax.set_ylabel(metric)
    ax.set_title(f"Costo vs. {metric}")
    ax.grid(alpha=0.3)
    return ax
