"""Genera las tablas y figuras del EDA de CISUTAC Holes/Spots sin alterar los datasets.

Separado de generate_eda_summary.py porque sólo requiere los datos de CISUTAC.
"""

import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection_eval import SIZE_LABELS, match_detections, summarize
from eda_utils import (box_xyxy, cisutac_coco_ground_truth, cisutac_inventory, cisutac_label_checks, cisutac_records,
                       cisutac_resize_table,
                       cisutac_size_quantiles, cisutac_size_thresholds, cisutac_split_proposal, cisutac_split_summary,
                       cisutac_tiling_table, coco_size_counts, dhash, equivalent_side, exact_duplicate_groups,
                       image_files, match_copy_instances, near_duplicate_pairs, scaled_sizes, shifted_iou,
                       yolo_class_names)

DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
OUT = ROOT / "reports" / "cisutac"
OUT.mkdir(parents=True, exist_ok=True)

# Estilo de las figuras: un color fijo por dataset (paleta categórica validada) y ejes/grilla recesivos.
COLORS = {"Holes": "#2a78d6", "Spots": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": "#c3c2b7",
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

# Semilla de las selecciones aleatorias de ejemplos (CIS-2); sólo se eligen imágenes de train/valid propuestos.
SEED = 2026
# Variante descartada en CIS-5, se conserva para comparar: test = lote de mayo, valid oficial.
TEST_MAY_FROM = "20260501"
# Revisión visual manual (CIS-5): misma prenda fotografiada otra vez con otro código hexadecimal.
SAME_GARMENT = [
    (("Holes", "20260317_143027_e451acfa_front_nobg_png"), ("Holes", "20260317_150658_4db3e9bc_front_nobg_png"),
     "misma prenda"),
    (("Holes", "20260304_143816_ccbf098d_front_nobg_png"), ("Spots", "20260304_134859_3d44fa0e_front_nobg_png"),
     "misma prenda"),
    (("Holes", "20260310_133911_067a21c2_front_nobg_png"), ("Spots", "20260310_141043_22092bd3_front_nobg_png"),
     "misma prenda"),
    (("Spots", "20260226_144202_69d39d0f_front_nobg_png"), ("Spots", "20260226_144419_e59495eb_front_nobg_png"),
     "probable (frente/espalda)"),
]


def write_csv(path, rows):
    """Escribe la tabla redondeando los decimales sólo al guardar (6 cifras significativas)."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: float(f"{v:.6g}") if isinstance(v, float) else v for k, v in row.items()}
                         for row in rows)


def plot_sizes(records_by_dataset, path):
    """CIS-3: distribución del lado equivalente de los defectos y cantidad de defectos por imagen."""
    fig, (ax_size, ax_count) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    edges = [2 ** (k / 3) for k in range(31)]  # de 1 a 1024 px, tres bins por octava (8, 16 y 32 son bordes)
    groups = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5), ("6–10", 6, 10), ("11–20", 11, 20),
              ("> 20", 21, 10 ** 6)]
    pct = {}
    for name, records in records_by_dataset.items():
        sides = [equivalent_side(i) for i in records["instances"] if i["version"] == "original"]
        pct[name] = [100 * sum(lo <= s < hi for s in sides) / len(sides) for lo, hi in zip(edges, edges[1:])]
        ax_size.stairs(pct[name], edges, color=COLORS[name], lw=2,
                       label=f"{name} (n = {len(sides):,} defectos)".replace(",", "."))
    for name, other in zip(pct, reversed(list(pct))):
        # Etiqueta directa donde la curva más se separa de la otra, para que no se pisen.
        b = max(range(len(edges) - 1), key=lambda j: pct[name][j] - pct[other][j])
        ax_size.annotate(name, ((edges[b] * edges[b + 1]) ** 0.5, pct[name][b]), xytext=(0, 4),
                         textcoords="offset points", ha="center", color=INK)
    for k, (name, records) in enumerate(records_by_dataset.items()):
        counts = [im["instancias"] for im in records["images"] if im["version"] == "original"]
        shares = [100 * sum(lo <= c <= hi for c in counts) / len(counts) for _, lo, hi in groups]
        ax_count.bar([g + (k - 0.5) * 0.4 for g in range(len(groups))], shares, width=0.38, color=COLORS[name],
                     label=f"{name} (n = {len(counts)} imágenes)")
        # La etiqueta de la segunda serie arranca en el borde izquierdo de su barra para no pisar la primera.
        ax_count.annotate(f"{shares[0]:.0f} %", (k * 0.4 - 0.39 if k else -0.2, shares[0]), xytext=(0, 3),
                          textcoords="offset points", ha="left" if k else "center", color=MUTED)
    ax_size.set_ylim(0, ax_size.get_ylim()[1] * 1.25)
    for x, text in ((8, "8 px"), (16, "16 px"), (32, "32 px: límite small de COCO")):
        ax_size.axvline(x, color="#898781", lw=0.8, zorder=0)
        ax_size.annotate(text, (x, 1), xycoords=("data", "axes fraction"), xytext=(3, -10),
                         textcoords="offset points", color=MUTED, fontsize=8)
    ax_size.set_xscale("log", base=2)
    ax_size.set_xticks([2 ** k for k in range(11)], [str(2 ** k) for k in range(11)])
    ax_size.set_xlim(1, 1024)
    ax_size.set_xlabel("Lado equivalente √(ancho·alto) de la caja, en px de la imagen original (escala log)")
    ax_size.set_ylabel("% de los defectos en cada intervalo")
    ax_size.set_title("Tamaño de los defectos", loc="left", fontsize=10)
    ax_count.set_xticks(range(len(groups)), [label for label, _, _ in groups])
    ax_count.set_xlabel("Defectos anotados en la imagen")
    ax_count.set_ylabel("% de las imágenes")
    ax_count.set_title("Defectos por imagen", loc="left", fontsize=10)
    for ax in (ax_size, ax_count):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    ax_size.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.88))
    ax_count.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def pick_resize_example(root, records, manifest):
    """Defecto de tamaño mediano fuera del test propuesto, con copia 640 coincidente (selección reproducible)."""
    split = {row["captura"]: row["split_propuesto"] for row in manifest}
    copy_of = {id(original): copy for original, copy in match_copy_instances(records)}
    candidates = [i for i in records["instances"] if id(i) in copy_of and split[i["captura"]] != "test"]
    target = median(equivalent_side(i) for i in records["instances"] if i["version"] == "original")
    chosen = min(candidates, key=lambda i: (abs(equivalent_side(i) - target), i["captura"], i["x_centro"]))
    paths = {im["version"]: root / im["split"] / "images" / im["archivo"]
             for im in records["images"] if im["captura"] == chosen["captura"]}
    return chosen, copy_of[id(chosen)], paths


def plot_resize_zoom(examples, path, window=128):
    """CIS-4: la misma zona de la prenda en la original, tras cada resize y en la copia real de 640×640."""
    fig, axes = plt.subplots(len(examples), 5, figsize=(12, 2.9 * len(examples)))
    for row_axes, (name, (inst, copy, paths)) in zip(axes, examples.items()):
        original = Image.open(paths["original"]).convert("RGB")
        width, height = original.size
        cx, cy = inst["x_centro"] * width, inst["y_centro"] * height
        x0 = min(max(cx - window / 2, 0), width - window)
        y0 = min(max(cy - window / 2, 0), height - window)
        panels = [("Original", original, 1, 1, equivalent_side(inst))]
        for side in (1280, 1024, 640):
            s = side / max(width, height)
            # BOX promedia los píxeles que cubre cada píxel de salida (equivalente a cv2.INTER_AREA).
            resized = original.resize((round(width * s), round(height * s)), Image.Resampling.BOX)
            panels.append((f"Lado mayor {side}", resized, s, s, equivalent_side(inst) * s))
        copy_image = Image.open(paths["640"]).convert("RGB")
        panels.append(("Copia 640×640 de Roboflow", copy_image, 640 / width, 640 / height, equivalent_side(copy)))
        for ax, (label, image, sx, sy, side_px) in zip(row_axes, panels):
            left, top, right, bottom = x0 * sx, y0 * sy, (x0 + window) * sx, (y0 + window) * sy
            crop = image.crop((int(left), int(top), int(right) + 1, int(bottom) + 1))
            ax.imshow(crop, interpolation="nearest",
                      extent=(int(left), int(left) + crop.width, int(top) + crop.height, int(top)))
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
            w, h = inst["ancho_px"] * sx, inst["alto_px"] * sy
            ax.add_patch(plt.Rectangle((cx * sx - 0.75 * w, cy * sy - 0.75 * h), 1.5 * w, 1.5 * h,
                                       fill=False, ec=COLORS[name], lw=1.2))
            ax.set_title(f"{label}\nlado del defecto ≈ {side_px:.1f} px".replace(".", ","), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        row_axes[0].set_ylabel(f"{name}\n{window}×{window} px de la original", fontsize=9, color=INK)
    fig.suptitle("Misma zona de la prenda a distintas resoluciones (el recuadro rodea la caja anotada, ampliada 1,5×)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def crop_around(image, inst, window):
    """Recorte cuadrado de `window` px centrado en la caja (en px de la imagen original)."""
    cx, cy = inst["x_centro"] * image.width, inst["y_centro"] * image.height
    x0 = int(min(max(cx - window / 2, 0), image.width - window))
    y0 = int(min(max(cy - window / 2, 0), image.height - window))
    return image.crop((x0, y0, x0 + window, y0 + window)), x0, y0


def draw_boxes(ax, chosen, others, x0, y0, color, margin=0):
    """Dibuja la caja elegida y, más finas, las demás cajas de la imagen; el halo blanco las separa del fondo."""
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    for inst, lw in [(chosen, 1.4)] + [(i, 0.8) for i in others if i is not chosen]:
        x1, y1, x2, y2 = box_xyxy(inst)
        ax.add_patch(plt.Rectangle((x1 - x0 - margin, y1 - y0 - margin), x2 - x1 + 2 * margin, y2 - y1 + 2 * margin,
                                   fill=False, ec=color, lw=lw, path_effects=halo))


def plot_examples_grid(examples, path):
    """CIS-2: imágenes completas con sus cajas; un círculo marca los defectos de lado < 32 px."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 12.5))
    for row_axes, (name, chosen) in zip(axes, examples.items()):
        for ax, (kind, (image_path, ins)) in zip(row_axes, chosen.items()):
            with Image.open(image_path) as im:
                im.draft("RGB", (im.width // 4, im.height // 4))
                im = im.convert("RGB")
            factor = im.width / ins[0]["img_ancho"]
            ax.imshow(im)
            for i in ins:
                x1, y1, x2, y2 = (v * factor for v in box_xyxy(i))
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=COLORS[name], lw=1.6))
                if equivalent_side(i) < 32:
                    ax.add_patch(plt.Circle(((x1 + x2) / 2, (y1 + y2) / 2), 80 * factor, fill=False,
                                            ec=COLORS[name], lw=1.2))
            count = f"{len(ins)} defecto" + ("s" if len(ins) > 1 else "")
            ax.set_title(f"{name} · {kind}\n{count} · lado mínimo {min(map(equivalent_side, ins)):.0f} px", fontsize=9)
            ax.axis("off")
    fig.suptitle("Ejemplos de train/valid con sus cajas (círculo: defecto de lado < 32 px, que no se ve a esta escala)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.5)
    fig.savefig(path, dpi=80, pil_kwargs={"quality": 85})


def plot_size_zoom(examples, path):
    """CIS-2: un defecto por rango de tamaño, a resolución completa y con su caja."""
    bins = [("lado < 8 px", 0, 8), ("8–16 px", 8, 16), ("16–32 px", 16, 32), ("32–64 px", 32, 64), ("≥ 64 px", 64, 1e9)]
    fig, axes = plt.subplots(len(examples), len(bins), figsize=(12, 2.8 * len(examples)))
    for row_axes, (name, (pool, paths, by_capture)) in zip(axes, examples.items()):
        rng = random.Random(f"{SEED}-zoom-{name}")
        for ax, (label, lo, hi) in zip(row_axes, bins):
            inst = rng.choice(sorted((i for i in pool if lo <= equivalent_side(i) < hi),
                                     key=lambda i: (i["captura"], i["x_centro"], i["y_centro"])))
            window = int(max(64, 3 * max(inst["ancho_px"], inst["alto_px"])))
            with Image.open(paths[inst["captura"]]) as im:
                crop, x0, y0 = crop_around(im.convert("RGB"), inst, window)
            ax.imshow(crop, interpolation="nearest")
            draw_boxes(ax, inst, by_capture[inst["captura"]], x0, y0, COLORS[name])
            ax.set_xlim(-0.5, window - 0.5)
            ax.set_ylim(window - 0.5, -0.5)
            ax.set_title(f"{name} · {label}\nlado {equivalent_side(inst):.1f} px · recorte {window} px".replace(".", ","),
                         fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)


def plot_thin_boxes(examples, path, window=48):
    """CIS-2: muestra de cajas con algún lado < 4 px, ampliadas para revisar si son defectos reales."""
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.8))
    cells = [(name, inst, paths, by_capture) for name, (sample, paths, by_capture) in examples.items()
             for inst in sample]
    for ax in axes.flat:
        ax.axis("off")
    for k, (ax, (name, inst, paths, by_capture)) in enumerate(zip(axes.flat, cells), start=1):
        with Image.open(paths[inst["captura"]]) as im:
            crop, x0, y0 = crop_around(im.convert("RGB"), inst, window)
        ax.imshow(crop, interpolation="nearest")
        # Recuadros 3 px más grandes que las cajas para no tapar los píxeles del defecto.
        draw_boxes(ax, inst, by_capture[inst["captura"]], x0, y0, COLORS[name], margin=3)
        ax.set_xlim(-0.5, window - 0.5)
        ax.set_ylim(window - 0.5, -0.5)
        ax.set_title(f"{k}. {name} · {inst['ancho_px']:.1f}×{inst['alto_px']:.1f} px".replace(".", ","), fontsize=8)
    fig.suptitle(f"Cajas con algún lado < 4 px: recortes de {window}×{window} px de la original "
                 "(recuadro grueso = caja revisada, finos = otras cajas; todos ampliados 3 px)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=90)


inventory, instances, manifests, comparison, quantile_rows, threshold_rows = [], [], [], [], [], []
check_rows, coco_rows, grid_examples, zoom_examples, thin_examples = [], [], {}, {}, {}
error_rows = []
resize_rows, tiling_rows, resize_examples = [], [], {}
all_files, originals, official_split, records_by_dataset = [], {}, {}, {}
for name, folder in (("Holes", "cisutac_holes"), ("Spots", "cisutac_spots")):
    root = DATA / folder
    classes = yolo_class_names(root / "data.yaml")
    records = cisutac_records(root, classes)
    records_by_dataset[name] = records
    print(f"{name}: clases={classes} | carpeta test en disco={(root / 'test').exists()}")
    inventory += [{"dataset": name, "clases": "/".join(classes), **row} for row in cisutac_inventory(records)]
    quantile_rows += [{"dataset": name, **row} for row in cisutac_size_quantiles(records)]
    threshold_rows += [{"dataset": name, **row} for row in cisutac_size_thresholds(records)]
    # Decisión del equipo: el análisis usa sólo la versión original; la copia 640 queda en el inventario.
    original_instances = [row for row in records["instances"] if row["version"] == "original"]
    instances += [{"dataset": name, **row} for row in original_instances]

    all_files += [p for split in ("train", "valid") for p in image_files(root / split / "images")]
    for im in records["images"]:
        if im["version"] == "original":
            originals[(name, im["captura"])] = root / im["split"] / "images" / im["archivo"]
            official_split[(name, im["captura"])] = im["split"]

    manifest = cisutac_split_proposal(records)
    manifests += [{"dataset": name, **row} for row in manifest]
    resize_rows += [{"dataset": name, **row} for row in cisutac_resize_table(records)]
    tiling_rows += [{"dataset": name, **row} for row in cisutac_tiling_table(records)]
    resize_examples[name] = pick_resize_example(root, records, manifest)
    check_rows += [{"dataset": name, **row} for row in cisutac_label_checks(records)]
    coco_rows += [{"dataset": name, **row} for row in coco_size_counts(manifest, original_instances)]

    # CIS-6: recall a IoU 0,5 de un detector perfecto salvo 1 px de corrimiento (en x e y) en la imagen
    # que recibe el modelo. El IoU no depende de la escala, así que equivale a evaluar a esa resolución.
    gt_all = cisutac_coco_ground_truth(records, sorted(row["captura"] for row in manifest), classes[0])
    long_side = {im["id"]: max(im["width"], im["height"]) for im in gt_all["images"]}
    for label, input_side in (("resolución nativa", None), ("lado mayor 1024", 1024), ("lado mayor 640", 640)):
        predictions = []
        for a in gt_all["annotations"]:
            shift = 1 if input_side is None else long_side[a["image_id"]] / input_side
            predictions.append({"image_id": a["image_id"], "score": 1.0,
                                "bbox": [a["bbox"][0] + shift, a["bbox"][1] + shift, a["bbox"][2], a["bbox"][3]]})
        result = summarize(match_detections(gt_all, predictions))
        error_rows.append({"dataset": name, "entrada_al_modelo": label, "error_px_en_la_entrada": 1,
                           "recall": round(result["recall"], 3),
                           **{f"recall {k}": round(result[f"recall {k}"], 3) for k in SIZE_LABELS}})

    # CIS-2: ejemplos reproducibles, sólo de train/valid propuestos (el test no se usa para mirar ejemplos).
    split_of = {row["captura"]: row["split_propuesto"] for row in manifest}
    pool = [i for i in original_instances if split_of[i["captura"]] != "test"]
    paths = {capture: originals[(name, capture)] for capture in split_of}
    by_capture = defaultdict(list)
    for i in pool:
        by_capture[i["captura"]].append(i)
    sides = {capture: [equivalent_side(i) for i in ins] for capture, ins in by_capture.items()}
    kinds = {"fácil: 1–2 defectos de lado ≥ 64 px": lambda s: len(s) <= 2 and min(s) >= 64,
             "difícil: algún defecto de lado < 8 px": lambda s: len(s) <= 3 and min(s) < 8,
             "muchos defectos: 10 o más": lambda s: len(s) >= 10}
    rng = random.Random(f"{SEED}-grilla-{name}")
    grid_examples[name] = {}
    for kind, condition in kinds.items():
        capture = rng.choice(sorted(c for c, s in sides.items() if condition(s)))
        grid_examples[name][kind] = (paths[capture], by_capture[capture])
    zoom_examples[name] = (pool, paths, by_capture)
    thin = sorted((i for i in pool if min(i["ancho_px"], i["alto_px"]) < 4),
                  key=lambda i: (i["captura"], i["x_centro"], i["y_centro"]))
    thin_examples[name] = (random.Random(f"{SEED}-finas-{name}").sample(thin, min(len(thin), 12 if name == "Spots" else 6)),
                           paths, by_capture)
    # Validación de la simulación: cajas de la copia real frente al estirado calculado de las originales.
    pairs = match_copy_instances(records)
    simulated = scaled_sizes([original for original, _ in pairs], "640x640")
    gaps = [abs(s - c) for (sw, sh), (_, copy) in zip(simulated, pairs)
            for s, c in ((sw, copy["ancho_px"]), (sh, copy["alto_px"]))]
    print(f"{name}: copia 640 real vs. simulada en {len(pairs)} cajas: |diferencia| mediana "
          f"{median(gaps):.2f} px, máxima {max(gaps):.2f} px")
    variants = {
        "oficial": [{**row, "split_propuesto": row["split_oficial"]} for row in manifest],
        "test_mayo": [{**row, "split_propuesto": "test" if row["fecha"] >= TEST_MAY_FROM else row["split_oficial"]}
                      for row in manifest],
        "propuesta": manifest,
    }
    for variant, rows in variants.items():
        comparison += [{"dataset": name, "variante": variant, **row}
                       for row in cisutac_split_summary(rows, original_instances)]

# CIS-5: duplicados exactos y casi duplicados perceptuales entre todos los originales de ambos datasets.
exact = exact_duplicate_groups(all_files)
hashes = {key: dhash(path) for key, path in originals.items()}
similar = [{"distancia_dhash": distance,
            "dataset_a": a[0], "captura_a": a[1], "split_a": official_split[a],
            "dataset_b": b[0], "captura_b": b[1], "split_b": official_split[b],
            "mismo_dia": a[1][:8] == b[1][:8]}
           for distance, a, b in near_duplicate_pairs(hashes, 7)]

fig, axes = plt.subplots(len(SAME_GARMENT), 2, figsize=(8, 3.6 * len(SAME_GARMENT)))
for row_axes, (a, b, verdict) in zip(axes, SAME_GARMENT):
    for ax, key in zip(row_axes, (a, b)):
        with Image.open(originals[key]) as im:
            im.draft("RGB", (im.width // 2, im.height // 2))
            im = im.convert("RGB")
            ax.imshow(im.crop((int(im.width * .1), int(im.height * .12), int(im.width * .9), int(im.height * .65))))
        date, time, code = key[1].split("_")[:3]
        ax.set_title(f"{key[0]} · {official_split[key]} · {date} {time[:2]}:{time[2:4]} · {code}\n{verdict}",
                     fontsize=9)
        ax.axis("off")
fig.suptitle("Misma prenda con distinto código hexadecimal (revisión visual, CIS-5)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "cis5_misma_prenda.jpg", dpi=70, pil_kwargs={"quality": 85})

plot_sizes(records_by_dataset, OUT / "cis3_distribucion_tamanos.png")
plot_resize_zoom(resize_examples, OUT / "cis4_zoom_resize.png")
plot_examples_grid(grid_examples, OUT / "cis2_ejemplos_imagenes.jpg")
plot_size_zoom(zoom_examples, OUT / "cis2_zoom_por_tamano.png")
plot_thin_boxes({name: thin_examples[name] for name in ("Spots", "Holes")}, OUT / "cis2_cajas_finas.png")
iou_rows = [{"lado_px": side, "desplazamiento_px": shift, "iou": round(shifted_iou(side, shift), 3)}
            for side in (3, 4, 8, 16, 18, 25, 32, 64) for shift in (1, 2)]

write_csv(OUT / "cis1_inventario.csv", inventory)
write_csv(OUT / "instancias_original.csv", instances)
write_csv(OUT / "cis3_cuantiles.csv", quantile_rows)
write_csv(OUT / "cis3_umbrales.csv", threshold_rows)
write_csv(OUT / "cis4_resize.csv", resize_rows)
write_csv(OUT / "cis4_tiles.csv", tiling_rows)
write_csv(OUT / "cis2_controles_etiquetas.csv", check_rows)
write_csv(OUT / "cis6_categorias_por_split.csv", coco_rows)
write_csv(OUT / "cis6_iou_desplazamiento.csv", iou_rows)
write_csv(OUT / "cis6_recall_con_error.csv", error_rows)
write_csv(OUT / "cis5_pares_dhash.csv", similar)
write_csv(OUT / "cis5_split_propuesto.csv", manifests)
write_csv(OUT / "cis5_comparacion_splits.csv", comparison)

for row in quantile_rows + threshold_rows:
    print(row)
print(f"\nDuplicados exactos (SHA-256, {len(all_files)} archivos): {len(exact)}")
print(f"Pares de originales con dHash a <= 7 bits: {len(similar)}")
for row in comparison:
    print(row)
print(f"\nResultados en {OUT.relative_to(ROOT)}")
