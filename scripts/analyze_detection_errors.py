"""Análisis de errores de detección en CISUTAC, al estilo de TIDE (Bolya et al., ECCV 2020). Sólo valid.

Responde por qué el mAP@0.5 es bajo: si el modelo alucina defectos, si no los encuentra o si los encuentra
pero encuadra mal la caja. Lee carpetas de corridas con predicciones.json y metricas.json, tal como las
escriben scripts/evaluate_cisutac.py y cisutac_baseline.evaluate, y usa src/detection_eval.py:

- error_breakdown, con el umbral de confianza elegido en valid de cada corrida: composición de las
  predicciones (aciertos, errores de localización, duplicados y alucinaciones), estado de los defectos
  reales por tamaño (detectados, sólo con caja mala, no detectados) y recall a IoU 0,1, 0,3 y 0,5;
- error_deltas, sin umbral, como el mAP: cuánto subiría mAP@0.5 si se corrigiera cada tipo de error.

Las corridas de test se saltean sin leer sus predicciones. Para cada corrida se verifica que la cantidad de
imágenes y cajas y el recall a IoU 0,5 coincidan con su metricas.json (mismas cajas reales).

Escribe en --salida (por defecto results/analisis_errores/, no versionado), por dataset:
errores_predicciones.csv, errores_defectos.csv, errores_delta_map.csv, errores_<dataset>.png y
errores_<dataset>_por_tamano.png. Con --ejemplos, además, errores_<corrida>_ejemplos.jpg: recortes de cada
tipo de error (scripts/plot_cisutac_predictions.py --por-error).

Ejemplos:
  python scripts/analyze_detection_errors.py --carpeta results/cisutac/baseline
  python scripts/analyze_detection_errors.py --carpeta results/cisutac/bloque_a/evaluaciones --patron "holes_*" \
      --ejemplos
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_eval import SIZE_EDGES, SIZE_LABELS, error_breakdown, error_deltas
from plot_cisutac_predictions import plot_error_examples, styled

DATA = Path(os.environ.get("REWEAR_DATA_DIR", ROOT / "data")).expanduser().resolve()
IOUS = (0.1, 0.3, 0.5)
# Un color por tipo de resultado, el mismo en todos los paneles (paleta validada para barras apiladas).
COLORS = {"aciertos": "#2a78d6", "localizacion": "#eb6834", "duplicados": "#1baf7a", "alucinaciones": "#eda100",
          "no_detectados": "#6250d6"}
NAMES = {"aciertos": "acierto (IoU ≥ 0,5)", "localizacion": "error de localización (IoU 0,1–0,5)",
         "duplicados": "duplicado", "alucinaciones": "alucinación (IoU < 0,1)", "no_detectados": "no detectado"}
STATES = {"detectados": ("aciertos", "detectado"), "caja_mala": ("localizacion", "sólo con caja mala"),
          "no_detectados": ("no_detectados", "no detectado")}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
# Estilo del proyecto, aplicado sólo mientras se dibuja (`styled`).
STYLE = {"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": "#c3c2b7",
         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
         "savefig.facecolor": SURFACE}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))  # unión, en orden de aparición
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()} for row in rows)


def load_run(folder):
    """Datos de una corrida; None si no es de valid. Acepta los dos formatos de metricas.json."""
    metrics = json.loads((folder / "metricas.json").read_text())
    config = metrics.get("configuracion", metrics)  # evaluate_cisutac.py guarda la configuración aparte
    if config["split"] != "valid":
        return None
    gt = json.loads((DATA / "derived" / f"cisutac_{config['dataset']}" / "coco_valid.json").read_text())
    gt["images"] = gt["images"][:metrics["imagenes"]]  # las dos evaluaciones recortan con `limit` igual
    kept = {im["id"] for im in gt["images"]}
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in kept]
    return {"corrida": folder.name, "carpeta": folder, "dataset": config["dataset"], "metrics": metrics, "gt": gt,
            "predictions": json.loads((folder / "predicciones.json").read_text())}


def analyze(run):
    m, gt = run["metrics"], run["gt"]
    threshold = m["umbral_confianza"]
    breakdown = error_breakdown(gt, run["predictions"], threshold, IOUS)
    if len(gt["annotations"]) != m["cajas_reales"] or abs(breakdown["recall"][0.5]["recall"] - m["recall"]) > 1e-9:
        print(f"  {run['corrida']}: las cajas reales no coinciden con las de su evaluación; se saltea")
        return None
    by_size = error_breakdown(gt, run["predictions"], threshold, IOUS, edges=SIZE_EDGES, labels=SIZE_LABELS)
    return {**run, "umbral": threshold, "errores": breakdown, "por_tamano": by_size["cajas"],
            "deltas": error_deltas(gt, run["predictions"])}


def table_rows(results):
    predictions, defects, deltas = [], [], []
    for r in results:
        kinds = r["errores"]["predicciones"]
        total = sum(kinds[k] for k in ("aciertos", "duplicados", "localizacion", "alucinaciones"))
        predictions.append({"dataset": r["dataset"], "corrida": r["corrida"], "umbral": r["umbral"],
                            "predicciones": total, **kinds,
                            **{f"pct_{k}": kinds[k] / total if total else None
                               for k in ("aciertos", "localizacion", "duplicados", "alucinaciones")}})
        recall = r["errores"]["recall"]
        for label, states in [*r["errores"]["cajas"].items(), ("total", None)]:
            if states is None:
                states = {s: sum(c[s] for c in r["errores"]["cajas"].values()) for s in STATES}
            row = {"dataset": r["dataset"], "corrida": r["corrida"], "rango": label,
                   "defectos": sum(states.values()), **states}
            for iou in IOUS:
                row[f"recall_iou{iou:g}"] = recall[iou]["recall"] if label == "total" else recall[iou]["por_tamano"][label]
                if label == "total":
                    row[f"ic95_iou{iou:g}"] = "–".join(f"{v:.3f}" for v in recall[iou]["ic95"])
            defects.append(row)
        deltas.append({"dataset": r["dataset"], "corrida": r["corrida"], **r["deltas"]})
    return predictions, defects, deltas


@styled(STYLE)
def plot_summary(results, path):
    """(a) composición de las predicciones, (b) recall según el IoU exigido, (c) mejora de mAP@0.5 por corrección."""
    n = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 2.2 + 0.8 * n), gridspec_kw={"width_ratios": [1.3, 0.9, 1.1]})
    names = [r["corrida"] for r in results]
    ys = range(n)[::-1]
    kinds = ("aciertos", "localizacion", "duplicados", "alucinaciones")
    for y, r in zip(ys, results):
        counts = r["errores"]["predicciones"]
        total = sum(counts[k] for k in kinds)
        left = 0
        for kind in kinds:
            share = counts[kind] / total if total else 0
            axes[0].barh(y, share, left=left, height=0.6, color=COLORS[kind], edgecolor=SURFACE, linewidth=1.5)
            if share >= 0.08:
                axes[0].text(left + share / 2, y, f"{counts[kind]}", ha="center", va="center", fontsize=8,
                             color="white" if kind in ("aciertos", "localizacion") else INK)
            left += share
        axes[0].text(1.01, y, f"n = {total}", va="center", fontsize=8, color=MUTED)
    axes[0].set_yticks(list(ys), names)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("fracción de las predicciones sobre el umbral de valid")
    axes[0].set_title("(a) ¿Qué son las predicciones?", loc="left", fontsize=10, color=INK)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in kinds],
                   labels=[NAMES[k] for k in kinds], loc="upper center", bbox_to_anchor=(0.5, -0.22),
                   ncol=2, frameon=False, fontsize=8)

    markers = "osD^vP*X"
    shades = ("#0b0b0b", "#52514e", "#8a8984")
    for k, r in enumerate(results):
        values = [r["errores"]["recall"][iou]["recall"] for iou in IOUS]
        axes[1].plot(IOUS, values, marker=markers[k % len(markers)], color=shades[k % len(shades)], linewidth=2,
                     markersize=7, label=r["corrida"])
    axes[1].set_xticks(IOUS, [f"{iou:g}".replace(".", ",") for iou in IOUS])
    axes[1].set_xlabel("IoU mínimo para contar un acierto")
    axes[1].set_ylabel("recall")
    axes[1].set_ylim(0, max(0.05, max(r["errores"]["recall"][0.1]["recall"] for r in results) * 1.25))
    axes[1].grid(axis="y", color=GRID)
    axes[1].set_title("(b) ¿Los encuentra pero encuadra mal?", loc="left", fontsize=10, color=INK)
    axes[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))

    fixes = ("localizacion", "duplicados", "alucinaciones", "no_detectados")
    height = 0.8 / len(fixes)
    for y, r in zip(ys, results):
        for j, kind in enumerate(fixes):
            value = r["deltas"][f"delta_{kind}"] or 0
            axes[2].barh(y + 0.3 - (j + 0.5) * height, value, height=height * 0.9, color=COLORS[kind])
    axes[2].set_yticks(list(ys), [f"mAP@0.5 = {r['deltas']['mAP@0.5']:.3f}".replace("0.5 =", "0,5 =").replace(".", ",")
                                  for r in results], fontsize=8)
    axes[2].set_xlabel("cuánto sube mAP@0.5 si se corrige sólo ese error")
    axes[2].grid(axis="x", color=GRID)
    axes[2].xaxis.set_major_locator(plt.MaxNLocator(5))
    axes[2].set_title("(c) ¿Qué error le cuesta más mAP?", loc="left", fontsize=10, color=INK)
    axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in fixes],
                   labels=["localización", "duplicados", "alucinaciones", "no detectados"], loc="upper center",
                   bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Errores de detección en CISUTAC {results[0]['dataset'].capitalize()}, valid "
                 "(con el umbral de confianza elegido en valid)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@styled(STYLE)
def plot_by_size(results, path, columns=3):
    """Estado de los defectos reales por lado equivalente, una corrida por panel."""
    n = len(results)
    rows = -(-n // columns)
    fig, axes = plt.subplots(rows, min(n, columns), figsize=(4.6 * min(n, columns), 3.4 * rows), squeeze=False,
                             sharey=True)
    for ax, r in zip(axes.flat, results):
        bottom = [0.0] * len(SIZE_LABELS)
        totals = [sum(r["por_tamano"][label].values()) for label in SIZE_LABELS]
        for state, (color_key, _) in STATES.items():
            shares = [r["por_tamano"][label][state] / t if t else 0 for label, t in zip(SIZE_LABELS, totals)]
            ax.bar(range(len(SIZE_LABELS)), shares, bottom=bottom, width=0.7, color=COLORS[color_key],
                   edgecolor=SURFACE, linewidth=1.5)
            bottom = [b + s for b, s in zip(bottom, shares)]
        ax.set_xticks(range(len(SIZE_LABELS)), [f"{label}\nn = {t}" for label, t in zip(SIZE_LABELS, totals)],
                      fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(r["corrida"], fontsize=9, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("fracción de los defectos reales")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c, _ in STATES.values()],
               labels=[name for _, name in STATES.values()], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"¿Qué defectos de CISUTAC {results[0]['dataset'].capitalize()} no encuentra? (valid, por lado "
                 "equivalente en la original)", fontsize=11, y=1.06)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Análisis de errores de detección (sólo valid).")
    parser.add_argument("--carpeta", type=Path, required=True,
                        help="carpeta con una subcarpeta por corrida (predicciones.json + metricas.json)")
    parser.add_argument("--patron", default="*", help="patrón de las subcarpetas a analizar, p. ej. 'holes_*'")
    parser.add_argument("--salida", type=Path, default=ROOT / "results" / "analisis_errores")
    parser.add_argument("--ejemplos", action="store_true", help="también una figura de recortes por corrida")
    args = parser.parse_args()

    folders = sorted(f for f in args.carpeta.glob(args.patron)
                     if (f / "metricas.json").exists() and (f / "predicciones.json").exists())
    results = []
    for folder in folders:
        run = load_run(folder)
        if run is None:
            print(f"  {folder.name}: no es de valid; se saltea sin leer sus predicciones")
            continue
        print(f"  {folder.name}: analizando")
        if (result := analyze(run)) is not None:
            results.append(result)
    if not results:
        sys.exit(f"No hay corridas de valid en {args.carpeta} con el patrón {args.patron!r}")

    args.salida.mkdir(parents=True, exist_ok=True)
    predictions, defects, deltas = table_rows(results)
    for name, rows in (("errores_predicciones", predictions), ("errores_defectos", defects),
                       ("errores_delta_map", deltas)):
        write_csv(args.salida / f"{name}.csv", rows)
    for dataset in sorted({r["dataset"] for r in results}):
        subset = [r for r in results if r["dataset"] == dataset]
        plot_summary(subset, args.salida / f"errores_{dataset}.png")
        plot_by_size(subset, args.salida / f"errores_{dataset}_por_tamano.png")
    if args.ejemplos:
        for r in results:
            plot_error_examples(r["dataset"], r["carpeta"], args.salida / f"errores_{r['corrida']}_ejemplos.jpg")

    import pandas as pd

    pd.set_option("display.width", 200, "display.max_columns", 30)
    show = ["corrida", "umbral", "predicciones", "pct_aciertos", "pct_localizacion", "pct_duplicados", "pct_alucinaciones"]
    print("\nPredicciones sobre el umbral:\n", pd.DataFrame(predictions)[show].round(3).to_string(index=False))
    show = ["corrida", "rango", "defectos", "detectados", "caja_mala", "no_detectados",
            *[f"recall_iou{iou:g}" for iou in IOUS]]
    print("\nDefectos reales:\n", pd.DataFrame(defects)[show].round(3).to_string(index=False))
    print("\nMejora de mAP@0.5 si se corrige cada error:\n",
          pd.DataFrame(deltas).drop(columns="dataset").round(3).to_string(index=False))
    print(f"\nResultados en {args.salida}")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
