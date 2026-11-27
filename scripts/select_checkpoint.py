"""Elige el checkpoint de una corrida de CISUTAC por mAP@0.5 por foto en valid (issue #5).

El best.pt de Ultralytics se elige con su validación interna sobre los tiles de valid, que casi no incluye tiles
sanos. Este script evalúa cada checkpoint guardado con la misma métrica y la misma unidad con que se reporta:
mAP@0.5 por foto en valid, con `cisutac_baseline.evaluate` y la configuración de inferencia de la corrida.

1. Lee la configuración de la corrida: `corrida.json` (de `cb.train`) o, si no existe, el `args.yaml` de
   Ultralytics. De `data` (la ruta del data.yaml) salen el dataset y el modo: `cisutac_<dataset>/data.yaml` es
   la foto completa con `imgsz`; `cisutac_<dataset>_tiles/t<T>_o<O>/...` son tiles de T px con O % de solapamiento.
   `--tambien-completa` suma la foto completa a los tiles en inferencia (no queda registrado en la corrida).
2. Busca weights/best.pt, weights/last.pt y weights/epochN.pt (con `save_period`) y determina la época de cada
   uno en la numeración de results.csv (desde 1). Ultralytics numera desde 0: epoch5.pt es la época 6.
3. Evalúa cada checkpoint SÓLO en valid (no hay forma de pedir test) y elige el de mayor mAP@0.5. Se consideran
   empatados los que coinciden al redondear a 3 decimales; entre ellos gana la época más temprana.
4. Escribe seleccion_checkpoint.csv (una fila por checkpoint) y seleccion_checkpoint.json (el elegido, el
   criterio y los argumentos de `cb.evaluate`) en la carpeta de la corrida o en --salida. Si no es una prueba,
   también los copia a reports/cisutac/seleccion_checkpoint/<corrida>/ para versionarlos.

Las evaluaciones se guardan y se reutilizan como en los notebooks, en
results/cisutac/bloque_a/evaluaciones/<corrida>__<checkpoint>__valid_<inferencia>/ (con --reevaluar se
recalculan). Con --prueba N se evalúan sólo N fotos y los archivos llevan el sufijo _prueba: no son resultados.

Cómo usar el JSON en la evaluación de test (issue #7), sin volver a elegir:
    sel = json.loads(Path(".../seleccion_checkpoint.json").read_text())
    assert not sel["modo_prueba"] and sel["split"] == "valid"
    pesos = Path(sel["carpeta"]) / sel["elegido"]["archivo"]      # verificar también sel["elegido"]["sha256"]
    cb.evaluate(<nombre de test>, sel["dataset"], pesos, split="test", **sel["inferencia"],
                threshold=sel["elegido"]["umbral_confianza"], confirm_test=True)

Kaggle, corrida de YOLO26 (issue #6), aunque sólo tenga best.pt y last.pt: después de entrenar en el notebook 02
    !python scripts/select_checkpoint.py --corrida results/cisutac/bloque_a/entrenamientos/holes_A3_tiles640_neg1_yolo26n
El comando hereda REWEAR_DATA_DIR del notebook. Si la corrida está en /kaggle/input (sólo lectura), agregar
--salida /kaggle/working/<carpeta>. Lo que se escribe en reports/ hay que descargarlo del Output de Kaggle.

Para guardar un checkpoint cada 5 épocas, pasar save_period=5 a `cb.train`, que lo envía a Ultralytics y lo
guarda en corrida.json. Hay que usar un nombre de corrida nuevo: si no, `cb.train` encuentra el best.pt anterior
y reutiliza esa corrida sin entrenar.
- 02b: NOMBRE_CORRIDA = 'holes_A3_tiles640_neg1_ext100_s5' y cb.train(..., save_period=5).
- 02: en entrenar_tiles, cb.train(..., save_period=5) y un nombre con sufijo, por ejemplo
  entrenar_tiles(f"A3_tiles{TILE}_neg{NEG_POR_POS:g}_s5", ...).

Ejemplos:
  python scripts/select_checkpoint.py --corrida results/cisutac/bloque_a/entrenamientos/holes_A3_tiles640_neg1
  python scripts/select_checkpoint.py --corrida results/cisutac/entrenamientos/holes_yolo11n_640 --prueba 3
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cisutac_baseline as cb  # noqa: E402

SPLIT = "valid"          # único split que evalúa este script: test se abre una sola vez en el issue #7
CONF = 0.001             # confianza mínima de las predicciones guardadas, igual que CONF_EVAL en los notebooks
DECIMALES_EMPATE = 3     # mAP@0.5 iguales a 3 decimales cuentan como empate
REPORTES = ROOT / "reports" / "cisutac" / "seleccion_checkpoint"
CRITERIO = (f"mayor mAP@0.5 por foto en valid (pycocotools, coordenadas de la original); empate al redondear a "
            f"{DECIMALES_EMPATE} decimales: gana la época más temprana")
# Métricas de Ultralytics que identifican la época de un checkpoint en results.csv.
METRICAS_ULTRALYTICS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
# Ruta del data.yaml: foto completa (cisutac_holes/data.yaml) o tiles (cisutac_holes_tiles/t640_o20/data_*.yaml).
PATRON_DATA = re.compile(r"cisutac_(holes|spots)(?:_tiles/t(\d+)_o(\d+))?/[^/]+\.ya?ml$")
COLUMNAS = ["archivo", "epoca", "mAP@0.5", "AP_small", "recall", "umbral_confianza", "mAP50_ultralytics", "elegido",
            "evaluacion"]

CONTROLES, AVISOS = [], []


def exigir(condicion, mensaje):
    if not condicion:
        raise SystemExit(f"ERROR: {mensaje}")


def control(condicion, mensaje):
    exigir(condicion, f"control fallido: {mensaje}")
    CONTROLES.append(mensaje)


def relativa(ruta):
    """Ruta relativa a la raíz del repo si está adentro; si no, tal cual (p. ej. /kaggle/input/...)."""
    try:
        return Path(ruta).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(ruta)


def leer_configuracion(carpeta, tambien_completa):
    """(dataset, argumentos de inferencia para cb.evaluate, archivo de origen) de una corrida."""
    if (carpeta / "corrida.json").is_file():
        origen = "corrida.json"
        config = json.loads((carpeta / origen).read_text(encoding="utf-8"))
    elif (carpeta / "args.yaml").is_file():
        import yaml

        origen = "args.yaml"
        config = yaml.safe_load((carpeta / origen).read_text(encoding="utf-8"))
    else:
        exigir(False, f"{carpeta} no tiene corrida.json ni args.yaml: no es una carpeta de entrenamiento")
    exigir("data" in config and "imgsz" in config, f"{origen} no tiene 'data' e 'imgsz'")
    m = PATRON_DATA.search(str(config["data"]).replace("\\", "/"))
    exigir(m is not None, f"no se reconoce el dataset ni el modo en {origen}: data = {config['data']}")
    dataset, tile, solapamiento = m.groups()
    imgsz = int(config["imgsz"])
    if tile:
        control(imgsz == int(tile), f"configuración ({origen}): tiles de {tile} px con {solapamiento} % de "
                                    f"solapamiento, imgsz {imgsz} igual al tile")
        inferencia = {"mode": "tiles", "imgsz": imgsz, "tile": int(tile), "overlap": int(solapamiento) / 100,
                      "also_full": tambien_completa, "conf": CONF}
    else:
        exigir(not tambien_completa, "--tambien-completa sólo se usa en corridas entrenadas con tiles")
        control(True, f"configuración ({origen}): foto completa con imgsz {imgsz}")
        inferencia = {"mode": "completa", "imgsz": imgsz, "conf": CONF}
    return dataset, inferencia, origen


def detalle_inferencia(inferencia):
    """Mismo sufijo que usa evaluar_valid en el notebook 02: completa640, tiles640_o20 o tiles640_o20+completa."""
    if inferencia["mode"] == "completa":
        return f"completa{inferencia['imgsz']}"
    return (f"tiles{inferencia['tile']}_o{round(100 * inferencia['overlap'])}"
            + ("+completa" if inferencia["also_full"] else ""))


def leer_results_csv(carpeta):
    ruta = carpeta / "results.csv"
    exigir(ruta.is_file(), f"falta {ruta}: hace falta para ubicar la época de best.pt y last.pt")
    with ruta.open(encoding="utf-8") as archivo:
        return [{k.strip(): v.strip() for k, v in fila.items()} for fila in csv.DictReader(archivo)]


def epoca_del_checkpoint(ruta, filas):
    """(época en la numeración de results.csv, cómo se dedujo) del checkpoint `ruta`.

    Al terminar el entrenamiento Ultralytics deja epoch = -1 en best.pt y last.pt; best.pt conserva en
    train_metrics las métricas de la época en que se guardó, que se buscan en results.csv.
    """
    import torch

    ckpt = torch.load(ruta, map_location="cpu", weights_only=False)
    guardada = ckpt.get("epoch", -1)
    metricas = ckpt.get("train_metrics") or {}
    del ckpt
    numero = re.fullmatch(r"epoch(\d+)", ruta.stem)
    if numero:
        n = int(numero.group(1))
        exigir(guardada in (-1, n), f"{ruta.name} dice ser de la época {guardada} (Ultralytics, desde 0)")
        return n + 1, "nombre del archivo (Ultralytics numera desde 0)"
    if guardada >= 0:  # entrenamiento cortado: el checkpoint no pasó por strip_optimizer
        return guardada + 1, "campo epoch del checkpoint (entrenamiento sin terminar)"
    if ruta.stem == "last":
        return int(filas[-1]["epoch"]), "última fila de results.csv"
    exigir(all(k in metricas for k in METRICAS_ULTRALYTICS), f"{ruta.name} no tiene train_metrics de Ultralytics")
    coinciden = [int(f["epoch"]) for f in filas
                 if all(math.isclose(float(f[k]), metricas[k], rel_tol=1e-4, abs_tol=1e-6) for k in METRICAS_ULTRALYTICS)]
    exigir(coinciden, f"las métricas de {ruta.name} no aparecen en results.csv")
    # Ultralytics sobrescribe best.pt cuando una época empata la mejor: vale la última que coincide.
    return coinciden[-1], "métricas de Ultralytics del checkpoint en results.csv"


def buscar_checkpoints(carpeta):
    pesos = carpeta / "weights"
    archivos = [p for p in pesos.glob("*.pt") if p.stem in ("best", "last") or re.fullmatch(r"epoch\d+", p.stem)]
    exigir(archivos, f"no hay best.pt, last.pt ni epochN.pt en {pesos}")
    return sorted(archivos)


def sha256(ruta):
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def esperado_en_valid(dataset, prueba):
    """Fotos y cajas reales que tiene que usar cada evaluación (las primeras `prueba` fotos en modo prueba)."""
    gt = cb.load_ground_truth(dataset, SPLIT)
    imagenes = gt["images"][:prueba] if prueba else gt["images"]
    ids = {im["id"] for im in imagenes}
    return len(imagenes), sum(a["image_id"] in ids for a in gt["annotations"])


def elegir(filas):
    return min(filas, key=lambda f: (-round(f["mAP@0.5"], DECIMALES_EMPATE), f["epoca"], f["archivo"]))


def comparar_con_notebook(corrida, detalle, filas):
    """Si el notebook ya evaluó el best.pt de esta corrida, el script tiene que medir lo mismo."""
    previa = cb.RESULTS / "evaluaciones" / f"{corrida}__valid_{detalle}" / "metricas.json"
    best = next((f for f in filas if f["archivo"] == "weights/best.pt"), None)
    if best is None or not previa.is_file():
        return
    m = json.loads(previa.read_text(encoding="utf-8"))
    if (Path(m["pesos"]).name == "best.pt" and m["imagenes"] == best["imagenes"]
            and abs(m["mAP@0.5"] - best["mAP@0.5"]) < 1e-3):
        control(True, f"best.pt da el mismo mAP@0.5 que la evaluación del notebook ({m['mAP@0.5']:.3f})")
    else:
        AVISOS.append(f"best.pt da mAP@0.5 {best['mAP@0.5']:.3f} y la evaluación del notebook "
                      f"{relativa(previa)} da {m['mAP@0.5']:.3f} ({m['imagenes']} fotos): revisar si es la misma corrida")


def escribir(destino, nombre_base, filas, resumen):
    destino.mkdir(parents=True, exist_ok=True)
    with (destino / f"{nombre_base}.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS, extrasaction="ignore")
        escritor.writeheader()
        for f in filas:
            escritor.writerow({k: (round(v, 6) if isinstance(v, float) else "" if v is None else v)
                               for k, v in f.items()})
    (destino / f"{nombre_base}.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Elige el checkpoint de una corrida por mAP@0.5 por foto en valid.")
    parser.add_argument("--corrida", type=Path, required=True,
                        help="carpeta de la corrida de entrenamiento (con weights/ y corrida.json o args.yaml)")
    parser.add_argument("--tambien-completa", action="store_true",
                        help="en corridas con tiles, sumar también la predicción de la foto completa")
    parser.add_argument("--prueba", type=int, nargs="?", const=3, default=None, metavar="N",
                        help="evaluar sólo las primeras N fotos de valid (3 si no se indica); no son resultados")
    parser.add_argument("--reevaluar", action="store_true", help="recalcular aunque existan evaluaciones guardadas")
    parser.add_argument("--salida", type=Path, help="carpeta para el CSV y el JSON (por defecto, la de la corrida)")
    args = parser.parse_args()
    exigir(args.prueba is None or args.prueba > 0, "--prueba necesita un número de fotos mayor que 0")

    carpeta = args.corrida.expanduser().resolve()
    exigir(carpeta.is_dir(), f"no existe la carpeta {args.corrida}")
    corrida = carpeta.name
    dataset, inferencia, origen = leer_configuracion(carpeta, args.tambien_completa)
    detalle = detalle_inferencia(inferencia)
    sufijo = f"_prueba{args.prueba}" if args.prueba else ""
    n_imagenes, n_cajas = esperado_en_valid(dataset, args.prueba)
    resultados = leer_results_csv(carpeta)
    mAP50_por_epoca = {int(f["epoch"]): float(f["metrics/mAP50(B)"]) for f in resultados}

    checkpoints = []
    for ruta in buscar_checkpoints(carpeta):
        epoca, como = epoca_del_checkpoint(ruta, resultados)
        checkpoints.append((epoca, ruta))
        control(epoca in mAP50_por_epoca, f"{ruta.name}: época {epoca}, deducida de: {como}")
    checkpoints.sort(key=lambda par: (par[0], par[1].name))

    filas = []
    for epoca, ruta in checkpoints:
        nombre = f"{corrida}__{ruta.stem}__{SPLIT}_{detalle}{sufijo}"
        print(f"Evaluando {ruta.name} (época {epoca}) en {SPLIT}...", flush=True)
        m = cb.evaluate(nombre, dataset, ruta, split=SPLIT, limit=args.prueba, reevaluate=args.reevaluar,
                        **inferencia)
        exigir(m["mAP@0.5"] is not None, f"{ruta.name}: mAP@0.5 no definido")
        filas.append({"archivo": f"weights/{ruta.name}", "epoca": epoca, "mAP@0.5": m["mAP@0.5"],
                      "AP_small": m["AP_small"], "recall": m["recall"], "umbral_confianza": m["umbral_confianza"],
                      "mAP50_ultralytics": mAP50_por_epoca[epoca], "elegido": "no",
                      "evaluacion": relativa(cb.RESULTS / "evaluaciones" / nombre / "metricas.json"),
                      "split": m["split"], "imagenes": m["imagenes"], "cajas_reales": m["cajas_reales"],
                      "ruta": ruta})
    control(all(f["split"] == SPLIT for f in filas), f"las {len(filas)} evaluaciones son de {SPLIT}")
    control(all((f["imagenes"], f["cajas_reales"]) == (n_imagenes, n_cajas) for f in filas),
            f"todas usan las mismas {n_imagenes} fotos y {n_cajas} cajas reales de {dataset} {SPLIT}")
    if not args.prueba:
        comparar_con_notebook(corrida, detalle, filas)

    elegido = elegir(filas)
    elegido["elegido"] = "sí"
    best = next((f for f in filas if f["archivo"] == "weights/best.pt"), None)
    resumen = {
        "corrida": corrida, "carpeta": relativa(carpeta), "dataset": dataset, "split": SPLIT,
        "modo_prueba": bool(args.prueba), "imagenes": n_imagenes, "cajas_reales": n_cajas,
        "inferencia": inferencia, "configuracion_leida_de": origen, "criterio": CRITERIO,
        "checkpoints_evaluados": len(filas),
        "elegido": {"archivo": elegido["archivo"], "epoca": elegido["epoca"], "sha256": sha256(elegido["ruta"]),
                    **{k: elegido[k] for k in ("mAP@0.5", "AP_small", "recall", "umbral_confianza", "evaluacion")}},
        "best_pt_ultralytics": best and {"epoca": best["epoca"], "mAP@0.5": best["mAP@0.5"]},
        "advertencia": [
            f"El mAP@0.5 de valid del elegido es optimista: es el máximo entre {len(filas)} checkpoints medidos "
            f"sobre las mismas fotos. El número que vale es el de test (issue #7).",
            f"Con pocas fotos en valid ({n_imagenes} en {dataset.capitalize()}) y {len(filas)} checkpoints, la "
            f"elección puede depender del ruido; por eso el número que vale es el de test.",
        ],
    }
    if args.prueba:
        resumen["advertencia"].insert(0, f"PRUEBA con {args.prueba} fotos: estos números no son resultados.")

    nombre_base = f"seleccion_checkpoint{'_prueba' if args.prueba else ''}"
    destinos = [(args.salida or carpeta).expanduser().resolve()]
    if not args.prueba:
        destinos.append(REPORTES / corrida)
    for destino in destinos:
        try:
            escribir(destino, nombre_base, filas, resumen)
        except OSError as error:
            exigir(False, f"no se pudo escribir en {destino} ({error}); usar --salida con una carpeta escribible")

    print(f"\n{'PRUEBA (no son resultados) — ' if args.prueba else ''}{corrida}: {len(filas)} checkpoints en "
          f"{SPLIT}, {n_imagenes} fotos, inferencia {detalle}")
    print(f"{'archivo':22s} {'época':>5s} {'mAP@0.5':>8s} {'AP_small':>8s} {'recall':>7s} {'umbral':>6s} "
          f"{'mAP50 Ultralytics':>17s}")
    for f in filas:
        ap_small = "—" if f["AP_small"] is None else f"{f['AP_small']:.3f}"
        print(f"{f['archivo']:22s} {f['epoca']:5d} {f['mAP@0.5']:8.3f} {ap_small:>8s} {f['recall']:7.3f} "
              f"{f['umbral_confianza']:6.2f} {f['mAP50_ultralytics']:17.3f}" + ("  ← elegido" if f is elegido else ""))
    print("\nControles:")
    for mensaje in CONTROLES:
        print(f"  [ok] {mensaje}")
    for mensaje in AVISOS:
        print(f"  [aviso] {mensaje}")
    print("\nEscrito en:")
    for destino in destinos:
        print(f"  {relativa(destino / nombre_base)}.csv y .json")


if __name__ == "__main__":
    main()
