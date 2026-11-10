"""Descarga reanudable de los datasets CISUTAC desde Hugging Face.

Los datasets alojados en Mendeley se documentan como descarga manual en
docs/DATASETS.md para no depender de enlaces internos inestables.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


DATASETS = {
    "holes": ("wargoninnovation/Garment_condition_holes", "cisutac_holes"),
    "spots": ("wargoninnovation/Garment_condition_spots", "cisutac_spots"),
}
BACKOFF_SECONDS = (60, 120, 300, 600, 600, 600)


def data_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    return Path(os.environ.get("REWEAR_DATA_DIR", root / "data")).expanduser().resolve()


def looks_complete(target: Path) -> bool:
    return (target / "train" / "images").is_dir() and any((target / "train" / "images").iterdir())


def hf_command(repo: str, target: Path) -> list[str]:
    if shutil.which("hf"):
        return ["hf", "download", repo, "--repo-type", "dataset", "--local-dir", str(target), "--max-workers", "2"]
    if shutil.which("huggingface-cli"):
        return ["huggingface-cli", "download", repo, "--repo-type", "dataset", "--local-dir", str(target), "--max-workers", "2"]
    raise RuntimeError("No se encontró `hf` ni `huggingface-cli`. Instalá huggingface_hub por separado.")


def download(name: str) -> None:
    repo, folder = DATASETS[name]
    target = data_root() / folder
    if looks_complete(target):
        print(f"{name}: ya existe una descarga completa en {target}; se omite.")
        return
    target.mkdir(parents=True, exist_ok=True)
    command = hf_command(repo, target)
    for attempt, delay in enumerate(BACKOFF_SECONDS, start=1):
        print(f"{name}: intento {attempt}/{len(BACKOFF_SECONDS)}")
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            print(f"{name}: descarga finalizada en {target}")
            return
        if attempt < len(BACKOFF_SECONDS):
            print(f"Falló el intento {attempt}; nuevo intento en {delay // 60} minuto(s).")
            time.sleep(delay)
    raise RuntimeError(f"{name}: la descarga falló después de {len(BACKOFF_SECONDS)} intentos; los parciales quedan en {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga datasets CISUTAC sin repetir archivos completos.")
    parser.add_argument("dataset", choices=[*DATASETS, "all"])
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else [args.dataset]
    for name in selected:
        download(name)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
