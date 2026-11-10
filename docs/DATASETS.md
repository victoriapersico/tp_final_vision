# Datasets de ReWear AI

Los datos originales son públicos pero **no se versionan en Git**. Todos se ubican debajo de la carpeta indicada por `REWEAR_DATA_DIR`; si la variable no existe, se usa `data/` en la raíz.

Los tamaños son aproximados y pueden cambiar levemente por metadatos del sistema. Conviene reservar espacio adicional para conservar comprimidos y extracción.

## Estructura esperada

```text
data/
├── cisutac_holes/
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   └── README.md
├── cisutac_spots/
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   └── README.md
├── fabric_spot_defect/
│   └── FabricSpotDefect An Annotated Dataset for Identifying Spot  Defects in Different Fabric Types/
│       └── FabricSpotDefect/{Orginal,Augmented}/
├── fabrid_v3/
│   ├── original/{defect free,hole Defect,missing yarn Defect,Needle Mark Defect,stain Defect}/
│   └── annotations_export/{train,valid,test}/{images,labels}/
├── raw_fabrid/
│   └── extracted/RAW_FABRID/{images,masks,MVTec}/
└── rise_samples/             # sólo muestras; corpus completo opcional
```

No corregir nombres como `Orginal` ni las mayúsculas de FabriD: son las rutas reales usadas por el EDA.

## 1. CISUTAC Holes

- Fuente: [wargoninnovation/Garment_condition_holes](https://huggingface.co/datasets/wargoninnovation/Garment_condition_holes).
- Versión: revisión actual del repositorio de Hugging Face.
- Licencia: CC BY 4.0; atribuir CISUTAC Project / Wargön Innovation.
- Tamaño local observado: aproximadamente 231 MB.
- Contenido: 761 imágenes de prendas y 1.497 instancias anotadas; `train` y `valid`.

Descarga automática:

```bash
python scripts/download_datasets.py holes
```

Equivalente manual:

```bash
hf download wargoninnovation/Garment_condition_holes \
  --repo-type dataset --local-dir data/cisutac_holes --max-workers 2
```

La carpeta oculta `.cache/huggingface/` permite reanudar; no borrarla durante una descarga parcial.

## 2. CISUTAC Spots

- Fuente: [wargoninnovation/Garment_condition_spots](https://huggingface.co/datasets/wargoninnovation/Garment_condition_spots).
- Versión: revisión actual del repositorio de Hugging Face.
- Licencia: CC BY 4.0; atribuir CISUTAC Project / Wargön Innovation.
- Tamaño local observado: aproximadamente 237 MB.
- Contenido: 906 imágenes y 4.777 instancias; `train` y `valid`.

```bash
python scripts/download_datasets.py spots
```

## 3. FabricSpotDefect

- Fuente y versión 1: [Mendeley Data](https://data.mendeley.com/datasets/6574nhzm8x/1).
- Licencia: CC BY 4.0; citar el dataset y su artículo asociado.
- Archivo observado: `FabricSpotDefect An Annotated Dataset for Identifying Spot  Defects in Different Fabric Types.zip`.
- Espacio observado: unos 898 MB comprimidos y 920 MB extraídos; reservar aproximadamente 2 GB.

Procedimiento:

1. Descargar el ZIP desde “Download all” en Mendeley.
2. Crear `data/fabric_spot_defect/`.
3. Extraer el ZIP dentro de esa carpeta, conservando el directorio superior original.
4. Verificar que exista `.../FabricSpotDefect/Orginal/{train,valid,test}`.

La descarga se deja manual porque Mendeley usa enlaces de archivo internos que pueden cambiar. No mezclar `Augmented/COCO` y `Augmented/YOLOv8` como si fueran imágenes originales diferentes: son exportaciones alternativas.

## 4. FabriD versión 3

- Fuente exacta: [FabriD v3 en Mendeley Data](https://data.mendeley.com/datasets/vncdwps24r/3).
- Licencia: CC BY 4.0.
- Archivos observados:
  - `FabriD_v3_annotations.zip`
  - `defect_free.zip`
  - `hole_defect.zip`
  - `missing_yarn_defect.zip`
  - `needle_mark_defect.zip`
  - `stain_defect.zip`
- Tamaño oficial agregado aproximado: 6,26 GB comprimidos. La copia local con ZIPs y extracción ocupa alrededor de 13 GB; reservar 14 GB.

Procedimiento manual:

1. En la página de la **versión 3**, descargar los seis archivos indicados.
2. Guardarlos temporalmente en `data/fabrid_v3/`.
3. Extraer los cinco ZIP de categorías dentro de `data/fabrid_v3/original/`, manteniendo los nombres de carpetas mostrados en la estructura esperada.
4. Extraer `FabriD_v3_annotations.zip` en `data/fabrid_v3/annotations_export/`.
5. Comprobar `train`, `valid`, `test`, `data.yaml` y 2.407 imágenes tanto en originales como en el export anotado.

No usar versiones 1 o 2 por error. Antes de entrenar deben resolverse las 14 posibles capturas compartidas entre splits documentadas en `reports/dataset_audit.md`.

## 5. RAW-FABRID

- Fuente y versión 1: [Mendeley Data](https://data.mendeley.com/datasets/db6g85xsyg/1).
- Licencia: CC BY 4.0.
- Archivo observado: `RAW_FABRID_v1.zip`.
- SHA-256 oficial verificado: `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`.
- Tamaño: aproximadamente 3,85 GB comprimidos y 8,2 GB conservando ZIP y extracción; reservar 9 GB.

Procedimiento manual:

1. Descargar `RAW_FABRID_v1.zip` desde Mendeley.
2. Guardarlo en `data/raw_fabrid/`.
3. Verificar el checksum antes de extraer:

   ```bash
   shasum -a 256 data/raw_fabrid/RAW_FABRID_v1.zip
   ```

4. Extraer dentro de `data/raw_fabrid/extracted/`.
5. Confirmar que exista `extracted/RAW_FABRID/MVTec/train/good` y los CSV/JSON del directorio raíz.

El EDA detectó 204 máscaras frente a 202 imágenes marcadas como defectuosas en el CSV. No corregir los originales; documentar cualquier decisión derivada aparte.

## 6. RISE front-only RGB (opcional)

- Fuente: [fnauman/fashion-second-hand-front-only-rgb](https://huggingface.co/datasets/fnauman/fashion-second-hand-front-only-rgb).
- Licencia: CC BY 4.0.
- Tamaño remoto verificado: 2.843.670.317 bytes (2,84 GB decimal), 14 archivos Parquet.
- Contenido: 31.638 registros; 28.248 train y 3.390 test.

El corpus completo todavía no forma parte del flujo reproducible principal. Sólo se inspeccionaron metadatos y tres muestras en `data/rise_samples/`. No descargarlo por defecto ni interpretar campos nulos de manchas/agujeros como ausencia confirmada de defectos.

Si el equipo autoriza su uso, puede descargarse explícitamente con:

```bash
hf download fnauman/fashion-second-hand-front-only-rgb \
  --repo-type dataset --local-dir data/rise --max-workers 2
```

## Comprobación final

Después de preparar los datos:

```bash
python scripts/generate_eda_summary.py
jupyter lab notebooks/01_eda_datasets_rewear_ai.ipynb
```

Los conteos regenerados deben compararse con `reports/eda_summary.json`. Diferencias grandes indican una versión, extracción o ruta incorrecta.
