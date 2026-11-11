# Borrador para el informe final: CISUTAC

> Texto para adaptar al informe (máximo 8 páginas, estilo IEEE). Las cifras salen de [`eda_cisutac.md`](eda_cisutac.md) y se regeneran con `scripts/generate_cisutac_eda.py`. Las referencias entre corchetes son marcadores a completar.

## Datos: prendas usadas (CISUTAC)

Para la detección supervisada en prendas usadas usamos CISUTAC Holes y Spots [CISUTAC], dos conjuntos públicos (CC BY 4.0) de fotografías de prendas posconsumo con cajas anotadas de agujeros y manchas. El repositorio contiene 761 y 906 archivos, pero el historial de versiones muestra que la mayoría de las fotos aparece dos veces: la original de 2160×3840 px y una copia redimensionada a 640×640. Por eso trabajamos sólo con las originales: 407 fotos con 785 agujeros y 486 fotos con 2.498 manchas. Ningún conjunto incluye prendas sin defectos ni máscaras completas, así que no permiten medir falsos positivos en prendas sanas ni métricas de anomalías por píxel.

## Escala de los defectos

Los defectos son pequeños aun a resolución completa. Medimos cada caja por su lado equivalente √(ancho·alto). La mediana es de 25 px en agujeros y 18 px en manchas, y el 58 % y el 67 %, respectivamente, son *small* según COCO (área < 32² px). Al redimensionar la foto completa a 640 px de lado mayor, la entrada por defecto de los detectores YOLO, la mediana cae a 4,2 y 3,0 px. Además, el 71 % y el 77 % de los defectos ocupan menos que una celda de la salida más fina del detector (stride 8).

La sensibilidad del IoU agrava el problema. En una caja de 4 px, un corrimiento de 1 px reduce el IoU a 0,39. Simulando con las cajas reales un detector perfecto que sólo se corre 1 px en la imagen de entrada, el recall a IoU 0,5 sería de 39 % (agujeros) y 30 % (manchas) a 640 px, frente a 98 % y 88 % a resolución nativa. Recorrer la foto con tiles de 640 px y 20 % de solapamiento, como en SAHI [SAHI], conserva el tamaño original de los defectos. El costo es de 32 tiles por imagen, unas 32 veces más píxeles que una pasada a 640, y el 97 % de los defectos queda entero en al menos un tile.

## Particiones y evaluación

El split oficial (train/valid) separa por código de captura, pero comparte días de fotografía entre particiones. Además, identificamos visualmente al menos una prenda refotografiada con otro código en ambos lados. El tamaño de los defectos también cambia con la fecha de captura. Por eso definimos particiones agrupadas por día (train/valid/test, alrededor de 74/11/15 % de las fotos) y elegimos, entre todas las combinaciones de días, la que mejor conserva la distribución de tamaños del conjunto completo. La selección usa sólo las etiquetas, nunca resultados de modelos.

Evaluamos agujeros y manchas por separado, siempre en coordenadas de la imagen original, con:

- precisión, recall y F1 (IoU ≥ 0,5, con el umbral de confianza elegido en validación);
- mAP@0.5 y mAP@0.5:0.95;
- AP_small;
- recall por rango de tamaño.

Reportamos además el tiempo por imagen e intervalos de confianza por bootstrap de imágenes, dado el tamaño reducido del test (64 y 66 fotos).

## Baseline a resolución reducida

> Cifras preliminares, medidas en validación. Reemplazarlas por las del test cuando el grupo apruebe el split y el modelo final.

Un YOLO11n preentrenado en COCO y ajustado con la foto completa a 640 px no encontró ningún defecto de menos de 32 px en validación. Logró mAP@0.5 de 0,032 en agujeros y 0,019 en manchas, con AP_small de 0, y sus aciertos se limitan a defectos grandes. Aplicar ese mismo modelo con tiles no lo mejora: en los tiles los defectos se ven unas 6 veces más grandes que en entrenamiento, y el modelo confunde con defectos ojalillos, lunares y costuras de ese tamaño. Por eso proponemos que el detector con tiles se entrene con recortes de la misma escala con la que infiere.

## Figuras sugeridas

- `cis4_zoom_resize.png`: la misma zona de la prenda a cada resolución (una figura de media columna basta).
- `cis3_distribucion_tamanos.png` o la tabla de CIS-4: tamaño de los defectos antes y después del resize.
- `baseline_ejemplos_holes.jpg`: en qué se equivoca un detector entrenado a 640 (sólo si sobra espacio).
