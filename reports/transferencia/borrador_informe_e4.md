# Borrador para el informe final: transferencia entre dominios (E4)

> Texto para adaptar al informe (máximo 8 páginas, estilo IEEE). Todavía **sin resultados**: los marcadores `[…]` se completan con el notebook 04 una vez evaluado el test. Las referencias entre corchetes son marcadores.

## Método

Para medir qué se pierde al llevar un detector entrenado con telas industriales a fotos de prendas usadas, entrenamos un YOLO11n [YOLO] con FabriD v3 [FabriD], usando sólo las clases agujero (*hole*) y mancha (*stain*), y lo aplicamos sin ajuste a CISUTAC [CISUTAC]. Las predicciones de agujero se evalúan contra CISUTAC Holes y las de mancha, contra CISUTAC Spots.

En FabriD usamos el split agrupado por captura, que evita que derivados de una misma foto queden a ambos lados, y excluimos los 19 casos de normalidad ambigua. Las imágenes que además contienen marcas de aguja o hilos faltantes quedan afuera, para no enseñar como fondo defectos reales. La receta de entrenamiento es la misma que la del detector entrenado en CISUTAC (E1), así la diferencia entre ambos son los datos de entrenamiento.

## Escala

El cambio de dominio no es sólo de apariencia. FabriD son primeros planos: el agujero mediano mide 32 px y la mancha mediana, 60 px en las imágenes de 640 px con las que se entrena. En CISUTAC, al reducir la prenda completa a 640 px, esos lados caen a 4,2 y 3,0 px.

Recorrer la foto original con *tiles* cambia esa escala. Con tiles de 640 px, el agujero mediano llega al modelo con 25 px, cerca de FabriD; las manchas recién se acercan con tiles de 320 px ampliados al doble (36 px). Por eso evaluamos la foto completa y tiles de 640 y 320 px, con 20 % de solapamiento y fusión por NMS [SAHI], y elegimos el tamaño en validación.

## Evaluación

Usamos las mismas métricas que en el E1, en coordenadas de la foto original: mAP@0.5, mAP@0.5:0.95, AP_small, precisión, recall y F1, y recall por rango de tamaño. Reportamos dos comparaciones:

1. **El mismo detector en el test de FabriD y en el de CISUTAC.** Mide cuánto cae al salir de su dominio, aunque mezcla el cambio de tela, de tipo de foto y de escala.
2. **El detector de FabriD frente al entrenado en CISUTAC, sobre el mismo test.** Aísla el efecto de los datos de entrenamiento.

Los pesos nunca se ajustan con CISUTAC. El umbral de confianza y el tamaño de tile se eligen en su conjunto de validación, igual que en el E1, y el test se usa una sola vez.

## Resultados

> Completar con el notebook 04. Tabla sugerida, una por tipo de defecto: filas = detector (E1 CISUTAC, E4 FabriD) × inferencia (foto completa, mejor tile); columnas = mAP@0.5, AP_small, recall < 16 px y recall 16–32 px.

[…]

## Limitaciones

- CISUTAC no incluye prendas sanas, así que los falsos positivos sólo se miden sobre prendas con defectos.
- La equivalencia *stain* → mancha es aproximada.
- FabriD conserva sesgos de lote y resolución.
- Elegir el umbral y el tile en validación de CISUTAC hace que la transferencia no sea completamente «a ciegas».
- Una sola semilla.

## Si el grupo aprueba el anexo con FabricSpotDefect

Una segunda fuente industrial, FabricSpotDefect [FSD], permite repetir la evaluación para manchas con otra escala de origen (mancha mediana de 36 px al entrenar, frente a 60 px en FabriD). Antes de usarla corregimos 228 fotos cuyas cajas estaban giradas 90° respecto del archivo y movimos a validación 8 fotos de entrenamiento que eran copias, o la misma pieza, de fotos de validación.
