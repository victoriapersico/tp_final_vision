# Guía conceptual para el oral y el informe

El profesor pidió que sepamos explicar **cómo y por qué funcionan** los modelos, sin la matemática detallada. Esta guía resume cada pieza del trabajo con esa idea, usando nuestros números como ejemplo. Los resultados citados son de **valid** salvo que se diga otra cosa. Las fuentes de cada cifra están en `notebooks/ejecutados/` y en los reportes de `reports/`.

## 1. El problema: defectos que desaparecen al achicar la foto

- Una foto de CISUTAC mide 2160×3840 px y el agujero mediano mide **25 px** de lado.
- Los detectores reciben la imagen achicada a **640 px** de lado mayor: el agujero pasa a medir **4,2 px**.
- Una CNN de detección "mira" la imagen en grillas. La más fina de YOLO tiene un paso (*stride*) de **8 px**: cada celda resume un cuadrado de 8×8 px de la entrada. Un defecto más chico que una celda deja muy poca señal.
- Además, en una caja chica, un error mínimo cambia mucho el IoU (cuánto se superponen la caja predicha y la real). En una caja de 4 px, correrse **1 px** ya baja el IoU a 0,39, debajo del 0,5 que se exige para contar un acierto. Por eso, aun un detector perfecto que se equivoca un píxel recuperaría a 640 px sólo el 39 % de los agujeros (98 % a resolución original).

**Idea para el oral:** el problema no es "el modelo es malo", es que la información se pierde antes de llegar al modelo.

## 2. YOLO: detección supervisada en una sola pasada

**Qué hace:** recibe una imagen y devuelve cajas con una confianza ("acá hay un agujero, con 0,8 de seguridad").

**Cómo funciona:**
1. **Backbone (CNN):** convoluciones apiladas que transforman la imagen en *mapas de características*. Las primeras capas detectan bordes y texturas; las profundas, formas más complejas. Cada vez que reducen la resolución, el campo receptivo crece.
2. **Neck:** combina mapas de distintas resoluciones, para que la información de alto nivel (qué es) llegue a los mapas finos (dónde está).
3. **Cabezas de detección a tres escalas,** con stride 8, 16 y 32. Cada celda de cada grilla predice si hay un objeto, de qué clase y cómo es su caja. Los objetos chicos dependen de la grilla más fina (stride 8).
4. **Postproceso:** muchas celdas vecinas predicen el mismo objeto. En YOLO11, la **NMS** (*non-maximum suppression*) se queda con la caja de mayor confianza y descarta las que se le superponen mucho.

**Aprendizaje por transferencia:** partimos de pesos entrenados en COCO (80 clases de objetos comunes) y los ajustamos con nuestras cajas. El backbone ya sabe extraer bordes y texturas; sólo tiene que aprender qué combinación significa "agujero".

**YOLO26** (sugerencia del profesor):
- se entrena e infiere *end-to-end*, sin NMS: el modelo aprende a dar una sola caja por objeto;
- simplifica cómo predice los bordes de la caja (no usa DFL);
- su grilla más fina **sigue siendo de stride 8**, así que no resuelve por sí solo el problema de escala de la foto completa.

En el notebook 02 se prueba cambiando `MODELO = "yolo26n.pt"`.

## 3. *Tiles*: no achicar la foto

**Idea:** en vez de achicar la foto entera, se la recorre con recortes (*tiles*) de 640×640 px a resolución original, con 20 % de solapamiento para que un defecto en el borde de un recorte aparezca entero en el vecino. Cada foto da 32 tiles.

**Inferencia (SAHI):** se corre el modelo en cada tile, se llevan las cajas a coordenadas de la foto original y se fusionan con NMS las que se repiten entre tiles vecinos.

**Por qué hay que entrenar también con tiles:** un modelo entrenado con la foto achicada aprendió defectos de ~4 px. En un tile los ve ~6 veces más grandes y no los reconoce: aplicado con tiles, el A0 baja de mAP@0.5 0,042 a 0,005 (ablación A2). Entrenado con tiles (A3), a la misma escala con la que infiere, sube a **0,131**.

**Resultado en test (Spots, evaluado una sola vez):** con tiles, el mAP@0.5 pasa de 0,002 a 0,097 y el recall, de 1,0 % a 17,7 %. De las 140 manchas de menos de 16 px, la foto completa no encuentra ninguna y los tiles encuentran 22.

**Decisiones de diseño del entrenamiento con tiles:**
- **tiles sin defectos (negativos):** si no hay, el modelo marca defectos en todos lados; si hay demasiados, los ignora. Probamos 0, 1 y 3 por cada tile con defectos y el mejor fue 1;
- **defectos cortados por el borde del tile:** se conservan si queda al menos la mitad;
- **mismo presupuesto de entrenamiento** (imágenes vistas) que la foto completa, para que la comparación sea justa.

**Costo:** unas 12 veces más tiempo por foto que la foto completa a 640 (0,45 s contra 0,04 s por foto en Kaggle).

## 4. Cómo se mide un detector

- **IoU:** área de la intersección sobre área de la unión de dos cajas. Con IoU ≥ 0,5, una predicción cuenta como acierto.
- **Precisión:** de lo que el modelo marcó, cuánto era un defecto real. **Recall:** de los defectos reales, cuántos encontró. Dependen del **umbral de confianza**: con un umbral bajo se encuentra más, pero aparecen más falsas alarmas. Elegimos el umbral que maximiza F1 en valid.
- **AP (*average precision*):** se ordenan las predicciones por confianza y se recorre la curva precisión-recall. AP es el área bajo esa curva, así que no depende de un umbral. **mAP@0.5** usa IoU 0,5; **mAP@0.5:0.95** promedia IoU de 0,5 a 0,95 y premia las cajas bien encuadradas.
- **AP_small:** el AP sólo sobre cajas de menos de 32×32 px. Es nuestra métrica clave.
- Todo se calcula **en coordenadas de la foto original**, para que la foto completa y los tiles se midan igual.
- **Intervalos por bootstrap:** se re-muestrean las fotos de valid muchas veces para ver cuánto podría cambiar el resultado con otras fotos. Con 43 fotos y 71 agujeros, los intervalos son anchos.

## 5. ¿Por qué el mAP es bajo? Análisis de errores

Inspirado en TIDE (Bolya et al., 2020). Cada predicción se clasifica como:

| Tipo | Qué significa |
|---|---|
| Acierto | IoU ≥ 0,5 con un defecto real |
| Error de localización | Lo encontró, pero la caja quedó mal: IoU entre 0,1 y 0,5 |
| Duplicado | Otra caja sobre un defecto que ya se encontró |
| Alucinación | Marcó algo donde no hay ningún defecto (IoU < 0,1) |

Cada defecto real queda **detectado**, **encontrado sólo con caja mala** o **no detectado**.

Dos medidas complementan esa clasificación:
- el recall a IoU 0,1, 0,3 y 0,5 muestra cuánto se recupera si se perdona el encuadre;
- cuánto sube el mAP si se corrige cada tipo de error indica cuál le cuesta más.

En la A3 oficial (mAP@0.5 0,131): con el umbral de valid encuentra 16 de 71 agujeros (recall 0,225) y el 72 % de lo que marca no es un acierto (precisión 0,276). Ningún agujero de menos de 8 px fue encontrado. La división entre alucinaciones, errores de localización y no detectados sale de `scripts/analyze_detection_errors.py`.

## 6. Detección de anomalías: aprender sólo lo normal

**Idea:** en vez de enseñar qué es un defecto, se enseña cómo es la tela sana. Lo que no se parezca a nada conocido es anómalo. Sirve cuando hay muchas fotos sanas y pocas o ninguna anotación de defectos.

**B0, estadística local (sin aprendizaje profundo):** mide el brillo medio y la variación de textura en una ventana de 15×15 px (filtros de caja, como en la materia) y marca lo que se aparta de los valores de la tela sana. Es el piso de comparación.

**B1, PaDiM:**
1. Pasa la tela sana por una CNN preentrenada en ImageNet y toma sus mapas de características.
2. Para **cada posición** del mapa, modela con una gaussiana cómo son las características normales ahí.
3. Una zona es anómala si su característica está lejos de esa gaussiana (distancia de Mahalanobis, que tiene en cuenta cómo varían y se correlacionan las características).

Supone que una misma posición significa siempre lo mismo, como en piezas alineadas. Una tela no cumple eso del todo.

**B2, PatchCore:**
1. **Características locales:** toma capas intermedias de una CNN preentrenada (capas 2 y 3). Las capas intermedias describen texturas y patrones; las últimas son demasiado específicas de ImageNet. Cada vector se promedia con sus vecinos (3×3), así describe una zona chica de la tela.
2. **Banco de memoria:** guarda los vectores de todas las zonas de las imágenes sanas.
3. **Coreset:** el banco es enorme, así que se conserva un subconjunto (1 %) elegido para cubrir toda la variedad de la tela sana. Gasta menos memoria y busca más rápido.
4. **Puntaje:** para cada zona de una imagen nueva, se busca la zona sana más parecida del banco. La distancia a esa vecina es el puntaje de anomalía. Eso da un **mapa por píxel**, y el puntaje del parche es el máximo del mapa.

A diferencia de PaDiM, no depende de la posición, porque compara cada zona contra todas las zonas sanas. Además **no se entrena con backpropagation**: la red queda fija y sólo se guardan características. Por eso necesita pocas imágenes sanas. En valid, con 100 parches normales llega a un AUROC de 0,855, con 500 a 0,868 y con 2000 a 0,882.

**¿Clasifica o segmenta?** Las dos cosas: el mapa por píxel localiza la anomalía, y su máximo da la decisión por parche. No hace falta Grad-CAM para ver dónde mira; el mapa ya lo muestra. Si el modelo funciona, el mapa de una tela sana debería ser uniforme y el de una defectuosa debería tener un pico sobre el defecto.

## 7. Cómo se mide la detección de anomalías

- **AUROC por parche o por imagen:** la probabilidad de que un ejemplo defectuoso elegido al azar tenga un puntaje mayor que uno sano. 0,5 equivale a tirar una moneda; 1 es perfecto. No depende de un umbral.
- **AUROC por píxel:** lo mismo, píxel por píxel contra la máscara del defecto. Los defectos grandes pesan más porque tienen más píxeles.
- **AUPRO:** para cada defecto mide qué fracción quedó marcada y promedia entre defectos, así uno chico pesa igual que uno grande. Se calcula hasta un 30 % de falsos positivos por píxel.
- Resultados en el test de RAW-FABRID:

  | Método | AUROC por parche | AUROC por píxel | AUPRO |
  |---|---:|---:|---:|
  | B0, estadística local | 0,596 | 0,594 | 0,270 |
  | B1, PaDiM | 0,864 | 0,925 | 0,804 |
  | B2, PatchCore (ResNet18, parches reducidos a 128 px, la mejor variante en valid) | 0,902 | 0,912 | 0,745 |

## 8. Segmentación supervisada y la comparación entre paradigmas (FabriD)

- Un **segmentador** clasifica cada píxel (defecto o no, o qué defecto) y aprende de máscaras anotadas. Se mide con **IoU** y **Dice** entre la máscara predicha y la real.
- **Por qué la comparación directa es en FabriD:**
  - CISUTAC no tiene prendas sanas, así que PatchCore no se puede entrenar ahí;
  - RAW-FABRID tiene etiquetas binarias, sin tipo de defecto;
  - en FabriD los dos paradigmas se evalúan sobre **el mismo test** (85 sanas y 278 defectuosas), con todos los defectos juntos como una sola clase, porque PatchCore no distingue tipos.
- **Métricas comunes:** AUROC por imagen y por píxel, AUPRO, IoU y Dice binarios, tiempo por imagen y cantidad de imágenes de entrenamiento.

## 9. Transferencia entre dominios

**Cambio de dominio:** el modelo se entrena con un tipo de foto y se usa con otro. Puede fallar aunque el defecto tenga el mismo nombre, porque cambian la tela, la luz, el fondo y, sobre todo, la escala. FabriD son primeros planos, con agujeros de ~32 px en la imagen de entrenamiento, y CISUTAC son prendas enteras, con agujeros de 4 px a 640. Por eso, al transferir, el tamaño del tile se elige para que el defecto llegue al modelo con un tamaño parecido al que vio al entrenar.

## 10. Preguntas probables y respuestas cortas

| Pregunta | Respuesta corta |
|---|---|
| ¿Por qué no usar directamente una resolución más alta? | Lo probamos (A1, foto completa a 1280): sube a 0,105, pero los defectos siguen siendo chicos. Los tiles conservan la resolución original con el mismo tamaño de entrada del modelo. |
| ¿Por qué 640 y 20 %? | Es la entrada nativa de YOLO. Con 20 % de solapamiento, el 97 % de los defectos queda entero en algún tile. Tiles de 1024 rindieron peor (0,084). |
| ¿Cuándo conviene anomalías y cuándo supervisado? | Anomalías, cuando hay muchas fotos sanas y pocos defectos anotados, o defectos de tipos imprevistos. Supervisado, cuando hay anotaciones y se necesita saber *qué* defecto es. La respuesta cuantitativa sale de la comparación en FabriD. |
| ¿Por qué el test se usa una sola vez? | Si se eligen umbrales o variantes mirando el test, el resultado queda optimista. Todo se decide en valid. |
| ¿Cómo evitaron la fuga de información? | CISUTAC se partió por día de captura, RAW-FABRID por imagen madre y FabriD por captura de origen. Las copias reducidas de las mismas fotos se descartaron. |
| ¿Por qué PatchCore usa una red de ImageNet si nunca vio telas? | Porque sus capas intermedias describen texturas y bordes en general. PatchCore sólo compara esas descripciones; no necesita que la red sepa qué es una tela. |
