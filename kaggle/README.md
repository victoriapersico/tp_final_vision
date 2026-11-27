# Lanzadores de Kaggle

Cada lanzador clona este repo (rama indicada en `RAMA`), baja los datos de su fuente oficial y ejecuta un notebook de `notebooks/` con papermill. No tienen lógica propia: para cambiar un experimento se edita el notebook, no el lanzador.

## Corridas pendientes (rama `sofi`, YOLO26n)

Corren en este orden de prioridad. Kaggle permite **2 sesiones con GPU a la vez**, así que conviene lanzar 1 y 2 juntas y la 3 cuando termine alguna.

| # | Lanzador | Ejecuta | Qué produce | GPU estimada |
|---|---|---|---|---|
| 1 | `kaggle_semillas_holes.ipynb` | `notebooks/02c_semillas_cisutac.ipynb` con Holes | A0, A1 (1280) y A3 con semillas 0, 1 y 2, en valid y **test** (incluye el test de Holes que faltaba) | ~1–1,5 h |
| 2 | `kaggle_semillas_spots.ipynb` | lo mismo con Spots | ídem | ~1–1,5 h |
| 3 | `kaggle_mejora_a3_holes.ipynb` | `notebooks/02d_mejora_a3_cisutac.ipynb` con Holes | A3 con negativos sorteados vs. negativos difíciles (100 % y 50 %), sólo valid | < 1 h |

Los tres ya tienen `MODO_PRUEBA = False` y `RAMA = "sofi"`: no hay que tocar nada.

## Cómo lanzarlos

**Forma recomendada (la que usamos siempre):**

1. Descargar el `.ipynb` del lanzador desde GitHub (rama `sofi`, carpeta `kaggle/`): botón *Download raw file*.
2. En Kaggle: *Create → New Notebook → File → Import Notebook → File* y subir el archivo.
3. Panel derecho → *Session options*: **Accelerator = GPU T4 x2** e **Internet = On**.
4. *Save Version* → *Save & Run All (Commit)* → en *Advanced Settings* elegir *Run with GPU for this session* → *Save*. Sigue corriendo aunque se cierre la compu.
5. Al terminar, en la pestaña *Output*: descargar el ZIP (`results_semillas_<dataset>.zip` o `results_mejora_a3_holes.zip`) y el notebook ejecutado (`02c_..._ejecutado_<dataset>.ipynb` / `02d_...`).

**Alternativa sin descargar archivos:** crear un notebook nuevo (*New Notebook*), borrar la celda de ejemplo y pegar una sola línea con el `.py` equivalente, por ejemplo para Holes:

```python
import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/victoriapersico/tp_final_vision/sofi/kaggle/kaggle_semillas_holes.py').read().decode())
```

(cambiar `kaggle_semillas_holes.py` por `kaggle_semillas_spots.py` o `kaggle_mejora_a3_holes.py`). Después seguir desde el paso 3.

**Opcional, semilla 0 sin reentrenar:** si alguien tiene el Output de la corrida YOLO26 del notebook 02 (la de Pepo, `results_bloque_a_<dataset>.zip`), agregarlo con *Add Input* antes de lanzar 1 y 2. La semilla 0 reutiliza esos pesos; si no está, se reentrena con la misma semilla.

## Qué hacer con los resultados

- Descomprimir los ZIP en la raíz del repo (crean `results/…`, que no se versiona).
- Guardar los notebooks ejecutados en `notebooks/ejecutados/` y commitearlos.
- Las tablas resumen quedan en `results/cisutac/bloque_a/semillas/` (02c) y `results/cisutac/bloque_a/mejora_a3/holes/` (02d); conviene copiar los CSV chicos a `reports/cisutac/` para versionarlos.

## Problemas conocidos

- **Corrida en "Queued" durante horas.** Nos pasó al lanzar desde la cuenta de Sofía el 26–27/11: varias versiones quedaron en *Queued* y, al cancelarlas, en *Cancelling…* por horas, mientras una sesión interactiva con GPU arrancaba en segundos. No encontramos la causa (la cuenta tiene el teléfono verificado y cuota libre). Si pasa, probar desde otra cuenta.
- **Límite de 2 sesiones con GPU.** Si aparece *GPU Session cap reached*, esperar a que termine una corrida.
- **Si una corrida se corta** (límite de 12 h o error), agregar su Output con *Add Input* y volver a lanzar: los notebooks reutilizan lo que ya está en `results/`.

## Lanzadores anteriores

| Lanzador | Ejecuta |
|---|---|
| `kaggle_bloque_a.ipynb` | notebook 02 (CISUTAC), con `DATASET` y `MODO_PRUEBA` a elegir |
| `kaggle_bloque_a_spots.ipynb` | notebook 02 para Spots con evaluación en test |
| `kaggle_bloque_b.ipynb` | notebook 03 (RAW-FABRID, CPU) |
| `kaggle_semillas.ipynb` | notebook 02c con parámetros a elegir (por defecto `MODO_PRUEBA = True`) |
