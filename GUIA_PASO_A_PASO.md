# Guía Paso a Paso: Localización de Centros de Distribución en Bolivia mediante Clustering 🇧🇴

Esta guía explica, paso a paso, qué hace el ejemplo `Clusters_Bolivia_Ejemplo.ipynb` (o su versión script `codigo_bolivia_clustering.py`) y **por qué** se hace cada cosa. Está pensada para estudiantes que ven clustering por primera vez aplicado a logística.

---

## 🎯 El problema que resolvemos

Una empresa quiere atender la demanda de **todo Bolivia** (1,000 localidades, desde Cobija hasta Tarija). La pregunta es:

> **¿En qué ciudades conviene construir los Centros de Distribución (CD) y qué localidades atendería cada uno?**

Responder "a ojo" es mala idea: construir de más encarece la operación; construir de menos dispara los costos de transporte. El **clustering** nos da una respuesta con datos: agrupa las localidades cercanas entre sí y ponderadas por su demanda, y cada grupo (cluster) queda asignado a un CD.

---

## Paso 0 — Las herramientas (librerías)

```python
import pandas as pd          # manejo de tablas (DataFrames)
import numpy as np           # cálculo numérico
import matplotlib.pyplot as plt  # gráficos
from sklearn.cluster import KMeans       # el algoritmo de clustering
from sklearn.preprocessing import StandardScaler  # escalado de variables
import folium               # mapas interactivos
import requests             # descargar el dataset desde GitHub
```

**¿Por qué tantas librerías?** Cada una resuelve una parte: pandas organiza los datos, scikit-learn hace el clustering, y Folium dibuja el mapa. En la vida real nadie programa K-Means desde cero: se usan librerías probadas.

---

## Paso 1 — Cargar los datos

```python
GITHUB_URL = "https://raw.githubusercontent.com/santiagonajera/LOCALIZACION-BOLIVIA/main/coordenadas_bolivia_1000_puntos.txt"
df = pd.read_csv(StringIO(requests.get(GITHUB_URL).text), sep="\t")
```

El dataset es un archivo de texto separado por **tabulaciones** (`\t`). Cada fila es una localidad de Bolivia:

| Columna | Qué contiene | Ejemplo |
|---|---|---|
| `LOCALIDAD` | Nombre de la localidad | Zona Comercial 4012 |
| `CIUDAD_BASE` | Ciudad principal cercana | Trinidad |
| `LATITUD`, `LONGITUD` | Posición geográfica | -14.83, -64.90 |
| `DEMANDA_PALLETS_ANUAL` | Pallets que pide al año | 1,250 |
| `REGION` / `ZONA_GEOGRAFICA` | Clasificación geográfica | Norte / Llanos |

**Momento de exploración:** antes de cualquier modelo, mira los datos. El ejemplo imprime cuántas localidades hay por ciudad y cuánta demanda hay por región. Conclusión clave: **la demanda no está repartida parejo** — el Oriente (Santa Cruz) y el eje central concentran la mayoría.

---

## Paso 2 — Preparar los datos para el clustering

```python
X = df[["LATITUD", "LONGITUD"]].values
X_scaled = StandardScaler().fit_transform(X)
pesos = df["DEMANDA_PALLETS_ANUAL"].values
```

Dos decisiones importantes:

1. **Escalar (StandardScaler):** convierte latitud y longitud a una misma escala (media 0, desviación 1). K-Means trabaja con distancias; si una variable tuviera números mucho más grandes que la otra, dominaría el resultado.
2. **`pesos` (sample_weight):** aquí está el truco del ejemplo. La demanda entra como **peso** de cada punto. Una localidad que pide 8,000 pallets "jala" el centro del cluster 8 veces más que una que pide 1,000. Así, los CDs quedan cerca de donde está la *demanda*, no solo donde hay más puntos.

---

## Paso 3 — ¿Cuántos CDs? El método del codo

K-Means no decide solo cuántos clusters crear: hay que decírselo (el parámetro **K**). El **método del codo** ayuda a elegirlo:

```python
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled, sample_weight=pesos)
    inercias.append(kmeans.inertia_)
```

- La **inercia** es la suma de las distancias al cuadrado de cada punto a su centro. A menor inercia, mejor agrupados están los puntos.
- Con K = 1 la inercia es altísima; conforme K sube, la inercia siempre baja... pero **cada vez baja menos**.
- El "codo" es el punto donde agregar otro CD ya no mejora mucho. El código lo detecta con la **segunda derivada** de la curva (donde la pendiente cambia más bruscamente) y lo limita a un rango razonable de 4 a 8 CDs.

En Bolivia el codo sugiere **K = 4**.

> 💡 **Interpretación logística:** cada CD cuesta dinero (alquiler, personal, inventario). El codo es el equilibrio entre "pocos CDs baratos pero lejanos" y "muchos CDs caros pero cercanos".

---

## Paso 4 — K-Means final y elección de los CDs

```python
kmeans_final = KMeans(n_clusters=k_sugerido, random_state=42, n_init=20)
df["CLUSTER"] = kmeans_final.fit_predict(X_scaled, sample_weight=pesos)
```

K-Means agrupa las 1,000 localidades en K clusters. Luego, para cada cluster:

1. **Centro de gravedad ponderado:** se calcula el promedio de latitud/longitud **ponderado por la demanda**:
   ```
   lat_cg = Σ(latitudᵢ × demandaᵢ) / Σ(demandaᵢ)
   ```
   Es el punto teórico ideal para el almacén.
2. **Del punto teórico a la realidad:** ese centro puede caer en medio de la selva o de un lago. Por eso el código busca la **localidad real más cercana** al centro de gravedad, usando distancia Haversine (ver Paso 5). Esa localidad se convierte en el CD.

**Resultado para Bolivia (K = 4):**

| CD | Ciudad | Coordenadas |
|---|---|---|
| CD #1 | Trinidad | -14.71, -64.90 |
| CD #2 | Santa Cruz de la Sierra | -17.75, -63.19 |
| CD #3 | Oruro | -17.84, -67.12 |
| CD #4 | Potosí | -19.59, -65.63 |

Fíjate en el mapa de Bolivia: cubre el norte (Trinidad para el Beni y Pando), el Oriente (Santa Cruz), el centro/Altiplano (Oruro) y el sur (Potosí). El clustering descubrió solo la estructura geográfica y de demanda del país.

---

## Paso 5 — Distancias reales: la fórmula Haversine

Las coordenadas están en grados sobre una Tierra **esférica**: restar latitudes como si fuera geometría plana da errores de kilómetros. La **fórmula Haversine** calcula la distancia real sobre la superficie terrestre:

```
a = sen²(Δlat/2) + cos(lat₁) · cos(lat₂) · sen²(Δlong)
d = 2 · R · arcsen(√a)        con R = 6,371 km
```

El código la implementa vectorizada con NumPy (calcula miles de distancias de una sola vez) y la usa en todas partes: para elegir el CD, para medir qué tan lejos queda cada localidad de su CD y para el resumen final.

---

## Paso 6 — Evaluar la solución: ¿qué tan buena es?

Se construyen dos tablas:

**Tabla 1 — Resumen por CD (`df_resultados`):** cuántas localidades atiende cada CD, demanda total, y distancia promedio/máxima a sus clientes.

| CD | Localidades | Demanda total | Dist. promedio | Dist. máxima |
|---|---|---|---|---|
| Trinidad | 65 | 79,088 | 189.7 km | 586.4 km |
| Santa Cruz | 310 | 427,124 | 15.2 km | 55.4 km |
| Oruro | 450 | 624,090 | 133.6 km | 201.7 km |
| Potosí | 175 | 248,623 | 107.2 km | 249.6 km |

**Tabla 2 — Asignaciones (`df_asignaciones`):** cada localidad con su CD asignado y su distancia. Esta tabla responde "¿quién atiende a quién?" — es justo lo que necesitaría el jefe de logística.

> 💡 **Dato para discutir en clase:** la distancia promedio de Santa Cruz es solo 15 km porque la demanda está súper concentrada; en cambio Trinidad atiende todo el norte con distancias enormes (Cobija está a 586 km). ¿Conviene un quinto CD en el norte? Eso lo exploran los ejercicios.

---

## Paso 7 — Visualizar: el mapa interactivo

```python
mapa = folium.Map(location=[lat_centro, lon_centro], zoom_start=6, tiles="OpenStreetMap")
```

- **Marcadores con casita 🏠** = los 4 CDs (con su nombre y coordenadas).
- **Círculos de color** = las 1,000 localidades. El **color** dice qué CD la atiende y el **tamaño** dice cuánta demanda tiene.
- Clic en cualquier punto → ventana con localidad, ciudad base, demanda, CD asignado y distancia.

El mapa se guarda como `mapa_bolivia_cds.html`: un archivo que puedes abrir en cualquier navegador o compartir por correo.

**Por qué importa visualizar:** un número en una tabla no convence a nadie. Un mapa muestra de un vistazo si la solución tiene sentido geográfico — y es lo primero que preguntaría tu jefe.

---

## Paso 8 — A practicar (ejercicios)

La sección 7 del notebook propone 6 ejercicios para profundizar:

1. **K = 5 manual:** ¿qué nueva ciudad aparece? ¿mejora la distancia promedio?
2. **K = 6:** ¿qué pasa con Trinidad y Cobija?
3. **Sin ponderación** (`sample_weight=None`): ¿cómo cambian los CDs si la demanda no importa?
4. **Análisis por región:** Altiplano vs. Llanos con `REGION` y `ZONA_GEOGRAFICA`.
5. **Función objetivo:** costo logístico = Σ(demanda × distancia). ¿Qué K lo minimiza?
6. **Reto:** cambiar K-Means por K-Medoids (`scikit-learn-extra`). Un medoide siempre es un punto real — ¿por qué eso puede convenir para ubicar un CD?

---

## 🧠 Resumen de la lógica completa

```
Datos de Bolivia (localidad, coords, demanda)
        │
        ▼
Escalar coords + demanda como peso
        │
        ▼
Método del codo ──► ¿cuántos CDs?  (K = 4)
        │
        ▼
K-Means ponderado ──► cada localidad tiene su cluster
        │
        ▼
Centro de gravedad de cada cluster ──► localidad real más cercana = CD
        │
        ▼
Distancias Haversine + tablas + mapa interactivo
        │
        ▼
Decisión logística: 4 CDs (Trinidad, Santa Cruz, Oruro, Potosí)
```
