# Localización Óptima de Centros de Distribución en Bolivia 🇧🇴

Ejemplo didáctico de **Machine Learning aplicado a logística**: ubicación de centros de distribución (CD) en Bolivia usando **K-Means ponderado por demanda**.

## 📁 Contenido

| Archivo | Descripción |
|---|---|
| `Clusters_Bolivia_Ejemplo.ipynb` | Notebook principal (Google Colab / Jupyter) |
| `coordenadas_bolivia_1000_puntos.txt` | Dataset: 1,000 localidades con coordenadas y demanda anual en pallets |

## 🚀 Cómo usarlo en Google Colab

1. Abre el notebook directamente en Colab:

   [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santiagonajera/LOCALIZACION-BOLIVIA/blob/main/Clusters_Bolivia_Ejemplo.ipynb)

2. Ejecuta todas las celdas (**Entorno de ejecución → Ejecutar todas**).
3. El notebook descarga el dataset automáticamente desde este repositorio; no necesitas subir ningún archivo.

## 🧠 Metodología

1. **Carga de datos**: 1,000 localidades bolivianas con latitud, longitud y demanda anual (pallets).
2. **Método del codo**: estimación del número óptimo de clusters (K).
3. **K-Means ponderado**: la demanda entra como `sample_weight` — los puntos con más pallets "jalan" más el centro del cluster.
4. **Distancias Haversine**: distancias reales en km sobre la superficie terrestre.
5. **Mapa interactivo (Folium)**: clientes, clusters y centros de distribución.

## 📊 Resultado esperado

Con K = 4, el algoritmo ubica los centros de distribución en: **Trinidad, Santa Cruz de la Sierra, Oruro y Potosí**.

## ✏️ Ejercicios para estudiantes

La sección 7 del notebook propone 6 ejercicios: variar K manualmente, quitar la ponderación por demanda, análisis por región (Altiplano vs. Llanos), función objetivo de costo logístico `suma(demanda × distancia)` y el reto de reemplazar K-Means por K-Medoids.
