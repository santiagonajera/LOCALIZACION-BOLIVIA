# =====================================================================================
# LOCALIZACIÓN ÓPTIMA DE CENTROS DE DISTRIBUCIÓN EN BOLIVIA MEDIANTE CLUSTERING
# -------------------------------------------------------------------------------------
# Ejemplo didáctico: K-Means ponderado por demanda + distancias Haversine + mapa Folium
#
# Uso:
#   pip install -r requirements.txt
#   python codigo_bolivia_clustering.py
#
# Salidas:
#   - Consola: K sugerido, CDs seleccionados y resumen por cluster
#   - codo_bolivia.png          (gráfico del método del codo)
#   - mapa_bolivia_cds.html     (mapa interactivo, abrir en el navegador)
#   - asignaciones_bolivia.csv  (quién atiende a quién)
# =====================================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from io import StringIO

import matplotlib
matplotlib.use("Agg")  # guardar el gráfico sin abrir ventana
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import folium


# =====================================================================================
# FUNCIÓN HAVERSINE (DISTANCIA EN KM)
# =====================================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calcula distancia Haversine en km entre:
    - un punto (lat1, lon1) y
    - uno o muchos puntos (lat2, lon2)
    """
    R = 6371.0  # radio promedio de la Tierra en km

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # km


# =====================================================================================
# 1. CARGA DE DATOS DESDE GITHUB
# =====================================================================================
GITHUB_URL = "https://raw.githubusercontent.com/santiagonajera/LOCALIZACION-BOLIVIA/main/coordenadas_bolivia_1000_puntos.txt"

resp = requests.get(GITHUB_URL)
resp.raise_for_status()
df = pd.read_csv(StringIO(resp.text), sep="\t")

DEM_COL = "DEMANDA_PALLETS_ANUAL"

print(f"Total puntos de demanda: {len(df):,}")
print(f"Demanda total: {df[DEM_COL].sum():,} pallets/año")

print("\nPuntos por ciudad base:")
print(df["CIUDAD_BASE"].value_counts())

print("\nDemanda por región:")
print(df.groupby("REGION")[DEM_COL].sum().sort_values(ascending=False))


# =====================================================================================
# 2. PREPARACIÓN DE DATOS Y CÁLCULO DE K ÓPTIMO (MÉTODO DEL CODO)
# =====================================================================================
# Variables para clustering: latitud y longitud (escaladas)
X = df[["LATITUD", "LONGITUD"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# La demanda entra como PESO: los puntos con más pallets "jalan" más el centro
pesos = df[DEM_COL].values

inercias = []
k_range = range(2, 21)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled, sample_weight=pesos)
    inercias.append(kmeans.inertia_)

# Segunda derivada de la inercia para encontrar el "codo"
d1 = np.diff(inercias)
d2 = np.diff(d1)
k_optimo = int(np.argmax(d2) + 3)      # offset por las dos derivadas
k_sugerido = min(max(k_optimo, 4), 8)  # limitar entre 4 y 8 CDs

print(f"\nK óptimo matemático: {k_optimo}")
print(f"K sugerido (ajustado): {k_sugerido}")

# Gráfico del codo
plt.figure(figsize=(6, 4))
plt.plot(list(k_range), inercias, "o-")
plt.axvline(k_sugerido, color="r", linestyle="--", label=f"K sugerido = {k_sugerido}")
plt.xlabel("K")
plt.ylabel("Inercia ponderada")
plt.title("Método del codo - Demanda en Bolivia")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("codo_bolivia.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico guardado: codo_bolivia.png")


# =====================================================================================
# 3. K-MEANS FINAL Y CENTROS DE DISTRIBUCIÓN
# =====================================================================================
kmeans_final = KMeans(n_clusters=k_sugerido, random_state=42, n_init=20)
df["CLUSTER"] = kmeans_final.fit_predict(X_scaled, sample_weight=pesos)

nombres_centros = []
coordenadas_centros = []

for i in range(k_sugerido):
    cluster_data = df[df["CLUSTER"] == i]

    # Centro de gravedad ponderado por demanda del cluster
    lat_cg = np.average(cluster_data["LATITUD"], weights=cluster_data[DEM_COL])
    lon_cg = np.average(cluster_data["LONGITUD"], weights=cluster_data[DEM_COL])

    # CD = localidad real más cercana al centro de gravedad (distancia Haversine)
    dist = haversine_km(
        lat_cg, lon_cg,
        cluster_data["LATITUD"].values,
        cluster_data["LONGITUD"].values
    )
    loc_centro = cluster_data.iloc[dist.argmin()]

    nombres_centros.append(loc_centro["CIUDAD_BASE"])
    coordenadas_centros.append([loc_centro["LATITUD"], loc_centro["LONGITUD"]])

print("\nCentros de Distribución seleccionados:")
for i in range(k_sugerido):
    print(f"  CD #{i+1}: {nombres_centros[i]}  ({coordenadas_centros[i][0]:.4f}, {coordenadas_centros[i][1]:.4f})")


# =====================================================================================
# 4. RESUMEN POR CD (df_resultados) - DISTANCIAS HAVERSINE
# =====================================================================================
resultados_clusters = []
for i in range(k_sugerido):
    cluster_data = df[df["CLUSTER"] == i]
    demanda_total = cluster_data[DEM_COL].sum()
    num_localidades = len(cluster_data)

    distancias = haversine_km(
        coordenadas_centros[i][0],
        coordenadas_centros[i][1],
        cluster_data["LATITUD"].values,
        cluster_data["LONGITUD"].values
    )

    resultados_clusters.append({
        "Cluster": i + 1,
        "Centro": nombres_centros[i],
        "Localidades": num_localidades,
        "Demanda_Total": demanda_total,
        "Demanda_Promedio": demanda_total / num_localidades,
        "Dist_Promedio_km": distancias.mean(),
        "Dist_Maxima_km": distancias.max(),
        "Latitud": coordenadas_centros[i][0],
        "Longitud": coordenadas_centros[i][1]
    })

df_resultados = pd.DataFrame(resultados_clusters)
print("\nResumen por Centro de Distribución:")
print(df_resultados.round(2).to_string(index=False))


# =====================================================================================
# 5. TABLA "QUIÉN ATIENDE A QUIÉN" (df_asignaciones)
# =====================================================================================
df_asignaciones = df[[
    "LOCALIDAD", "CIUDAD_BASE", "LATITUD", "LONGITUD",
    DEM_COL, "TIPO_LOCALIDAD", "REGION",
    "ZONA_GEOGRAFICA", "CLUSTER"
]].copy()

df_asignaciones.rename(columns={DEM_COL: "DEMANDA"}, inplace=True)

df_asignaciones["CD_ASIGNADO"] = df_asignaciones["CLUSTER"].apply(lambda x: f"CD #{x+1}")
df_asignaciones["CENTRO_DISTRIBUCION"] = df_asignaciones["CLUSTER"].apply(lambda x: nombres_centros[x])

# Distancia Haversine de cada localidad a su CD
df_asignaciones["DISTANCIA_CD_KM"] = [
    haversine_km(row["LATITUD"], row["LONGITUD"],
                 coordenadas_centros[row["CLUSTER"]][0],
                 coordenadas_centros[row["CLUSTER"]][1])
    for _, row in df_asignaciones.iterrows()
]

df_asignaciones = df_asignaciones.sort_values(["CLUSTER", "DEMANDA"], ascending=[True, False])
df_asignaciones.to_csv("asignaciones_bolivia.csv", index=False, encoding="utf-8-sig")
print("\nAsignaciones guardadas: asignaciones_bolivia.csv")


# =====================================================================================
# 6. MAPA FOLIUM: CLIENTES + CENTROS + ASIGNACIÓN
# =====================================================================================
lat_centro = df["LATITUD"].mean()
lon_centro = df["LONGITUD"].mean()

mapa = folium.Map(location=[lat_centro, lon_centro], zoom_start=6, tiles="OpenStreetMap")

colores = ["red", "blue", "green", "purple", "orange", "darkred", "lightred", "beige"]

# Centros de distribución (marcadores con ícono de casa)
for i in range(k_sugerido):
    lat_cd, lon_cd = coordenadas_centros[i]
    nombre_cd = nombres_centros[i]

    folium.Marker(
        location=[lat_cd, lon_cd],
        popup=folium.Popup(f"CD #{i+1}<br>{nombre_cd}<br>({lat_cd:.4f}, {lon_cd:.4f})", max_width=250),
        icon=folium.Icon(color=colores[i], icon="home", prefix="fa"),
        tooltip=f"CD #{i+1} - {nombre_cd}"
    ).add_to(mapa)

# Clientes (todas las localidades)
for _, row in df_asignaciones.iterrows():
    cl = row["CLUSTER"]
    demanda = row["DEMANDA"]

    # tamaño del punto según demanda
    if demanda >= 5000:
        r = 7
    elif demanda >= 1000:
        r = 5
    else:
        r = 3

    folium.CircleMarker(
        location=[row["LATITUD"], row["LONGITUD"]],
        radius=r,
        color=colores[cl],
        fill=True,
        fill_color=colores[cl],
        fill_opacity=0.6,
        weight=1,
        popup=folium.Popup(
            f"<b>{row['LOCALIDAD']}</b><br>"
            f"Ciudad base: {row['CIUDAD_BASE']}<br>"
            f"Demanda: {demanda:,.0f} pallets/año<br>"
            f"Atendido por: {row['CD_ASIGNADO']} ({row['CENTRO_DISTRIBUCION']})<br>"
            f"Distancia aprox: {row['DISTANCIA_CD_KM']:.1f} km",
            max_width=250
        )
    ).add_to(mapa)

# Leyenda
leyenda_html = """
<div style="position: fixed;
     bottom: 10px; left: 10px; width: 260px; height: auto;
     background-color: white; z-index:9999; font-size:13px;
     border:2px solid grey; border-radius: 5px; padding: 10px">
     <b>Centros de Distribución (CD) - Bolivia</b><br>
     <i>Color = cluster / CD</i><br><br>
"""
for i in range(k_sugerido):
    leyenda_html += f"CD #{i+1} - {nombres_centros[i]}<br>"
leyenda_html += "</div>"

mapa.get_root().html.add_child(folium.Element(leyenda_html))

# Guardar el mapa como HTML
mapa.save("mapa_bolivia_cds.html")
print("Mapa guardado: mapa_bolivia_cds.html  (ábrelo en tu navegador)")
print("\n¡Listo! Clustering de Bolivia completado.")
