import onecode
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 110

# ──────────────────────────────────────────────
# Helpers (doivent être en dehors de run)
# ──────────────────────────────────────────────

def anomalies(serie, facteur=2.0):
    med = serie.median()
    mad = (serie - med).abs().median()
    seuil = med + facteur * mad
    return seuil, serie > seuil

def normaliser(serie):
    s = np.log10(serie.replace(0, np.nan))
    return (s - s.min()) / (s.max() - s.min())

def calc_hillshade(mnt, azimuth=315, altitude=45, res=25):
    az = np.radians(360 - azimuth + 90)
    alt = np.radians(altitude)
    dz_dy, dz_dx = np.gradient(mnt, res, res)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    asp = np.arctan2(dz_dy, -dz_dx)
    hs = np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - asp)
    return np.clip(hs, 0, 1)

# ──────────────────────────────────────────────
# La fonction run() : C'est ICI que OneCode cherche les widgets
# ──────────────────────────────────────────────

def run():
    # 1. DÉCLARATION DES WIDGETS (Doivent être AU DÉBUT de run())
    # On utilise les types et formats de la documentation
    
    f_geo = onecode.file_input(
        key="fichier_geochimie",
        value="data/Points_geochimie_AMBAZAC.geojson",
        label="Points Géochimiques (GeoJSON)"
    )

    f_mnt = onecode.file_input(
        key="fichier_mnt",
        value="data/MNT_25M_AMBAZAC_IMAGE.tif",
        label="MNT (GeoTIFF)"
    )

    f_mad = onecode.slider(
        key='facteur_mad', 
        value=2.0, 
        min=1.0, 
        max=5.0, 
        step=0.1, 
        label="Facteur MAD"
    )

    p_au = onecode.slider('poids_au', 0.5, min=0.0, max=1.0, step=0.1, label="Poids Or (Au)")
    p_as = onecode.slider('poids_as', 0.2, min=0.0, max=1.0, step=0.1, label="Poids Arsenic (As)")
    p_w  = onecode.slider('poids_w',  0.2, min=0.0, max=1.0, step=0.1, label="Poids Tungstène (W)")
    p_bi = onecode.slider('poids_bi', 0.1, min=0.0, max=1.0, step=0.1, label="Poids Bismuth (Bi)")

    q_top = onecode.slider('quantile_cible', 0.95, min=0.80, max=0.99, step=0.01, label="Top Quantile")

    # 2. LOGIQUE D'EXÉCUTION
    onecode.Logger.info("Chargement des données...")

    # Utilisation de str() car file_input renvoie un objet en mode UI
    data = gpd.read_file(str(f_geo))
    
    with rasterio.open(str(f_mnt)) as src:
        mnt = src.read(1).astype(float)
        mnt[mnt == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Calcul simplifié du score
    score = (
        normaliser(data['Au_ppb']) * p_au +
        normaliser(data['As_ppm']) * p_as +
        normaliser(data['W_ppm'])  * p_w  +
        normaliser(data['Bi_ppm']) * p_bi
    ).fillna(0)

    # Sortie visuelle
    fig, ax = plt.subplots(figsize=(10, 8))
    mnt_hs = calc_hillshade(mnt)
    ax.imshow(mnt_hs, cmap='gray', extent=extent, origin='upper')
    sc = ax.scatter(data.X, data.Y, c=score, cmap='hot_r', s=20)
    plt.colorbar(sc, label='Score de Potentiel')
    
    # Sauvegarde obligatoire pour l'affichage Output
    plt.savefig(onecode.file_output('carte', 'outputs/carte_potentiel.png', make_path=True))
    plt.close()

    onecode.Logger.info("Traitement terminé.")
