# This file is your entry point:
# - add your Python files and folder inside this 'flows' folder
# - add your imports
# - just don't change the name of the function 'run()' nor this filename ('analyse_ambazac.py')

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

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 110


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def anomalies(serie, facteur=2.0):
    med = serie.median()
    mad = (serie - med).abs().median()
    seuil = med + facteur * mad
    return seuil, serie > seuil


def normaliser(serie):
    s = np.log10(serie.replace(0, np.nan))
    return (s - s.min()) / (s.max() - s.min())


def calc_pente(mnt, res=25):
    dz_dy, dz_dx = np.gradient(mnt, res, res)
    return np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))


def calc_aspect(mnt, res=25):
    dz_dy, dz_dx = np.gradient(mnt, res, res)
    return np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360


def calc_hillshade(mnt, azimuth=315, altitude=45, res=25):
    az = np.radians(360 - azimuth + 90)
    alt = np.radians(altitude)
    dz_dy, dz_dx = np.gradient(mnt, res, res)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    asp = np.arctan2(dz_dy, -dz_dx)
    hs = np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - asp)
    return np.clip(hs, 0, 1)


# ──────────────────────────────────────────────
# Flow principal
# ──────────────────────────────────────────────

def run():
    # ── ENTRÉES (WIDGETS) ───────────────────────────────────────────────────
    # Déclaration en début de fonction pour l'affichage UI
    
    f_geo_input = onecode.file_input(
        key='fichier_geochimie',
        value='data/Points_geochimie_AMBAZAC.geojson',
        label="Points Geochimiques (GeoJSON)"
    )

    f_mnt_input = onecode.file_input(
        key='fichier_mnt',
        value='data/MNT_25M_AMBAZAC_IMAGE.tif',
        label="MNT (GeoTIFF)"
    )

    f_mad = onecode.slider(
        key='facteur_mad', 
        value=2.0, 
        min=1.0, 
        max=5.0, 
        step=0.1, 
        label="Sensibilite Anomalies (Facteur MAD)"
    )

    p_au = onecode.slider('poids_au', 0.5, min=0.0, max=1.0, step=0.1, label="Poids Or (Au)")
    p_as = onecode.slider('poids_as', 0.2, min=0.0, max=1.0, step=0.1, label="Poids Arsenic (As)")
    p_w  = onecode.slider('poids_w',  0.2, min=0.0, max=1.0, step=0.1, label="Poids Tungstene (W)")
    p_bi = onecode.slider('poids_bi', 0.1, min=0.0, max=1.0, step=0.1, label="Poids Bismuth (Bi)")

    q_top = onecode.slider('quantile_cible', 0.95, min=0.80, max=0.99, step=0.01, label="Top Quantile Cible")

    # ── 1. CHARGEMENT DES DONNÉES ─────────────────────────────────────────
    
    onecode.Logger.info("Chargement des donnees...")
    
    # Correction : str() pour assurer la lecture du chemin par geopandas
    data = gpd.read_file(str(f_geo_input))
    
    with rasterio.open(str(f_mnt_input)) as src:
        mnt = src.read(1).astype(float)
        mnt[mnt == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        res = src.res[0]

    indicateurs = ['Au_ppb', 'As_ppm', 'W_ppm', 'Sn_ppm', 'Bi_ppm']

    # ── 2. CALCULS ET ANALYSES ────────────────────────────────────────────
    
    onecode.Logger.info(f"Traitement en cours (MAD factor: {f_mad})...")

    # Génération des graphiques de distribution
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, elem in enumerate(indicateurs):
        vals = np.log10(data[elem][data[elem] > 0])
        axes[i].hist(vals, bins=25, color='steelblue', edgecolor='white')
        axes[i].set_title(elem)
    
    plt.tight_layout()
    plt.savefig(onecode.file_output('distributions', 'outputs/01_distributions.png', make_path=True))
    plt.close()

    # ── 3. CARTE DE POTENTIEL ─────────────────────────────────────────────
    
    # Calcul du score combiné basé sur les poids des widgets
    total_poids = p_au + p_as + p_w + p_bi
    if total_poids > 0:
        score = (
            normaliser(data['Au_ppb']) * (p_au/total_poids) +
            normaliser(data['As_ppm']) * (p_as/total_poids) +
            normaliser(data['W_ppm'])  * (p_w/total_poids) +
            normaliser(data['Bi_ppm']) * (p_bi/total_poids)
        ).fillna(0)
    else:
        score = normaliser(data['Au_ppb'])

    # Identification des zones cibles
    seuil_score = score.quantile(q_top)
    cibles = data[score >= seuil_score]

    # Visualisation finale
    mnt_hs = calc_hillshade(mnt, res=res)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mnt_hs, cmap='gray', extent=extent, origin='upper')
    
    sc = ax.scatter(data.X, data.Y, c=score, cmap='YlOrRd', s=20, alpha=0.6)
    ax.scatter(cibles.X, cibles.Y, c='cyan', marker='*', s=50, label=f'Top {int((1-q_top)*100)}%')
    
    plt.colorbar(sc, label='Score de Potentiel')
    ax.set_title(f"Carte de Potentiel Mineral - Ambazac (MAD: {f_mad})")
    ax.legend()
    
    plt.savefig(onecode.file_output('carte_potentiel', 'outputs/02_carte_potentiel.png', make_path=True))
    plt.close()

    onecode.Logger.info(f"Traitement termine. {len(cibles)} points identifies dans le top quantile.")
