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

# Configuration visuelle
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 110

# ──────────────────────────────────────────────
# HELPERS (Fonctions de calcul)
# ──────────────────────────────────────────────

def anomalies(serie, facteur=2.0):
    med = serie.median()
    mad = (serie - med).abs().median()
    seuil = med + facteur * mad
    return seuil, (serie > seuil)

def normaliser(serie):
    s = np.log10(serie.replace(0, np.nan))
    return (s - s.min()) / (s.max() - s.min())

def calc_pente(mnt, res=25):
    dz_dy, dz_dx = np.gradient(mnt, res, res)
    return np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

def calc_hillshade(mnt, azimuth=315, altitude=45, res=25):
    az = np.radians(360 - azimuth + 90)
    alt = np.radians(altitude)
    dz_dy, dz_dx = np.gradient(mnt, res, res)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    asp = np.arctan2(dz_dy, -dz_dx)
    hs = np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - asp)
    return np.clip(hs, 0, 1)

# ──────────────────────────────────────────────
# FLOW PRINCIPAL (L'application OneCode)
# ──────────────────────────────────────────────

def run():
    # --- 1. ENTRÉES : DÉCLARATION DES WIDGETS ---
    
    fichier_geo = onecode.file_input(
        key='fichier_geochimie',
        label="1. Téléchargez vos points (GeoJSON)",
        types=[("GeoJSON", ".geojson .json")]
    )

    fichier_mnt = onecode.file_input(
        key='fichier_mnt',
        label="2. Téléchargez votre MNT (TIFF)",
        types=[("GeoTIFF", ".tif .tiff")]
    )

    facteur_mad = onecode.slider(
        key='facteur_mad', value=2.0, min=1.0, max=4.0, step=0.1,
        label="Sensibilité Anomalies (MAD)"
    )

    # Sliders de poids pour le score de potentiel
    p_au = onecode.slider('poids_au', 0.40, min=0.0, max=1.0, label="Poids Or (Au)")
    p_as = onecode.slider('poids_as', 0.20, min=0.0, max=1.0, label="Poids Arsenic (As)")
    p_w  = onecode.slider('poids_w',  0.20, min=0.0, max=1.0, label="Poids Tungstène (W)")
    p_bi = onecode.slider('poids_bi', 0.20, min=0.0, max=1.0, label="Poids Bismuth (Bi)")

    q_top = onecode.slider('quantile_top', 0.95, min=0.80, max=0.99, step=0.01, 
                           label="Seuil Zones Prioritaires (0.95 = top 5%)")

    # --- 2. VÉRIFICATION DES FICHIERS ---
    if fichier_geo is None or fichier_mnt is None:
        onecode.Logger.info("👋 Bienvenue ! Veuillez charger vos fichiers GeoJSON et MNT dans l'onglet Input pour démarrer l'analyse.")
        return

    # --- 3. CHARGEMENT ET CALCULS ---
    onecode.Logger.info("🚀 Lancement de l'analyse géochimique et topographique...")

    # Chargement
    data = gpd.read_file(str(fichier_geo))
    with rasterio.open(str(fichier_mnt)) as src:
        mnt = src.read(1).astype(float)
        mnt[mnt == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Dérivés MNT
    mnt_hs = calc_hillshade(mnt)
    
    # Calcul du Score de Potentiel
    total_poids = p_au + p_as + p_w + p_bi
    if total_poids == 0: total_poids = 1
    
    score = (
        normaliser(data['Au_ppb']) * (p_au/total_poids) +
        normaliser(data['As_ppm']) * (p_as/total_poids) +
        normaliser(data['W_ppm'])  * (p_w/total_poids)  +
        normaliser(data['Bi_ppm']) * (p_bi/total_poids)
    ).fillna(0)

    # --- 4. GÉNÉRATION DES SORTIES (OUTPUTS) ---

    # Carte 1 : Carte de Potentiel
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mnt_hs, cmap='gray', extent=extent, origin='upper')
    ax.imshow(mnt, cmap='terrain', extent=extent, origin='upper', alpha=0.3)
    
    sc = ax.scatter(data.geometry.x, data.geometry.y, c=score, cmap='hot_r', s=20, alpha=0.8)
    
    # Marquage des zones top prioritaires
    seuil_top = score.quantile(q_top)
    top_pts = data[score >= seuil_top]
    ax.scatter(top_pts.geometry.x, top_pts.geometry.y, c='cyan', marker='*', s=60, edgecolors='black', label=f'Cibles Top {int((1-q_top)*100)}%')
    
    plt.colorbar(sc, label='Score de Potentiel')
    ax.legend()
    ax.set_title("Carte Prédictive de Potentiel Minéral")
    
    plt.savefig(onecode.file_output('carte_potentiel', 'outputs/carte_potentiel.png', make_path=True))
    plt.close()

    # Carte 2 : Analyse des anomalies (MAD)
    # (On pourrait en ajouter d'autres ici selon le même modèle)

    onecode.Logger.info(f"✅ Analyse terminée. {len(top_pts)} cibles identifiées. Résultats disponibles dans l'onglet Output.")
