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

# Configuration de base
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 110


# ──────────────────────────────────────────────
# Fonctions Helpers
# ──────────────────────────────────────────────

def anomalies(serie, facteur=2.0):
    """Calcule le seuil d'anomalie basé sur la médiane et le MAD."""
    med = serie.median()
    mad = (serie - med).abs().median()
    seuil = med + facteur * mad
    return seuil, serie > seuil


def normaliser(serie):
    """Normalise une série en log10 entre 0 et 1."""
    s = np.log10(serie.replace(0, np.nan))
    return (s - s.min()) / (s.max() - s.min())


def calc_hillshade(mnt, azimuth=315, altitude=45, res=25):
    """Calcule l'ombrage pour le rendu visuel du MNT."""
    az = np.radians(360 - azimuth + 90)
    alt = np.radians(altitude)
    dz_dy, dz_dx = np.gradient(mnt, res, res)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    asp = np.arctan2(dz_dy, -dz_dx)
    hs = np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - asp)
    return np.clip(hs, 0, 1)


# ──────────────────────────────────────────────
# Fonction principale (Exécutée par OneCode)
# ──────────────────────────────────────────────

def run():
    # ── 1. DÉCLARATION DES WIDGETS (S'affichent dans l'onglet Input) ──
    # Important : Ces fonctions doivent être dans run() pour être détectées
    
    # Entrées de fichiers
    input_geo = onecode.file_input(
        key="fichier_geochimie",
        value="data/Points_geochimie_AMBAZAC.geojson",
        label="Points Geochimiques (GeoJSON)"
    )

    input_mnt = onecode.file_input(
        key="fichier_mnt",
        value="data/MNT_25M_AMBAZAC_IMAGE.tif",
        label="MNT (GeoTIFF)"
    )

    # Paramètres d'analyse via Sliders
    f_mad = onecode.slider(
        key='facteur_mad', 
        value=2.0, 
        min=1.0, 
        max=5.0, 
        step=0.1, 
        label="Sensibilité Anomalies (Facteur MAD)"
    )

    p_au = onecode.slider('poids_au', 0.5, min=0.0, max=1.0, step=0.1, label="Poids Or (Au)")
    p_as = onecode.slider('poids_as', 0.2, min=0.0, max=1.0, step=0.1, label="Poids Arsenic (As)")
    p_w  = onecode.slider('poids_w',  0.2, min=0.0, max=1.0, step=0.1, label="Poids Tungstène (W)")
    p_bi = onecode.slider('poids_bi', 0.1, min=0.0, max=1.0, step=0.1, label="Poids Bismuth (Bi)")

    quantile_top = onecode.slider(
        key='quantile_cible', 
        value=0.95, 
        min=0.80, 
        max=0.99, 
        step=0.01, 
        label="Top Quantile Cible (Priorité)"
    )

    # ── 2. CHARGEMENT ET TRAITEMENT ─────────────────────────────────────
    
    onecode.Logger.info("Chargement des fichiers...")
    
    # Correction : str(widget) extrait le chemin du fichier sélectionné
    data = gpd.read_file(str(input_geo))
    
    with rasterio.open(str(input_mnt)) as src:
        mnt = src.read(1).astype(float)
        mnt[mnt == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Calcul du score de potentiel basé sur les sliders
    onecode.Logger.info("Calcul du score de potentiel...")
    score = (
        normaliser(data['Au_ppb']) * p_au +
        normaliser(data['As_ppm']) * p_as +
        normaliser(data['W_ppm'])  * p_w  +
        normaliser(data['Bi_ppm']) * p_bi
    ).fillna(0)

    # Identification des meilleures cibles
    seuil_score = score.quantile(quantile_top)
    cibles = data[score >= seuil_score]

    # ── 3. GÉNÉRATION DES SORTIES (Outputs) ──────────────────────────────
    
    onecode.Logger.info("Génération de la carte...")
    mnt_hs = calc_hillshade(mnt)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mnt_hs, cmap='gray', extent=extent, origin='upper')
    ax.imshow(mnt, cmap='terrain', extent=extent, origin='upper', alpha=0.3)
    
    # Affichage des points avec couleur selon le score
    sc = ax.scatter(data.X, data.Y, c=score, cmap='hot_r', s=15, alpha=0.7)
    # Mise en évidence du top quantile
    ax.scatter(cibles.X, cibles.Y, c='cyan', marker='*', s=40, edgecolors='black', label='Cibles prioritaires')
    
    plt.colorbar(sc, label='Score de Potentiel Minéral')
    ax.set_title(f"Analyse Ambazac - MAD: {f_mad}")
    ax.legend()

    # Sauvegarde dans le dossier output
    output_path = onecode.file_output('carte_potentiel', 'outputs/01_carte_potentiel.png', make_path=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    onecode.Logger.info(f"Traitement terminé. {len(cibles)} points prioritaires identifiés.")
