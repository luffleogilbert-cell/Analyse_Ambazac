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
    return np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))


def calc_aspect(mnt, res=25):
    dz_dy, dz_dx = np.gradient(mnt, res, res)
    return np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360


def run():
    # ── ENTRÉES (WIDGETS) ───────────────────────────────────────────────────

    # Correction : on stocke le résultat des inputs
    # Note : str(widget) permet de s'assurer qu'on récupère bien le chemin du fichier
    f_geo = onecode.file_input(
        'fichier_geochimie',
        'data/Points_geochimie_AMBAZAC.geojson',
        label="Points Geochimiques (GeoJSON)"
    )

    f_mnt = onecode.file_input(
        'fichier_mnt',
        'data/MNT_25M_AMBAZAC_IMAGE.tif',
        label="MNT (GeoTIFF)"
    )

    f_mad = onecode.slider('facteur_mad', 2.0, min=1.0, max=5.0, step=0.5, label="Sensibilite Anomalies (Facteur MAD)")

    p_au = onecode.slider('poids_au', 0.5, min=0.0, max=1.0, step=0.1, label="Poids Or (Au)")
    p_as = onecode.slider('poids_as', 0.2, min=0.0, max=1.0, step=0.1, label="Poids Arsenic (As)")
    p_w = onecode.slider('poids_w', 0.2, min=0.0, max=1.0, step=0.1, label="Poids Tungstene (W)")
    p_bi = onecode.slider('poids_bi', 0.1, min=0.0, max=1.0, step=0.1, label="Poids Bismuth (Bi)")

    quantile_top = onecode.slider('quantile_cible', 0.95, min=0.80, max=0.99, step=0.01, label="Top Quantile Cible")

    # ── 1. CHARGEMENT DES DONNÉES ─────────────────────────────────────────

    onecode.Logger.info("Chargement des donnees...")

    # CORRECTION CRITIQUE : Conversion en str() pour éviter l'erreur DataSourceError
    data = gpd.read_file(str(f_geo))

    with rasterio.open(str(f_mnt)) as src:
        mnt = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        res = src.res[0]

    # ... [Le reste de ton algorithme d'analyse reste identique] ...
    # Assure-toi de bien utiliser f_mad, p_au, etc. dans tes calculs suivants

    # Exemple pour la suite de ton code :
    # med_au, is_anom_au = anomalies(data['Au_ppb'], facteur=f_mad)

    onecode.Logger.info("Traitement termine avec succes.")

    # Sorties
    # plt.savefig(onecode.file_output('carte_finale', 'outputs/carte_potentiel.png', make_path=True))
