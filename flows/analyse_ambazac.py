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
    # Déclaration au début pour assurer l'affichage dans l'UI

    input_geo = onecode.file_input(
        'fichier_geochimie',
        'data/Points_geochimie_AMBAZAC.geojson',
        label="Fichier geochimique (GeoJSON)",
        types=[("GeoJSON", ".geojson .json")]
    )

    input_mnt = onecode.file_input(
        'fichier_mnt',
        'data/MNT_25M_AMBAZAC_IMAGE.tif',
        label="Image MNT (GeoTIFF)",
        types=[("GeoTIFF", ".tif .tiff")]
    )

    f_mad = onecode.slider(
        'facteur_mad',
        2.0,
        label="Facteur MAD (seuil anomalie)",
        min=1.0,
        max=4.0,
        step=0.1
    )

    p_au = onecode.slider(
        'poids_au',
        0.40,
        label="Poids Au dans le score de potentiel",
        min=0.0,
        max=1.0,
        step=0.05
    )

    p_as = onecode.slider(
        'poids_as',
        0.20,
        label="Poids As dans le score de potentiel",
        min=0.0,
        max=1.0,
        step=0.05
    )

    p_w = onecode.slider(
        'poids_w',
        0.20,
        label="Poids W dans le score de potentiel",
        min=0.0,
        max=1.0,
        step=0.05
    )

    p_bi = onecode.slider(
        'poids_bi',
        0.20,
        label="Poids Bi dans le score de potentiel",
        min=0.0,
        max=1.0,
        step=0.05
    )

    q_top = onecode.slider(
        'quantile_top',
        0.95,
        label="Quantile seuil zones prioritaires (0.95 = top 5%)",
        min=0.80,
        max=0.99,
        step=0.01
    )

    # ── 1. Chargement des données ─────────────────────────────────────────

    onecode.Logger.info("Chargement des donnees geochimiques...")
    # Correction : str() force la récupération du chemin du fichier
    data = gpd.read_file(str(input_geo))

    elements = [
        'Au_ppb', 'Ag_ppm', 'Al_pct', 'As_ppm', 'Ba_ppm', 'Be_ppm',
        'Bi_ppm', 'Ca_pct', 'Cd_ppm', 'Ce_ppm', 'Co_ppm', 'Cr_ppm',
        'Cs_ppm', 'Cu_ppm', 'Fe_pct', 'Ga_ppm', 'Ge_ppm', 'Hf_ppm',
        'In_ppm', 'K_pct', 'La_ppm', 'Li_ppm', 'Mg_pct', 'Mn_ppm',
        'Mo_ppm', 'Na_pct', 'Nb_ppm', 'Ni_ppm', 'P_ppm', 'Pb_ppm',
        'Rb_ppm', 'Re_ppm', 'S_pct', 'Sb_ppm', 'Sc_ppm', 'Se_ppm',
        'Sn_ppm', 'Sr_ppm', 'Ta_ppm', 'Te_ppm', 'Th_ppm', 'Ti_pct',
        'Tl_ppm', 'U_ppm', 'V_ppm', 'W_ppm', 'Y_ppm', 'Zn_ppm', 'Zr_ppm'
    ]

    indicateurs = ['Au_ppb', 'As_ppm', 'W_ppm', 'Sn_ppm', 'Bi_ppm',
                   'Cu_ppm', 'Pb_ppm', 'Zn_ppm', 'Ag_ppm', 'Sb_ppm']

    onecode.Logger.info(f"{len(data)} points, {len(elements)} elements, CRS : {data.crs}")

    onecode.Logger.info("Chargement du MNT...")
    with rasterio.open(str(input_mnt)) as src:
        mnt = src.read(1).astype(float)
        mnt[mnt == src.nodata] = np.nan
        transform = src.transform
        bounds = src.bounds

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    onecode.Logger.info(f"MNT : {mnt.shape[1]}x{mnt.shape[0]} px")

    # ── 2. Statistiques descriptives ──────────────────────────────────────

    onecode.Logger.info("Calcul des statistiques descriptives...")
    resultats = []
    for elem in indicateurs:
        valeurs = data[elem]
        resultats.append({
            'element': elem,
            'min':     round(valeurs.min(), 2),
            'median':  round(valeurs.median(), 2),
            'mean':    round(valeurs.mean(), 2),
            'max':     round(valeurs.max(), 2),
            'CV%':     round(valeurs.std() / valeurs.mean() * 100, 1)
        })

    stats = pd.DataFrame(resultats).set_index('element')
    onecode.Logger.info(f"\n{stats.to_string()}")

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()
    for i, elem in enumerate(indicateurs):
        vals = np.log10(data[elem][data[elem] > 0])
        axes[i].hist(vals, bins=30, color='steelblue', edgecolor='white', linewidth=0.4)
        axes[i].axvline(vals.median(), color='red', linewidth=1.2, linestyle='--')
        axes[i].set_title(elem, fontsize=10)
        axes[i].set_xlabel('log10')
    fig.suptitle('Distributions des elements indicateurs (log10)', y=1.01)
    plt.tight_layout()
    plt.savefig(onecode.file_output('distributions', 'outputs/01_distributions.png', make_path=True),
                dpi=130, bbox_inches='tight')
    plt.close()

    # ── 3. Anomalies géochimiques ─────────────────────────────────────────

    onecode.Logger.info(f"Detection des anomalies (facteur MAD = {f_mad})...")
    for elem in indicateurs:
        seuil, masque = anomalies(data[elem], facteur=f_mad)
        onecode.Logger.info(f"  {elem}  seuil={seuil:.2f}  anomalies={masque.sum()} ({masque.mean()*100:.1f}%)")

    elements_carte = ['Au_ppb', 'As_ppm', 'W_ppm', 'Sn_ppm']
    cmaps = ['YlOrRd', 'PuRd', 'Blues', 'Greens']

    fig, axes = plt.subplots(2, 2, figsize=(13, 11
