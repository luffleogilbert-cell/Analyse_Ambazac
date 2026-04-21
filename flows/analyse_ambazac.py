import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import warnings

# IMPORTATION DIRECTE (crucial pour l'analyseur PyCG selon la doc)
from onecode import Logger, file_input, slider, file_output

warnings.filterwarnings('ignore')

def normaliser(serie):
    s = np.log10(serie.replace(0, np.nan))
    return (s - s.min()) / (s.max() - s.min())

def run():
    # --- DÉCLARATION DES ÉLÉMENTS (Syntaxe exacte de la doc) ---
    
    # On définit les entrées
    # La doc précise : csv_reader('label', 'default.csv') 
    # ou file_input('key', 'default_path')
    
    f_geo = file_input('fichier_geochimie', 'data/Points_geochimie_AMBAZAC.geojson', label="Données Géo")
    f_mnt = file_input('fichier_mnt', 'data/MNT_25M_AMBAZAC_IMAGE.tif', label="MNT")

    # Sliders avec la syntaxe simplifiée de la doc
    f_mad = slider('facteur_mad', 2.0, min=1.0, max=5.0)
    p_au  = slider('poids_au', 0.5, min=0.0, max=1.0)
    p_as  = slider('poids_as', 0.2, min=0.0, max=1.0)
    p_w   = slider('poids_w', 0.2, min=0.0, max=1.0)
    p_bi  = slider('poids_bi', 0.1, min=0.0, max=1.0)

    # --- LOGIQUE DE CALCUL ---
    Logger.info(f"Analyse lancée avec MAD : {f_mad}")

    # Chargement
    data = gpd.read_file(str(f_geo))
    
    with rasterio.open(str(f_mnt)) as src:
        mnt = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Calcul du score
    score = (
        normaliser(data['Au_ppb']) * p_au +
        normaliser(data['As_ppm']) * p_as +
        normaliser(data['W_ppm']) * p_w +
        normaliser(data['Bi_ppm']) * p_bi
    ).fillna(0)

    # --- SORTIE (Syntaxe exacte file_output de la doc) ---
    fig, ax = plt.subplots()
    ax.imshow(mnt, extent=extent, cmap='terrain')
    ax.scatter(data.X, data.Y, c=score, cmap='hot_r', s=10)
    
    # La doc dit : plt.savefig(file_output('key', 'path.png'))
    plt.savefig(file_output('carte_finale', 'outputs/resultat.png'))
    plt.close()
