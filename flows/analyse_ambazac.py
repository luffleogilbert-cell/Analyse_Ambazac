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

from onecode import (
    Logger,
    file_input,
    file_output,
    slider,
)

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 110


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def anomalies(serie, facteur=2.0):
    """Détection des anomalies par Médiane + facteur × MAD."""
    med = serie.median()
    mad = (serie - med).abs().median()
    seuil = med + facteur * mad
    return seuil, serie > seuil


def normaliser(serie):
    """Normalisation log entre 0 et 1."""
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

    # ── Paramètres d'entrée ──────────────────────────────────────────────

    fichier_geo = file_input(
        'fichier_geochimie',
        'data/Points_geochimie_AMBAZAC.geojson',
        label="Fichier géochimique (GeoJSON)",
        types=[("GeoJSON", ".geojson .json")]
    )

    fichier_mnt = file_input(
        'fichier_mnt',
        'data/MNT_25M_AMBAZAC_IMAGE.tif',
        label="Image MNT (GeoTIFF)",
        types=[("GeoTIFF", ".tif .tiff")]
    )

    facteur_mad = slider(
        'facteur_mad',
        2.0,
        label="Facteur MAD (seuil anomalie)",
        min=1.0,
        max=4.0,
        step=0.1
    )

    poids_au = slider(
        'poids_au',
        0.40,
        label="Poids Au dans le score de potentiel",
        min=0.0,
        max=1.0,
        step=0.05
    )

    poids_as = slider(
        'poids_as',
        0.20,
        label="Poids As dans le score de potentiel",
        min=0.0,
        max=1.0,
        step=0.05
    )

    poids_w = slider(
        'poids_w',
        0.20,
        label="Poids W dans le score de potentiel",
        min=0.0,
        max=1.0,
        step=0.05
    )

    poids_bi = slider(
        'poids_bi',
        0.20,
        label="Poids Bi dans le score de potentiel",
        min=0.0,
        max=1.0,
        step=0.05
    )

    quantile_top = slider(
        'quantile_top',
        0.95,
        label="Quantile — seuil zones prioritaires (ex: 0.95 = top 5%)",
        min=0.80,
        max=0.99,
        step=0.01
    )

    # ── 1. Chargement des données ─────────────────────────────────────────

    Logger.info("Chargement des données géochimiques...")
    data = gpd.read_file(fichier_geo)

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

    Logger.info(f"{len(data)} points, {len(elements)} éléments, CRS : {data.crs}")

    Logger.info("Chargement du MNT...")
    with rasterio.open(fichier_mnt) as src:
        mnt = src.read(1).astype(float)
        mnt[mnt == src.nodata] = np.nan
        transform = src.transform
        bounds = src.bounds

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    Logger.info(f"MNT : {mnt.shape[1]}x{mnt.shape[0]} px | valeurs {np.nanmin(mnt):.0f}–{np.nanmax(mnt):.0f}")

    # ── 2. Statistiques descriptives ──────────────────────────────────────

    Logger.info("Calcul des statistiques descriptives...")
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
    Logger.info(f"\n{stats.to_string()}")

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()
    for i, elem in enumerate(indicateurs):
        vals = np.log10(data[elem][data[elem] > 0])
        axes[i].hist(vals, bins=30, color='steelblue', edgecolor='white', linewidth=0.4)
        axes[i].axvline(vals.median(), color='red', linewidth=1.2, linestyle='--')
        axes[i].set_title(elem, fontsize=10)
        axes[i].set_xlabel('log₁₀')
    fig.suptitle('Distributions des éléments indicateurs (log₁₀) — trait rouge = médiane', y=1.01)
    plt.tight_layout()
    plt.savefig(file_output(key="distributions", value="outputs/01_distributions.png", make_path=True),
                dpi=130, bbox_inches='tight')
    plt.close()

    # ── 3. Anomalies géochimiques ─────────────────────────────────────────

    Logger.info(f"Détection des anomalies (facteur MAD = {facteur_mad})...")
    for elem in indicateurs:
        seuil, masque = anomalies(data[elem], facteur=facteur_mad)
        Logger.info(f"  {elem:10s}  seuil={seuil:.2f}  anomalies={masque.sum()} ({masque.mean()*100:.1f}%)")

    elements_carte = ['Au_ppb', 'As_ppm', 'W_ppm', 'Sn_ppm']
    cmaps = ['YlOrRd', 'PuRd', 'Blues', 'Greens']

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()
    for i, (elem, cmap) in enumerate(zip(elements_carte, cmaps)):
        ax = axes[i]
        seuil, masque = anomalies(data[elem], facteur=facteur_mad)
        vals = data[elem]
        norm = mcolors.LogNorm(vmin=vals[vals > 0].min(), vmax=vals.max())
        sc = ax.scatter(data.X[~masque], data.Y[~masque], c=vals[~masque],
                        cmap=cmap, norm=norm, s=12, alpha=0.5)
        ax.scatter(data.X[masque], data.Y[masque], c=vals[masque],
                   cmap=cmap, norm=norm, s=50, alpha=0.9,
                   edgecolors='black', linewidths=0.7, zorder=5)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(f'{elem} — {masque.sum()} anomalies')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    fig.suptitle(f'Anomalies géochimiques — seuil Médiane+{facteur_mad}×MAD')
    plt.tight_layout()
    plt.savefig(file_output(key="anomalies", value="outputs/02_anomalies.png", make_path=True),
                dpi=130, bbox_inches='tight')
    plt.close()

    # ── 4. Analyse MNT ───────────────────────────────────────────────────

    Logger.info("Calcul des dérivés topographiques...")
    mnt_pente = calc_pente(mnt)
    mnt_aspect = calc_aspect(mnt)
    mnt_hs = calc_hillshade(mnt)
    Logger.info(f"Pente moyenne : {np.nanmean(mnt_pente):.1f}°  |  max : {np.nanmax(mnt_pente):.1f}°")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(mnt_hs, cmap='gray', extent=extent, origin='upper')
    im0 = axes[0].imshow(mnt, cmap='terrain', extent=extent, origin='upper', alpha=0.55)
    plt.colorbar(im0, ax=axes[0], label='Valeur relative (0–255)', shrink=0.8)
    axes[0].set_title('Image MNT avec ombrage')
    im1 = axes[1].imshow(mnt_pente, cmap='RdYlGn_r', extent=extent, origin='upper', vmax=30)
    plt.colorbar(im1, ax=axes[1], label='Pente (°)', shrink=0.8)
    axes[1].set_title('Pentes')
    im2 = axes[2].imshow(mnt_aspect, cmap='hsv', extent=extent, origin='upper')
    plt.colorbar(im2, ax=axes[2], label='Aspect (°)', shrink=0.8)
    axes[2].set_title('Aspect (orientation des versants)')
    for ax in axes:
        ax.scatter(data.X, data.Y, c='white', s=6, alpha=0.35, zorder=3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    fig.suptitle('Analyse topographique du MNT, Ambazac (valeurs relatives)')
    plt.tight_layout()
    plt.savefig(file_output(key="mnt", value="outputs/03_MNT.png", make_path=True),
                dpi=130, bbox_inches='tight')
    plt.close()

    # ── 5. Corrélations et ACP ────────────────────────────────────────────

    Logger.info("Calcul de la matrice de corrélation...")
    df_log = data[indicateurs].apply(lambda x: np.log10(x.replace(0, np.nan))).dropna()
    corr = df_log.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.4, annot_kws={'size': 8})
    ax.set_title('Corrélations entre indicateurs (log₁₀)')
    plt.tight_layout()
    plt.savefig(file_output(key="correlations", value="outputs/04_correlations.png", make_path=True),
                dpi=130, bbox_inches='tight')
    plt.close()

    Logger.info("Corrélations avec Au :")
    Logger.info(f"\n{corr['Au_ppb'].drop('Au_ppb').sort_values(key=abs, ascending=False).round(3).to_string()}")

    Logger.info("Calcul de l'ACP...")
    df_acp = data[elements].apply(lambda x: np.log10(x.replace(0, np.nan))).dropna()
    idx = df_acp.index
    X = StandardScaler().fit_transform(df_acp)
    acp = PCA(n_components=6)
    scores_acp = acp.fit_transform(X)
    loadings = acp.components_.T
    var = acp.explained_variance_ratio_ * 100

    for i, v in enumerate(var):
        Logger.info(f"  PC{i+1} : {v:.1f}%  (cumulé : {var[:i+1].sum():.1f}%)")

    fig, ax = plt.subplots(figsize=(9, 7))
    au = data.loc[idx, 'Au_ppb'].values
    sc = ax.scatter(scores_acp[:, 0], scores_acp[:, 1],
                    c=np.log10(au + 0.01), cmap='YlOrRd', s=18, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='log₁₀(Au ppb)')
    poids_arr = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top10 = np.argsort(poids_arr)[-10:]
    scale = 10
    for i in top10:
        ax.annotate('', xy=(loadings[i, 0]*scale, loadings[i, 1]*scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='navy', lw=1.2))
        ax.text(loadings[i, 0]*scale*1.12, loadings[i, 1]*scale*1.12,
                elements[i].split('_')[0], fontsize=8, color='navy')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)')
    ax.set_title('ACP — colorié par teneur en Au')
    plt.tight_layout()
    plt.savefig(file_output(key="acp", value="outputs/05_ACP.png", make_path=True),
                dpi=130, bbox_inches='tight')
    plt.close()

    # ── 6. Carte de potentiel minéral ─────────────────────────────────────

    Logger.info(f"Calcul du score de potentiel (Au={poids_au}, As={poids_as}, W={poids_w}, Bi={poids_bi})...")

    # Normalisation des poids (au cas où leur somme != 1)
    total = poids_au + poids_as + poids_w + poids_bi
    if total == 0:
        raise ValueError("La somme des poids ne peut pas être nulle.")
    p_au = poids_au / total
    p_as = poids_as / total
    p_w  = poids_w  / total
    p_bi = poids_bi / total

    score = (
        normaliser(data['Au_ppb']) * p_au +
        normaliser(data['As_ppm']) * p_as +
        normaliser(data['W_ppm'])  * p_w  +
        normaliser(data['Bi_ppm']) * p_bi
    ).fillna(0)

    seuil_top = score.quantile(quantile_top)
    top = data[score > seuil_top]

    Logger.info(f"Score : min={score.min():.3f}  max={score.max():.3f}  moy={score.mean():.3f}")
    Logger.info(f"Top {(1-quantile_top)*100:.0f}% : {len(top)} points (seuil={seuil_top:.3f})")

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(mnt_hs, cmap='gray', extent=extent, origin='upper')
    ax.imshow(mnt, cmap='terrain', extent=extent, origin='upper', alpha=0.35)
    sc = ax.scatter(data.X, data.Y, c=score, cmap='hot_r', s=25, alpha=0.85,
                    vmin=score.quantile(0.1), vmax=score.quantile(0.95), zorder=4)
    ax.scatter(top.X, top.Y, c='cyan', s=70, marker='*',
               edgecolors='black', linewidths=1.0, zorder=5,
               label=f'Top {(1-quantile_top)*100:.0f}% ({len(top)} pts)')
    plt.colorbar(sc, ax=ax, label='Score de potentiel', shrink=0.8)
    ax.legend(fontsize=9)
    ax.set_title(
        f'Potentiel minéral : Au({p_au*100:.0f}%) + As({p_as*100:.0f}%) '
        f'+ W({p_w*100:.0f}%) + Bi({p_bi*100:.0f}%)'
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.tight_layout()
    plt.savefig(file_output(key="potentiel_mineral", value="outputs/06_potentiel_mineral.png", make_path=True),
                dpi=130, bbox_inches='tight')
    plt.close()

    Logger.info("Analyse terminée — tous les fichiers de sortie sont dans outputs/")
