"""Statistiques par tuile de Sen1Floods11 HandLabeled (étape 1.5).

Calcule pour chaque tuile la part d'eau, d'eau permanente, d'eau de crue,
de pixels invalides, de NaN radar et de nuages (heuristique), puis écrit
docs/stats_tuiles.csv et docs/figures/hist_crue.png.

Usage (depuis la racine du dépôt) :
    python src/stats_tuiles.py
"""
import csv
import glob
import os

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

ROOT = "data/sen1floods11"
OUT_CSV = "docs/stats_tuiles.csv"
OUT_FIG = "docs/figures/hist_crue.png"

DILATATION_JRC = 3     # pixels (~30 m) : tolérance autour de l'eau permanente
MIN_PIXELS_CRUE = 50   # composantes de crue plus petites ignorées (~0,5 ha)


def lire(dossier, nom, suffixe):
    with rasterio.open(f"{ROOT}/{dossier}/{nom}_{suffixe}.tif") as src:
        return src.read().astype("float32")


def splits_officiels():
    """Associe chaque tuile à son découpage officiel (train/valid/test/bolivia)."""
    correspondance = {}
    for fichier in glob.glob(f"{ROOT}/splits/flood_handlabeled/flood_*_data.csv"):
        split = os.path.basename(fichier).replace("flood_", "").replace("_data.csv", "")
        with open(fichier) as f:
            for ligne in csv.reader(f):
                if ligne:
                    correspondance[ligne[0].replace("_S1Hand.tif", "")] = split
    return correspondance


def stats_tuile(nom, split):
    lab = lire("LabelHand", nom, "LabelHand")[0]
    jrc = lire("JRCWaterHand", nom, "JRCWaterHand")[0]
    s1 = lire("S1Hand", nom, "S1Hand")
    s2 = lire("S2Hand", nom, "S2Hand")

    valide = lab != -1
    n = max(int(valide.sum()), 1)
    eau = (lab == 1) & valide
    perm = (jrc == 1) & valide
    crue = eau & ~perm

    # Variante robuste : eau permanente élargie de quelques pixels,
    # puis suppression des petites taches de "crue" (bords de rivières, mares).
    perm_dil = ndimage.binary_dilation(jrc == 1, iterations=DILATATION_JRC)
    crue_r = eau & ~perm_dil
    composantes, nb = ndimage.label(crue_r)
    if nb:
        tailles = ndimage.sum(crue_r, composantes, range(1, nb + 1))
        garder = np.isin(composantes, np.flatnonzero(tailles >= MIN_PIXELS_CRUE) + 1)
        crue_r = crue_r & garder

    # Heuristique simple de nuage sur S2 L1C : pixel très brillant en bleu (B2) et en rouge (B4).
    nuage = (s2[1] > 2000) & (s2[3] > 2000)

    return {
        "tile": nom,
        "pays": nom.split("_")[0],
        "split_officiel": split,
        "valide_pct": 100 * valide.mean(),
        "eau_pct": 100 * eau.sum() / n,
        "perm_pct": 100 * perm.sum() / n,
        "crue_pct": 100 * crue.sum() / n,
        "crue_robuste_pct": 100 * crue_r.sum() / n,
        "s1_nan_pct": 100 * np.isnan(s1).any(axis=0).mean(),
        "s2_vide_pct": 100 * (s2 == 0).all(axis=0).mean(),
        "nuage_pct": 100 * nuage.mean(),
    }


def main():
    splits = splits_officiels()
    noms = sorted(os.path.basename(p).replace("_LabelHand.tif", "")
                  for p in glob.glob(f"{ROOT}/LabelHand/*.tif"))
    df = pd.DataFrame([stats_tuile(nom, splits.get(nom, "?")) for nom in noms])
    df.round(3).to_csv(OUT_CSV, index=False)
    print(f"{len(df)} tuiles -> {OUT_CSV}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["crue_pct"], bins=50, color="#3b6ea5")
    ax[0].set(title="Crue = masque - JRC", xlabel="% de pixels valides en crue", ylabel="nombre de tuiles")
    ax[1].hist(df["crue_robuste_pct"], bins=50, color="#3b6ea5")
    ax[1].set(title=f"Crue robuste (JRC +{DILATATION_JRC} px, taches < {MIN_PIXELS_CRUE} px retirées)",
              xlabel="% de pixels valides en crue")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=110)
    print(f"Histogramme -> {OUT_FIG}")


if __name__ == "__main__":
    main()
