"""Étape 2.2 : convertir les tuiles « perm_water » de Sen1Floods11 au format du projet.

Constat (26/10/2026) : le radar de perm_water n'est pas en dB mais ramené entre 0 et 1
par une transformation non documentée, et les tuiles ne font pas 512 x 512.
Il n'y a pas de Sentinel-2.

Conversion approximative en dB, par bande, calée sur les médianes « terre » et
« eau permanente » (JRC) de Sen1Floods11 HandLabeled :
    VV_dB ≈ 57,9 · x − 55,0        VH_dB ≈ 45,5 · x − 46,4
Précision estimée : 2 à 3 dB. Ces tuiles servent donc uniquement à mesurer les
fausses alertes du mode « radar seul » sur de l'eau normale dans le monde entier,
pas à l'entraînement (le modèle pourrait apprendre la source au lieu de la classe).

Usage : python src/convertir_perm_water.py
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import S1F11  # noqa: E402

PENTE = np.array([57.9, 45.5], dtype="float32")[:, None, None]
ORIGINE = np.array([-55.0, -46.4], dtype="float32")[:, None, None]
TAILLE = 512
ENTREE = S1F11 / "perm_water" / "S1Perm"
SORTIE = S1F11 / "perm_water_db"


def recadrer(a, remplissage):
    sortie = np.full((a.shape[0], TAILLE, TAILLE), remplissage, dtype="float32")
    h, w = min(TAILLE, a.shape[1]), min(TAILLE, a.shape[2])
    y0, x0 = (a.shape[1] - h) // 2, (a.shape[2] - w) // 2
    sortie[:, :h, :w] = a[:, y0:y0 + h, x0:x0 + w]
    return sortie


def lisible(chemin):
    """Vrai si le fichier s'ouvre et se lit entièrement (détecte les fichiers tronqués)."""
    import rasterio
    try:
        with rasterio.open(chemin) as src:
            return src.read().shape == (2, TAILLE, TAILLE)
    except Exception:
        return False


def main():
    import rasterio
    SORTIE.mkdir(parents=True, exist_ok=True)
    fichiers = sorted(ENTREE.glob("sentinel_*.tif"))
    for k, f in enumerate(fichiers, 1):
        cible = SORTIE / f.name.replace("sentinel_", "perm_").replace(".tif", "_S1.tif")
        if cible.exists() and lisible(cible):
            continue
        with rasterio.open(f) as src:
            x = src.read().astype("float32")
            profil = src.profile
        db = np.where((x > 0) & np.isfinite(x), PENTE * x + ORIGINE, np.nan)
        db = recadrer(db, np.nan)
        profil.update(height=TAILLE, width=TAILLE, count=2, dtype="float32", compress="deflate", nodata=None)
        # écriture dans un fichier temporaire puis renommage : un arrêt brutal
        # ne laisse jamais de fichier à moitié écrit sous le nom final
        temp = cible.with_suffix(".tmp.tif")
        with rasterio.open(temp, "w", **profil) as dst:
            dst.write(db)
        os.replace(temp, cible)
        if k % 100 == 0:
            print(f"{k}/{len(fichiers)}")
    print(f"{len(fichiers)} tuiles -> {SORTIE}")


if __name__ == "__main__":
    main()
