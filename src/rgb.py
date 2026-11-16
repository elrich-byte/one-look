"""Conversion des images Sentinel-2 en images couleur (RGB) et en fausses couleurs.

Réglage de luminosité FIXE, identique pour toutes les images (pas d'ajustement
image par image, qui changerait l'apparence d'une image à l'autre et fausserait
la comparaison entre modèles) : réflectance x GAIN, écrêtée entre 0 et 1.

Usage : python src/rgb.py --exporter    # PNG de toutes les tuiles de l'index -> data/rgb/
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402

GAIN = 3.5          # réflectance 0,29 -> blanc ; les paysages courants vont de 0,03 à 0,25
# Indices des bandes dans les 13 bandes S2 : B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12
B2, B3, B4, B8, B8A, B11, B12 = 1, 2, 3, 7, 8, 11, 12

COMPOSITIONS = {
    "rgb": (B4, B3, B2),          # couleurs naturelles
    "infrarouge": (B8, B4, B3),   # végétation en rouge, eau très sombre
    "swir": (B12, B8A, B4),       # zones brûlées très visibles, eau noire
}


def composer(s2, nom="rgb", gain=GAIN):
    """s2 : (13, H, W) en réflectance x 10 000 -> image (H, W, 3) uint8."""
    bandes = COMPOSITIONS[nom]
    a = np.stack([s2[b] for b in bandes], axis=-1).astype("float32") / 10000.0
    a = np.nan_to_num(a)
    return (np.clip(a * gain, 0, 1) * 255).round().astype("uint8")


def rgb_float(s2):
    """Version flottante (3, H, W) entre 0 et 1, entrée des modèles « RGB seul »."""
    a = np.stack([s2[B4], s2[B3], s2[B2]]).astype("float32") / 10000.0
    return np.clip(np.nan_to_num(a) * GAIN, 0, 1)


def main():
    import pandas as pd
    from PIL import Image
    from donnees import charger

    ap = argparse.ArgumentParser()
    ap.add_argument("--exporter", action="store_true")
    args = ap.parse_args()
    if not args.exporter:
        return
    index = pd.read_csv(DATA / "index.csv").fillna("")
    index = index[index["s2"] != ""]
    sortie = DATA / "rgb"
    sortie.mkdir(exist_ok=True)
    for k, (_, ligne) in enumerate(index.iterrows(), 1):
        cible = sortie / f"{ligne['id']}.png"
        if cible.exists():
            continue
        _, s2 = charger(ligne)
        Image.fromarray(composer(s2)).save(cible)
        if k % 50 == 0:
            print(f"{k}/{len(index)}", flush=True)
    print(f"{len(index)} images -> {sortie}")


if __name__ == "__main__":
    main()
