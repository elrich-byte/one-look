"""Étape 2.4 : exporter quelques zones brûlées de CaBuAr comme scènes « inconnues ».

Un incendie est un phénomène que le modèle n'a jamais vu : il doit répondre INCONNU
(ou au moins ne pas lever d'alerte inondation). CaBuAr ne contient que Sentinel-2
(12 bandes, sans B10) : on insère une bande B10 nulle pour obtenir 13 bandes.

Usage : python -m pip install datasets hdf5plugin
        python src/exporter_cabuar.py --n 15
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402

SORTIE = DATA / "scenes"


def trouver(item, nb_canaux=None, dims=None):
    for k, v in item.items():
        a = np.asarray(v)
        if dims and a.ndim != dims:
            continue
        if nb_canaux is None or nb_canaux in a.shape:
            return k, a
    return None, None


def main():
    import rasterio
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--min-brule", type=float, default=5.0, help="%% minimal de zone brûlée")
    args = ap.parse_args()

    ds = load_dataset("DarthReca/california_burned_areas", name="post-fire", split="train", streaming=True)
    SORTIE.mkdir(parents=True, exist_ok=True)
    lignes, k = [], 0
    for item in ds:
        cle_img, img = trouver(item, nb_canaux=12, dims=3)
        cle_msk, msk = trouver({c: v for c, v in item.items() if c != cle_img}, dims=None)
        if img is None:
            print("Clés trouvées :", list(item.keys()))
            raise SystemExit("Image à 12 canaux introuvable : adapter le script.")
        if img.shape[-1] == 12:                      # (H, W, C) -> (C, H, W)
            img = np.moveaxis(img, -1, 0)
        msk = np.squeeze(np.asarray(item.get("mask", msk)))
        brule = 100 * float((msk > 0).mean())
        if brule < args.min_brule:
            continue
        img = img.astype("float32")
        s2 = np.insert(img, 10, 0, axis=0)[:, :512, :512]   # B10 nulle, 13 bandes
        ident = f"inc_cabuar_{k:02d}"
        with rasterio.open(SORTIE / f"{ident}_S2.tif", "w", driver="GTiff", height=s2.shape[1],
                           width=s2.shape[2], count=13, dtype="float32", compress="deflate") as dst:
            dst.write(s2)
        lignes.append(dict(id=ident, categorie="zone_brulee_cabuar", brule_pct=round(brule, 1)))
        print(f"{ident} : {brule:.1f} % brûlé")
        k += 1
        if k >= args.n:
            break
    with open(SORTIE / "manifest_cabuar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "categorie", "brule_pct"])
        w.writeheader()
        w.writerows(lignes)
    print(f"{k} scènes exportées.")


if __name__ == "__main__":
    main()
