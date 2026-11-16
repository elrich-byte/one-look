"""Exporter des zones brûlées de CaBuAr au format du projet (classe FEU, + NORMAL).

CaBuAr (Hugging Face, DarthReca/california_burned_areas, fichier raw/patched/512x512.hdf5,
5,9 Go) contient 534 zones de 512 x 512 en Californie, chacune avec une image S2 L2A
avant l'incendie (pre_fire), après (post_fire) et le masque brûlé.
Le fichier est d'abord téléchargé en entier dans data/cabuar/ (une seule fois),
puis lu en local.

- post_fire avec au moins 2 % de brûlé -> classe FEU (<id>_S2.tif + <id>_masque.tif)
- pre_fire de la même zone -> classe NORMAL (cabuar_avant_<k>_S2.tif), même lieu sans feu

12 bandes (B1-B9, B8A, B11, B12, sans B10) : on insère une bande B10 nulle pour avoir
les 13 bandes du projet. Toutes ces zones sont en Californie : apprentissage seulement.

Usage : python src/exporter_cabuar.py --n 80
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402

SORTIE = DATA / "scenes"
FICHIER = "datasets/DarthReca/california_burned_areas/raw/patched/512x512.hdf5"


def ecrire(chemin, a, dtype="float32"):
    import rasterio
    with rasterio.open(chemin, "w", driver="GTiff", height=a.shape[1], width=a.shape[2],
                       count=a.shape[0], dtype=dtype, compress="deflate") as dst:
        dst.write(a.astype(dtype))


def vers_13_bandes(img):
    a = np.moveaxis(np.asarray(img), -1, 0).astype("float32")      # (12, H, W)
    return np.insert(a, 10, 0, axis=0)                              # B10 nulle


def main():
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    import h5py
    import hdf5plugin  # noqa: F401  (filtre de compression BZip2)

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--min-brule", type=float, default=2.0, help="%% minimal de zone brûlée")
    args = ap.parse_args()

    SORTIE.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    local = hf_hub_download("DarthReca/california_burned_areas", "raw/patched/512x512.hdf5",
                            repo_type="dataset", local_dir=DATA / "cabuar")
    h = h5py.File(local, "r")
    cles = sorted(h.keys())
    rng = np.random.default_rng(0)
    rng.shuffle(cles)                     # tirage varié parmi les incendies et les plis
    lignes, k = [], 0
    for cle in cles:
        g = h[cle]
        masque = np.asarray(g["mask"])[..., 0] > 0
        brule = 100 * float(masque.mean())
        if brule < args.min_brule:
            continue
        ident = f"cabuar_{k:03d}"
        if not (SORTIE / f"{ident}_S2.tif").exists():
            ecrire(SORTIE / f"{ident}_S2.tif", vers_13_bandes(g["post_fire"]))
            ecrire(SORTIE / f"{ident}_masque.tif", masque[None].astype("uint8"), "uint8")
            if "pre_fire" in g:
                ecrire(SORTIE / f"cabuar_avant_{k:03d}_S2.tif", vers_13_bandes(g["pre_fire"]))
        lignes.append(dict(id=ident, cle=cle, pli=int(g.attrs.get("fold", -1)), brule_pct=round(brule, 1)))
        print(f"{ident} ({cle}) : {brule:.1f} % brûlé", flush=True)
        k += 1
        if k >= args.n:
            break
    with open(SORTIE / "manifest_cabuar.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=["id", "cle", "pli", "brule_pct"])
        w.writeheader()
        w.writerows(lignes)
    print(f"{k} zones exportées.")


if __name__ == "__main__":
    main()
