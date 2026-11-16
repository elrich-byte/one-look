"""Étape 3 : comprendre Sentinel-2 (quelles bandes voient l'eau, que montre le RGB).

Produit :
- docs/figures/signatures_s2.png : réflectance moyenne par bande pour l'eau de crue,
  l'eau permanente et la terre (pixels annotés de Sen1Floods11, ensemble train) ;
- docs/separabilite_eau.csv : pour chaque bande ou indice, capacité à séparer
  eau / terre (AUC : 0,5 = hasard, 1 = parfait) ;
- docs/figures/compositions_s2.png : mêmes tuiles en RGB, infrarouge et SWIR.

Usage : python src/explorer_s2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BANDES_S2, DATA, DOCS  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CACHE = DATA / "cache"
LONGUEURS_ONDE = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]  # nm


def pixels_train(max_par_tuile=1500, graine=0):
    rng = np.random.default_rng(graine)
    index = pd.read_csv(DATA / "index.csv").fillna("")
    train = index[(index["source"] == "sen1floods11") & (index["ensemble"] == "train")]
    X2, X1, Y = [], [], []
    for i in train["id"]:
        d = np.load(CACHE / f"{i}.npz")
        s2, s1, m, j = d["s2"], d["s1"], d["masque"], d["jrc"]
        ok = (m != -1) & np.isfinite(s1).all(0) & (s2 != 0).any(0)
        idx = np.flatnonzero(ok.ravel())
        if idx.size == 0:
            continue
        idx = rng.choice(idx, min(max_par_tuile, idx.size), replace=False)
        classe = np.where(m.ravel()[idx] == 0, 0, np.where(j.ravel()[idx] == 1, 2, 1))  # 0 terre, 1 crue, 2 perm
        X2.append(s2.reshape(13, -1)[:, idx].T / 10000.0)
        X1.append(s1.reshape(2, -1)[:, idx].T)
        Y.append(classe)
    return np.concatenate(X2), np.concatenate(X1), np.concatenate(Y)


def indice(a, b):
    return (a - b) / (a + b + 1e-6)


def main():
    from sklearn.metrics import roc_auc_score
    X2, X1, Y = pixels_train()
    print(f"{len(Y)} pixels : terre {np.mean(Y == 0):.0%}, crue {np.mean(Y == 1):.0%}, permanente {np.mean(Y == 2):.0%}")

    # 1. Signatures spectrales
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c, nom, coul in [(0, "terre", "#8c6d31"), (1, "eau de crue", "#3b6ea5"), (2, "eau permanente", "#16324f")]:
        moy = X2[Y == c].mean(0)
        ax.plot(LONGUEURS_ONDE, moy, "o-", label=nom, color=coul)
    for x0, x1, coul in [(450, 690, "#f2f2f2")]:
        ax.axvspan(x0, x1, color=coul, zorder=0)
    ax.text(470, ax.get_ylim()[1] * 0.92, "visible (RGB)", fontsize=9)
    for lo, b in zip(LONGUEURS_ONDE, BANDES_S2):
        ax.annotate(b, (lo, 0), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7, annotation_clip=False)
    ax.set(xlabel="longueur d'onde (nm)", ylabel="réflectance moyenne",
           title="Signature spectrale Sentinel-2 (Sen1Floods11, train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "figures" / "signatures_s2.png", dpi=110)

    # 2. Séparabilité eau / terre
    eau = (Y > 0).astype(int)
    r, g, b = X2[:, 3], X2[:, 2], X2[:, 1]
    caract = {f"S2 {n}": X2[:, k] for k, n in enumerate(BANDES_S2)}
    caract.update({
        "S1 VV": X1[:, 0], "S1 VH": X1[:, 1],
        "RGB luminosité (R+G+B)/3": (r + g + b) / 3,
        "RGB bleu - rouge": b - r,
        "NDWI (B3, B8)": indice(X2[:, 2], X2[:, 7]),
        "MNDWI (B3, B11)": indice(X2[:, 2], X2[:, 11]),
        "NDVI (B8, B4)": indice(X2[:, 7], X2[:, 3]),
    })
    lignes = []
    for nom, v in caract.items():
        auc = roc_auc_score(eau, v)
        lignes.append(dict(caracteristique=nom, auc=round(max(auc, 1 - auc), 3),
                           sens="plus bas = eau" if auc < 0.5 else "plus haut = eau",
                           famille="RGB" if nom.startswith("RGB") or nom in ("S2 B2", "S2 B3", "S2 B4") else
                           "radar" if nom.startswith("S1") else "S2 hors visible"))
    df = pd.DataFrame(lignes).sort_values("auc", ascending=False)
    df.to_csv(DOCS / "separabilite_eau.csv", index=False)
    print(df.to_string(index=False))

    # 3. Compositions colorées
    from donnees import lire
    from rgb import composer
    tuiles = [("USA_770353", "sen1floods11/S2Hand/USA_770353_S2Hand.tif", "crue autour d'un fleuve"),
              ("Pakistan_474121", "sen1floods11/S2Hand/Pakistan_474121_S2Hand.tif", "nuages"),
              ("USA_181503", "sen1floods11/S2Hand/USA_181503_S2Hand.tif", "champs sombres"),
              ("feu_evia_2021", "scenes/feu_evia_2021_S2.tif", "feu (Grèce)"),
              ("feu_blue_mountains_2020", "scenes/feu_blue_mountains_2020_S2.tif", "feu (Australie)")]
    fig, axes = plt.subplots(len(tuiles), 3, figsize=(10, 3.3 * len(tuiles)))
    for k, (ident, chemin, desc) in enumerate(tuiles):
        s2 = lire(DATA / chemin)
        for c, nom in enumerate(["rgb", "infrarouge", "swir"]):
            axes[k, c].imshow(composer(s2, nom))
            axes[k, c].axis("off")
            if k == 0:
                axes[k, c].set_title({"rgb": "RGB (B4 B3 B2)", "infrarouge": "Infrarouge (B8 B4 B3)",
                                      "swir": "SWIR (B12 B8A B4)"}[nom])
        axes[k, 0].text(-12, 256, f"{ident}\n{desc}", rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "figures" / "compositions_s2.png", dpi=90)
    print("figures -> docs/figures/signatures_s2.png, compositions_s2.png")


if __name__ == "__main__":
    main()
