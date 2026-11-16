"""Vidéo de démonstration (MP4) : le modèle analyse des tuiles jamais vues.

Pour chaque tuile : image RGB, image infrarouge, vérité, décision du modèle RGB et du
modèle complet. Sortie : docs/demo_one_look.mp4

Usage : python src/video_demo.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402
from decision import Detecteur  # noqa: E402
from donnees import lire  # noqa: E402
from rgb import composer  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COULEURS = {"INONDATION": "#2f6db3", "FEU": "#c4501b", "NORMAL": "#2e8b57", "INCONNU": "#7a5aa6"}
SIGNES = {"INONDATION": "ALERTE INONDATION", "FEU": "ALERTE FEU", "NORMAL": "NORMAL", "INCONNU": "INCONNU"}


def image_titre(texte, sous_titre):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0e1726")
    fig.text(0.5, 0.56, texte, color="white", ha="center", fontsize=40, weight="bold")
    fig.text(0.5, 0.42, sous_titre, color="#b8c4d6", ha="center", fontsize=18)
    return fig


def image_tuile(ligne, s1, s2, d_rgb, d_cpl):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0e1726")
    for k, (nom, titre) in enumerate([("rgb", "Ce que voit une caméra couleur (RGB)"), ("swir", "Infrarouge (SWIR)")]):
        ax = fig.add_axes([0.03 + 0.33 * k, 0.12, 0.31, 0.62])
        ax.imshow(composer(s2, nom)); ax.axis("off")
        ax.set_title(titre, color="white", fontsize=12)
    fig.text(0.03, 0.9, ligne["id"], color="white", fontsize=20, weight="bold")
    fig.text(0.03, 0.84, f"Vérité : {ligne['classe']}  ·  {ligne['categorie']}  ·  ensemble {ligne['ensemble']}",
             color="#b8c4d6", fontsize=13)
    for k, (nom, d) in enumerate([("Modèle RGB", d_rgb), ("Modèle complet", d_cpl)]):
        y = 0.62 - 0.3 * k
        fig.text(0.7, y + 0.08, nom, color="#b8c4d6", fontsize=14)
        fig.text(0.7, y, SIGNES[d["decision"]], color="white", fontsize=22, weight="bold",
                 bbox=dict(facecolor=COULEURS[d["decision"]], edgecolor="none", pad=8))
        if d["probabilites"]:
            fig.text(0.7, y - 0.07, "  ".join(f"{c[:4]} {v:.0%}" for c, v in d["probabilites"].items()),
                     color="#b8c4d6", fontsize=11)
    return fig


def vers_tableau(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def main():
    import imageio.v2 as imageio
    index = pd.read_csv(DATA / "index.csv").fillna("")
    rgb, cpl = Detecteur("cnn_rgb"), Detecteur("cnn_complet")
    choix = []
    for classe, n in [("INONDATION", 4), ("FEU", 3), ("NORMAL", 4), ("INCONNU", 2)]:
        l = index[(index["classe"] == classe) & index["ensemble"].isin(["test", "inconnu"]) & (index["s2"] != "")]
        choix += list(l.sample(min(n, len(l)), random_state=1).itertuples(index=False))
    images = [vers_tableau(image_titre("One Look", "Une image, un regard, une décision — à bord du satellite"))] * 3
    for ligne in choix:
        ligne = ligne._asdict()
        s2 = lire(DATA / ligne["s2"]); s1 = lire(DATA / ligne["s1"]) if ligne["s1"] else None
        images += [vers_tableau(image_tuile(ligne, s1, s2, rgb.decider(s1, s2), cpl.decider(s1, s2)))] * 3
    images += [vers_tableau(image_titre("Pays jamais vus", "Bolivie · Sri Lanka · Somalie · Grèce · Sardaigne · Turquie · Australie"))] * 3
    imageio.mimsave(DOCS / "demo_one_look.mp4", images, fps=1, codec="libx264", quality=8)
    print(f"{len(images)} images -> {DOCS / 'demo_one_look.mp4'}")


if __name__ == "__main__":
    main()
