"""Étape 5 : tableau comparatif et figures de résultats à partir de docs/resultats_modeles.json.

Sorties : docs/tableau_comparatif.md, docs/figures/comparaison_modeles.png,
          docs/figures/confusions.png
Usage : python src/tableau.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DOCS  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NOMS = {"cnn_rgb": "Petit CNN (RGB)", "mobilenet_rgb": "MobileNetV3 pré-entraîné (RGB)",
        "yolo_rgb": "YOLOv8n, rectangles (RGB)", "clip_zero_shot": "CLIP sans entraînement (RGB)",
        "cnn_complet": "Petit CNN (13 bandes + radar)"}
ORDRE = ["cnn_rgb", "mobilenet_rgb", "yolo_rgb", "clip_zero_shot", "cnn_complet"]


def pct(v):
    return "n.d." if v is None else f"{100 * v:.0f} %"


def f2(v):
    return "n.d." if v is None else f"{v:.2f}"


def main():
    r = json.loads((DOCS / "resultats_modeles.json").read_text(encoding="utf-8"))
    modeles = [m for m in ORDRE if m in r]
    lignes = ["| Modèle | F1 macro | Inondation P / R | Feu P / R | Fausses alertes | Inconnu (AUROC) | Taille | ms / image (1 cœur) |",
              "|---|---|---|---|---|---|---|---|"]
    for m in modeles:
        t = r[m]["test"]
        emb = r[m].get("embarque", {})
        e = emb.get("int8") or emb.get("fp32") or {}
        taille = f"{e.get('taille_mo')} Mo" + (" (INT8)" if "int8" in emb else "")
        ms = e.get("ms_par_image_1coeur") or e.get("ms_par_image_cpu")
        inc = r[m].get("inconnu", {}).get("auroc_test")
        lignes.append(f"| {NOMS[m]} | {f2(t['f1_macro'])} | {pct(t['INONDATION']['precision'])} / {pct(t['INONDATION']['rappel'])} "
                      f"| {pct(t['FEU']['precision'])} / {pct(t['FEU']['rappel'])} | {pct(t['fausses_alertes'])} "
                      f"| {f2(inc)} | {taille} | {ms if ms is not None else 'n.d.'} |")
    texte = ["# Tableau comparatif (test : pays jamais vus)", "",
             "Test : 35 inondations (Bolivie, Sri Lanka, Somalie), 4 feux (Grèce, Sardaigne, Australie ×2), "
             "41 tuiles normales. P = précision, R = rappel. Fausses alertes = part des tuiles normales "
             "déclarées INONDATION ou FEU.", ""] + lignes + [""]
    if "cnn_complet" in r and "eau_permanente_radar_seul" in r["cnn_complet"]:
        p = r["cnn_complet"]["eau_permanente_radar_seul"]
        texte += [f"Radar seul (cnn_complet, optique masquée) sur {p['n']} tuiles d'eau permanente dans le monde : "
                  f"{pct(p['fausses_alertes'])} de fausses alertes. Attention : en radar seul, ce modèle n'alerte jamais "
                  f"(rappel 0 sur les inondations de test, voir docs/resultats_capteurs.json) : ce chiffre ne prouve rien.", ""]
    (DOCS / "tableau_comparatif.md").write_text("\n".join(texte), encoding="utf-8")
    print("\n".join(texte))

    # Figure : F1 et fausses alertes par modèle
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    noms = [NOMS[m].replace(" (", "\n(") for m in modeles]
    couleurs = ["#c98a2e" if "complet" in m else "#2f6db3" for m in modeles]
    ax[0].barh(noms, [r[m]["test"]["f1_macro"] for m in modeles], color=couleurs)
    ax[0].set(title="F1 macro (plus haut = mieux)", xlim=(0, 1))
    ax[1].barh(noms, [r[m]["test"]["fausses_alertes"] for m in modeles], color=couleurs)
    ax[1].set(title="Fausses alertes sur tuiles normales (plus bas = mieux)", xlim=(0, 1))
    ax[1].set_yticklabels([])
    for a in ax:
        a.invert_yaxis(); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(DOCS / "figures" / "comparaison_modeles.png", dpi=110)

    # Figure : matrices de confusion
    fig, axes = plt.subplots(1, len(modeles), figsize=(3.6 * len(modeles), 3.6))
    for a, m in zip(np.atleast_1d(axes), modeles):
        c = np.array(r[m]["test"]["confusion"])
        a.imshow(c, cmap="Blues")
        for i in range(3):
            for j in range(3):
                a.text(j, i, c[i, j], ha="center", va="center", color="white" if c[i, j] > c.max() / 2 else "black")
        a.set_xticks(range(3), ["Normal", "Inond.", "Feu"]); a.set_yticks(range(3), ["Normal", "Inond.", "Feu"])
        a.set_title(NOMS[m].replace(" (", "\n("), fontsize=9); a.set_xlabel("prédit"); a.set_ylabel("vrai")
    fig.tight_layout(); fig.savefig(DOCS / "figures" / "confusions.png", dpi=110)


if __name__ == "__main__":
    main()
