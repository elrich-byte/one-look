"""Démonstration interactive (navigateur) : analyser une tuile jamais vue.

On choisit une tuile de l'ensemble test ou inconnu, ou on dépose ses propres GeoTIFF.
L'application affiche l'image en couleurs naturelles et en infrarouge (SWIR), puis la
décision du modèle RGB et du modèle complet (radar + 13 bandes), avec leurs probabilités.

Usage : python src/demo.py     puis ouvrir http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402
from decision import Detecteur  # noqa: E402
from donnees import lire  # noqa: E402
from rgb import composer  # noqa: E402

ICONES = {"INONDATION": "🌊 ALERTE INONDATION", "FEU": "🔥 ALERTE FEU", "NORMAL": "✅ NORMAL", "INCONNU": "❓ INCONNU"}


def exemples():
    index = pd.read_csv(DATA / "index.csv").fillna("")
    sel = index[index["ensemble"].isin(["test", "inconnu"]) & (index["s2"] != "")]
    return {f"{r['id']}  [vérité : {r['classe']}]": r for _, r in sel.iterrows()}


def texte(nom, r):
    lignes = [f"### {nom} : {ICONES[r['decision']]}"]
    if r["raison"]:
        lignes.append(f"*{r['raison']}*")
    if r["probabilites"]:
        lignes.append(" · ".join(f"{k} {v:.0%}" for k, v in r["probabilites"].items()))
    return "\n\n".join(lignes)


def main():
    import gradio as gr
    ex = exemples()
    modeles = {"Modèle RGB (3 bandes)": Detecteur("cnn_rgb"), "Modèle complet (radar + 13 bandes)": Detecteur("cnn_complet")}

    def analyser(choix, fichier_s2, fichier_s1):
        if fichier_s2 is not None:
            s2 = lire(fichier_s2); s1 = lire(fichier_s1) if fichier_s1 is not None else None
        else:
            r = ex[choix]
            s2 = lire(DATA / r["s2"]) if r["s2"] else None
            s1 = lire(DATA / r["s1"]) if r["s1"] else None
        images = [composer(s2, "rgb"), composer(s2, "swir")] if s2 is not None and s2.shape[0] == 13 else [None, None]
        return images[0], images[1], *[texte(n, m.decider(s1, s2)) for n, m in modeles.items()]

    with gr.Blocks(title="One Look") as app:
        gr.Markdown("# One Look : une image, un regard, une décision\nIA embarquée : **inondation, feu, normal ou inconnu**.")
        with gr.Row():
            choix = gr.Dropdown(list(ex), value=list(ex)[0], label="Tuile de test (pays jamais vus) ou inconnue")
            f2 = gr.File(label="… ou votre GeoTIFF Sentinel-2 (13 bandes)")
            f1 = gr.File(label="GeoTIFF Sentinel-1 (VV, VH en dB, facultatif)")
        bouton = gr.Button("Analyser", variant="primary")
        with gr.Row():
            img_rgb = gr.Image(label="Couleurs naturelles (RGB)")
            img_swir = gr.Image(label="Infrarouge (SWIR B12 B8A B4)")
        with gr.Row():
            sortie_rgb = gr.Markdown(); sortie_complet = gr.Markdown()
        bouton.click(analyser, [choix, f2, f1], [img_rgb, img_swir, sortie_rgb, sortie_complet])
    app.launch()


if __name__ == "__main__":
    main()
