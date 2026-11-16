"""Étape 4 : modèle vision-langage sans entraînement (CLIP, Hugging Face).

CLIP compare une image à des phrases (« une image satellite d'une zone inondée »…)
et choisit la plus proche : aucun apprentissage sur nos données (« zero-shot »).
Il sert de comparaison avec nos petits modèles : 600 Mo, trop gros pour un satellite.

Usage : python -m pip install transformers
        python src/clip_zero_shot.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402
from entrainer import CACHE, CLASSES, metriques  # noqa: E402
from rgb import composer  # noqa: E402

MODELE = "openai/clip-vit-base-patch32"
PHRASES = {
    "NORMAL": ["a satellite image of farmland and fields", "a satellite image of a green forest",
               "a satellite image of a river and fields", "a satellite image of a lake",
               "a satellite image of a town", "a satellite image of clouds", "a satellite image of desert"],
    "INONDATION": ["a satellite image of a flooded area", "a satellite image of flood water covering fields",
                   "a satellite image of an inundated river plain"],
    "FEU": ["a satellite image of a burned forest after a wildfire", "a satellite image of a dark burn scar",
            "a satellite image of land burned by fire"],
}


def vecteur(sortie):
    """Selon la version de transformers, get_*_features renvoie un tenseur ou un objet."""
    if hasattr(sortie, "norm"):
        return sortie
    for nom in ("text_embeds", "image_embeds", "pooler_output"):
        if getattr(sortie, nom, None) is not None:
            return getattr(sortie, nom)
    raise TypeError(type(sortie))


def main():
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    modele = CLIPModel.from_pretrained(MODELE).eval()
    proc = CLIPProcessor.from_pretrained(MODELE)
    phrases = [p for c in CLASSES for p in PHRASES[c]]
    appartenance = np.array([CLASSES.index(c) for c in CLASSES for _ in PHRASES[c]])
    with torch.no_grad():
        t = vecteur(modele.get_text_features(**proc(text=phrases, return_tensors="pt", padding=True)))
        t = (t / t.norm(dim=-1, keepdim=True)).numpy()
    index = pd.read_csv(DATA / "index.csv").fillna("")
    res = dict(modele="clip_zero_shot", entree="rgb", parametres=int(sum(p.numel() for p in modele.parameters())))
    duree = []
    for e in ("val", "test"):
        l = index[(index["ensemble"] == e) & index["classe"].isin(CLASSES) & (index["s2"] != "")]
        l = l[[(CACHE / f"{i}.npz").exists() for i in l["id"]]]
        y, p = [], []
        for _, r in l.iterrows():
            img = Image.fromarray(composer(np.load(CACHE / f"{r['id']}.npz")["s2"]))
            t0 = time.perf_counter()
            with torch.no_grad():
                v = vecteur(modele.get_image_features(**proc(images=img, return_tensors="pt")))
            duree.append(time.perf_counter() - t0)
            v = (v / v.norm(dim=-1, keepdim=True)).numpy()[0]
            sim = t @ v
            p.append(int(np.argmax([sim[appartenance == k].mean() for k in range(3)])))
            y.append(CLASSES.index(r["classe"]))
        res[e] = metriques(np.array(y), np.array(p))
    res["embarque"] = {"fp32": dict(taille_mo=round(res["parametres"] * 4 / 1e6, 1),
                                    ms_par_image_1coeur=None, ms_par_image_cpu=round(1000 * np.median(duree), 1))}
    fichier = DOCS / "resultats_modeles.json"
    tout = json.loads(fichier.read_text(encoding="utf-8")) if fichier.exists() else {}
    tout["clip_zero_shot"] = res
    fichier.write_text(json.dumps(tout, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"CLIP test : F1 {res['test']['f1_macro']:.2f} | fausses alertes {res['test']['fausses_alertes']:.2f}")


if __name__ == "__main__":
    main()
