"""Cache de tuiles réduites (1 pixel sur 4 -> 128 x 128) pour explorer et tester vite.

Garde les valeurs brutes (S1 en dB, S2 en réflectance x 10 000), le masque d'eau
et l'eau permanente. Écrit data/cache/<id>.npz. Reprend là où il s'est arrêté.

Usage : python src/cache.py            (python src/cache.py --perm pour inclure l'eau permanente)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402
from donnees import lire  # noqa: E402

PAS = 4
DOSSIER = DATA / "cache"


def main(limite_secondes=None):
    import time
    debut = time.time()
    index = pd.read_csv(DATA / "index.csv").fillna("")
    if "--perm" not in sys.argv:
        index = index[index["source"] != "sen1floods11_perm_water"]
    DOSSIER.mkdir(exist_ok=True)
    faits = 0
    for _, l in index.iterrows():
        cible = DOSSIER / f"{l['id']}.npz"
        if cible.exists():
            continue
        d = {}
        if l["s1"]:
            d["s1"] = lire(DATA / l["s1"])[:, ::PAS, ::PAS]
        if l["s2"]:
            d["s2"] = lire(DATA / l["s2"])[:, ::PAS, ::PAS]
        if l["masque"]:
            d["masque"] = lire(DATA / l["masque"])[0, ::PAS, ::PAS]
        if l["source"] == "sen1floods11":
            jrc = l["masque"].replace("LabelHand", "JRCWaterHand")
            d["jrc"] = lire(DATA / jrc)[0, ::PAS, ::PAS]
        temp = DOSSIER / f"{l['id']}.tmp.npz"
        np.savez_compressed(temp, **d)
        temp.replace(cible)
        faits += 1
        if limite_secondes and time.time() - debut > limite_secondes:
            break
    reste = sum(not (DOSSIER / f"{i}.npz").exists() for i in index["id"])
    print(f"{faits} tuiles ajoutées, {reste} restantes")


if __name__ == "__main__":
    nombres = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(float(nombres[0]) if nombres else None)
