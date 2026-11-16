"""Le modèle complet face à un seul capteur (scénario réel : un satellite = un capteur).

Évalue cnn_complet sur le test avec : radar + optique, radar seul (optique masquée),
optique seule (radar masqué), avec le seuil d'alerte réglé sur val.
Sortie : docs/resultats_capteurs.json

Usage : python src/evaluer_capteurs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402
from donnees import charger_stats  # noqa: E402
from entrainer import charger_ensemble, charger_modele, metriques, predire  # noqa: E402
from evaluer import decider_avec_seuil  # noqa: E402


def main():
    res_modele = json.loads((DOCS / "resultats_modeles.json").read_text(encoding="utf-8"))["cnn_complet"]
    seuil = res_modele["seuil_alerte"]
    modele = charger_modele("cnn_complet")
    index = pd.read_csv(DATA / "index.csv").fillna("")
    X, Y, _ = charger_ensemble(index, "test", "complet", charger_stats())
    sortie = {}
    for nom, masque in [("radar + optique", None), ("radar seul", slice(2, 15)), ("optique seule", slice(0, 2))]:
        Xm = X.copy()
        if masque is not None:
            Xm[:, masque] = 0
        m = metriques(Y, decider_avec_seuil(predire(modele, Xm), seuil))
        sortie[nom] = m
        print(f"{nom:16s} F1 {m['f1_macro']:.2f} | inondation P {m['INONDATION']['precision']} R {m['INONDATION']['rappel']:.2f}"
              f" | feu R {m['FEU']['rappel']:.2f} | fausses alertes {m['fausses_alertes']:.2f}")
    (DOCS / "resultats_capteurs.json").write_text(json.dumps(sortie, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
