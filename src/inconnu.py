"""Étape 2.4 : entrées « inconnues » synthétiques, générées à partir de tuiles de test.

Elles testent la capacité du système à répondre INCONNU plutôt que NORMAL ou ALERTE.
Deux familles :
- format invalide : doit être rejeté par donnees.verifier() avant le modèle ;
- format valide mais contenu absurde : doit être rattrapé par le score de confiance
  ou de détection hors distribution du modèle (étape 4).

Usage : python src/inconnu.py   (vérifie que chaque type se comporte comme prévu)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donnees import EntreeInvalide, charger, charger_stats, preparer  # noqa: E402

# type -> famille attendue
TYPES = {
    "bandes_manquantes": "format",     # S2 avec 4 bandes (image RGB+NIR)
    "valeurs_brutes_s1": "format",     # S1 en valeurs linéaires au lieu de dB
    "aucun_capteur": "format",         # S1 et S2 vides
    "bruit_aleatoire": "contenu",      # bruit blanc au bon format
    "image_constante": "contenu",      # tuile uniforme
    "nuage_total_sans_s1": "contenu",  # S2 entièrement blanc, pas de radar
    "bandes_melangees": "contenu",     # ordre des bandes S2 permuté
}


def generer(type_, s1, s2, graine=0):
    rng = np.random.default_rng(graine)
    if type_ == "bandes_manquantes":
        return None, s2[[1, 2, 3, 7]]
    if type_ == "valeurs_brutes_s1":
        return 10 ** (s1 / 10), s2
    if type_ == "aucun_capteur":
        return np.full_like(s1, np.nan), np.zeros_like(s2)
    if type_ == "bruit_aleatoire":
        return rng.uniform(-30, 0, s1.shape).astype("float32"), rng.uniform(0, 5000, s2.shape).astype("float32")
    if type_ == "image_constante":
        return np.full_like(s1, -12.0), np.full_like(s2, 1500.0)
    if type_ == "nuage_total_sans_s1":
        return None, (7000 + rng.normal(0, 300, s2.shape)).clip(1, None).astype("float32")
    if type_ == "bandes_melangees":
        return s1, s2[rng.permutation(s2.shape[0])]
    raise ValueError(type_)


def main():
    import pandas as pd
    from config import DATA
    index = pd.read_csv(DATA / "index.csv").fillna("")
    ligne = index[(index["ensemble"] == "test") & (index["classe"] == "ALERTE")].iloc[0]
    s1, s2 = charger(ligne)
    stats = charger_stats()
    for type_, famille in TYPES.items():
        try:
            preparer(*generer(type_, s1, s2), stats)
            resultat = "passe le contrôle de format"
        except EntreeInvalide as e:
            resultat = f"rejeté ({e})"
        attendu = "rejeté" if famille == "format" else "passe"
        ok = resultat.startswith(attendu)
        print(f"{'OK ' if ok else 'KO '} {type_:22s} [{famille}] -> {resultat}")


if __name__ == "__main__":
    main()
