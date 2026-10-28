"""Étape 2.5 : prétraitement commun à l'entraînement, à l'évaluation et à la démo.

Une tuile devient un tableau de 15 canaux (2 S1 + 13 S2), 512 x 512, normalisé
avec des statistiques calculées sur l'ensemble d'entraînement uniquement.
Un capteur absent ou illisible est remplacé par des zéros (= la moyenne après
normalisation), et signalé dans `dispo` (S1, S2).

Usage :
  python src/donnees.py --stats      # calcule data/normalisation.json (train uniquement)
  python src/donnees.py --verifier   # contrôle toutes les tuiles de data/index.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BANDES_S1, BANDES_S2, DATA  # noqa: E402

TAILLE = 512
S1_MIN, S1_MAX = -50.0, 5.0      # dB, écrêtage
S2_MAX = 2.0                     # réflectance (après / 10 000), écrêtage
DISPO_MIN = 0.2                  # part minimale de pixels valides pour considérer un capteur lisible
FICHIER_STATS = DATA / "normalisation.json"


class EntreeInvalide(Exception):
    """Entrée qui ne respecte pas le format attendu : la décision sera INCONNU."""


def lire(chemin):
    import rasterio
    with rasterio.open(chemin) as src:
        return src.read().astype("float32")


def charger(ligne, racine=DATA):
    """Lit S1 et S2 d'une ligne de l'index (chemins relatifs à data/). None si absent."""
    s1 = lire(Path(racine) / ligne["s1"]) if ligne.get("s1") else None
    s2 = lire(Path(racine) / ligne["s2"]) if ligne.get("s2") else None
    return s1, s2


def verifier(s1, s2):
    """Contrôles de format avant le modèle. Lève EntreeInvalide avec la raison."""
    if s1 is None and s2 is None:
        raise EntreeInvalide("aucun capteur fourni")
    if s1 is not None and s1.shape != (len(BANDES_S1), TAILLE, TAILLE):
        raise EntreeInvalide(f"S1 : forme {s1.shape}, attendu (2, 512, 512)")
    if s2 is not None and s2.shape != (len(BANDES_S2), TAILLE, TAILLE):
        raise EntreeInvalide(f"S2 : forme {s2.shape}, attendu (13, 512, 512)")
    if s1 is not None:
        v = s1[np.isfinite(s1)]
        # En dB, la rétrodiffusion est presque toujours négative ; en linéaire, toujours positive.
        if v.size and ((v >= 0).mean() > 0.95 or not -60 < np.median(v) < 10):
            raise EntreeInvalide(f"S1 : médiane {np.median(v):.2f}, valeurs pas en dB")
    if s2 is not None:
        v = s2[np.isfinite(s2)]
        # Quelques pixels négatifs isolés existent (artefacts de rééchantillonnage) :
        # on ne rejette que si plus de 1 % des valeurs sont hors plage.
        if v.size and ((v > 30000) | (v < 0)).mean() > 0.01:
            raise EntreeInvalide("S2 : valeurs hors de la plage de réflectance x 10 000")


def masques_validite(s1, s2):
    """Pixels lisibles par capteur (booléens 512 x 512), None si capteur absent."""
    m1 = None if s1 is None else np.isfinite(s1).all(axis=0)
    m2 = None if s2 is None else (np.isfinite(s2).all(axis=0) & (s2 != 0).any(axis=0))
    return m1, m2


def preparer(s1, s2, stats):
    """Retourne x (15, 512, 512) float32 normalisé et dispo = [S1 lisible, S2 lisible]."""
    verifier(s1, s2)
    m1, m2 = masques_validite(s1, s2)
    dispo = np.array([m1 is not None and m1.mean() >= DISPO_MIN,
                      m2 is not None and m2.mean() >= DISPO_MIN], dtype=bool)
    if not dispo.any():
        raise EntreeInvalide("aucun capteur lisible (S1 absent ou vide, S2 absent ou vide)")

    x = np.zeros((len(BANDES_S1) + len(BANDES_S2), TAILLE, TAILLE), dtype="float32")
    moy, ec = np.array(stats["moyenne"], "float32"), np.array(stats["ecart_type"], "float32")
    if dispo[0]:
        a = np.clip(s1, S1_MIN, S1_MAX)
        a = (a - moy[:2, None, None]) / ec[:2, None, None]
        x[:2] = np.where(m1, np.nan_to_num(a), 0)
    if dispo[1]:
        a = np.clip(s2 / 10000.0, 0, S2_MAX)
        a = (a - moy[2:, None, None]) / ec[2:, None, None]
        x[2:] = np.where(m2, np.nan_to_num(a), 0)
    return x, dispo


def calculer_stats(index, pas=8):
    """Moyenne et écart-type par canal, sur l'ensemble train uniquement (1 pixel sur pas²)."""
    train = index[(index["ensemble"] == "train") & (index["classe"] != "INCONNU")]
    somme = np.zeros(15)
    somme2 = np.zeros(15)
    n = np.zeros(15)
    for _, ligne in train.iterrows():
        s1, s2 = charger(ligne)
        m1, m2 = masques_validite(s1, s2)
        if s1 is not None:
            a = np.clip(s1, S1_MIN, S1_MAX)[:, ::pas, ::pas]
            ok = m1[::pas, ::pas]
            somme[:2] += a[:, ok].sum(1); somme2[:2] += (a[:, ok] ** 2).sum(1); n[:2] += ok.sum()
        if s2 is not None:
            a = np.clip(s2 / 10000.0, 0, S2_MAX)[:, ::pas, ::pas]
            ok = m2[::pas, ::pas]
            somme[2:] += a[:, ok].sum(1); somme2[2:] += (a[:, ok] ** 2).sum(1); n[2:] += ok.sum()
    moy = somme / n
    ec = np.sqrt(somme2 / n - moy ** 2)
    return {"canaux": BANDES_S1 + BANDES_S2, "moyenne": moy.round(6).tolist(),
            "ecart_type": ec.round(6).tolist(), "nb_tuiles_train": int(len(train)),
            "s1_ecretage_db": [S1_MIN, S1_MAX], "s2_echelle": "valeur / 10000, écrêtée à [0, 2]"}


def charger_stats():
    with open(FICHIER_STATS, encoding="utf-8") as f:
        return json.load(f)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--verifier", action="store_true")
    args = ap.parse_args()
    index = pd.read_csv(DATA / "index.csv").fillna("")

    if args.stats:
        stats = calculer_stats(index)
        with open(FICHIER_STATS, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        for c, m, e in zip(stats["canaux"], stats["moyenne"], stats["ecart_type"]):
            print(f"{c:4s} moyenne {m:9.4f}  écart-type {e:8.4f}")
        print(f"-> {FICHIER_STATS}")

    if args.verifier:
        stats = charger_stats()
        bilan = {"ok": 0, "S1 seul": 0, "S2 seul": 0, "invalide": 0}
        for _, ligne in index.iterrows():
            try:
                x, dispo = preparer(*charger(ligne), stats)
                assert x.shape == (15, TAILLE, TAILLE) and np.isfinite(x).all()
                bilan["ok" if dispo.all() else "S1 seul" if dispo[0] else "S2 seul"] += 1
            except EntreeInvalide as e:
                bilan["invalide"] += 1
                print(f"{ligne['id']} ({ligne['classe']}) : {e}")
        print(bilan)


if __name__ == "__main__":
    main()
