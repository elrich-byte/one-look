"""Étape 2.5 : jeu de données PyTorch prêt pour l'entraînement (étapes 3 et 4).

- Classes : NORMAL = 0, ALERTE = 1. La zone grise et les tuiles écartées sont exclues.
- Modality dropout (entraînement) : masque S1 ou S2 au hasard, pour qu'un même
  modèle fonctionne avec S1 seul, S2 seul ou les deux (cadrage, section 6).
- Réduction optionnelle de la taille (ex. 128 px) pour un modèle léger.

Exemple :
    from dataset import TuilesDataset
    train = TuilesDataset("train", taille=128, modality_dropout=0.3)
    x, y, dispo = train[0]     # x : (15, 128, 128), y : 0/1, dispo : (2,)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402
from donnees import charger, charger_stats, preparer  # noqa: E402

CLASSES = {"NORMAL": 0, "ALERTE": 1}


def reduire(x, taille):
    """Moyenne par blocs : 512 -> taille (diviseur de 512)."""
    f = x.shape[-1] // taille
    return x.reshape(x.shape[0], taille, f, taille, f).mean(axis=(2, 4))


def lignes(ensemble, index=None):
    index = pd.read_csv(DATA / "index.csv").fillna("") if index is None else index
    return index[(index["ensemble"] == ensemble) & (index["classe"].isin(CLASSES))].reset_index(drop=True)


def exemple(ligne, stats, taille=512, modality_dropout=0.0, rng=None):
    """Version numpy (testable sans torch) : renvoie x, y, dispo."""
    x, dispo = preparer(*charger(ligne), stats)
    if modality_dropout and dispo.all() and rng is not None and rng.random() < modality_dropout:
        cache = rng.integers(2)            # 0 : cacher S1, 1 : cacher S2
        sl = slice(0, 2) if cache == 0 else slice(2, 15)
        x[sl] = 0
        dispo = dispo.copy()
        dispo[cache] = False
    if taille != x.shape[-1]:
        x = reduire(x, taille)
    return x.astype("float32"), CLASSES[ligne["classe"]], dispo


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # torch absent : seule la version numpy est disponible
    Dataset = object


class TuilesDataset(Dataset):
    def __init__(self, ensemble, taille=512, modality_dropout=0.0, graine=0):
        self.lignes = lignes(ensemble)
        self.stats = charger_stats()
        self.taille = taille
        self.modality_dropout = modality_dropout if ensemble == "train" else 0.0
        self.rng = np.random.default_rng(graine)

    def __len__(self):
        return len(self.lignes)

    def __getitem__(self, i):
        x, y, dispo = exemple(self.lignes.iloc[i], self.stats, self.taille, self.modality_dropout, self.rng)
        return torch.from_numpy(x), torch.tensor(y), torch.from_numpy(dispo)

    def poids_classes(self):
        """Poids inverses des fréquences, pour compenser le déséquilibre ALERTE / NORMAL."""
        n = self.lignes["classe"].map(CLASSES).value_counts().sort_index()
        return (len(self.lignes) / (len(n) * n)).tolist()
