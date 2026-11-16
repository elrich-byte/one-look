"""Étape 4 : entraîner et comparer les modèles (CPU, quelques minutes chacun).

Modèles :
  cnn_rgb        petit CNN appris de zéro, entrée RGB (3 canaux)
  cnn_complet    même CNN, entrée complète (13 bandes S2 + 2 radar), modality dropout
  mobilenet_rgb  MobileNetV3-small pré-entraîné sur ImageNet (photos), entrée RGB

Données : tuiles réduites à 128 x 128 (cache, 1 pixel sur 4 : ~40 m).
Classes : NORMAL, INONDATION, FEU. Choix du meilleur état sur val, test évalué une fois.
Sorties : modeles/<nom>.pt, docs/resultats_modeles.json.

Usage : python src/entrainer.py cnn_rgb cnn_complet mobilenet_rgb
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS, RACINE  # noqa: E402
from donnees import charger_stats, normaliser  # noqa: E402
from rgb import rgb_float  # noqa: E402

CLASSES = ["NORMAL", "INONDATION", "FEU"]
CACHE = DATA / "cache"
MODELES = RACINE / "modeles"
IMAGENET_MOY = np.array([0.485, 0.456, 0.406], "float32")[:, None, None]
IMAGENET_EC = np.array([0.229, 0.224, 0.225], "float32")[:, None, None]
torch.set_num_threads(max(1, torch.get_num_threads()))
try:  # certificats Windows pour télécharger les poids pré-entraînés
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass


# ---------------------------------------------------------------- données
def entree(d, mode, stats):
    """Tableau d'entrée du modèle depuis un fichier du cache. None si impossible."""
    s1 = d["s1"] if "s1" in d else None
    s2 = d["s2"] if "s2" in d else None
    if mode == "complet":
        if s1 is None and s2 is None:
            return None
        x, dispo = normaliser(s1, s2, stats)
        return x if dispo.any() else None
    if s2 is None or not (s2 != 0).any(0).mean() > 0.2:
        return None
    x = rgb_float(s2)
    if mode == "imagenet":
        x = (x - IMAGENET_MOY) / IMAGENET_EC
    else:
        x = x * 2 - 1
    return x.astype("float32")


def charger_ensemble(index, ensemble, mode, stats, classes=CLASSES):
    lignes = index[(index["ensemble"] == ensemble) & index["classe"].isin(classes)]
    X, Y, ids = [], [], []
    for _, l in lignes.iterrows():
        f = CACHE / f"{l['id']}.npz"
        if not f.exists():
            continue
        x = entree(np.load(f), mode, stats)
        if x is None:
            continue
        X.append(x); Y.append(CLASSES.index(l["classe"]) if l["classe"] in CLASSES else -1); ids.append(l["id"])
    return np.stack(X), np.array(Y), ids


def augmenter(x, rng):
    if rng.random() < 0.5:
        x = x[:, :, ::-1]
    if rng.random() < 0.5:
        x = x[:, ::-1, :]
    return np.rot90(x, rng.integers(4), axes=(1, 2)).copy()


# ---------------------------------------------------------------- modèles
class PetitCNN(nn.Module):
    def __init__(self, canaux, n_classes=3):
        super().__init__()
        couches, c = [], canaux
        for sortie in (32, 64, 96, 128):
            couches += [nn.Conv2d(c, sortie, 3, padding=1, bias=False), nn.BatchNorm2d(sortie),
                        nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = sortie
        self.corps = nn.Sequential(*couches)
        self.tete = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.tete(self.corps(x))


def mobilenet(n_classes=3):
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


CONFIGS = {
    "cnn_rgb": dict(mode="rgb", modele=lambda: PetitCNN(3), epoques=40, lr=1e-3, dropout_capteur=0),
    "cnn_complet": dict(mode="complet", modele=lambda: PetitCNN(15), epoques=40, lr=1e-3, dropout_capteur=0.3),
    "mobilenet_rgb": dict(mode="imagenet", modele=mobilenet, epoques=25, lr=3e-4, dropout_capteur=0),
}


# ---------------------------------------------------------------- métriques
def metriques(y, p):
    """y, p : classes vraies et prédites (0 NORMAL, 1 INONDATION, 2 FEU)."""
    res = {}
    for k, nom in enumerate(CLASSES[1:], 1):
        vp = int(((p == k) & (y == k)).sum()); fp = int(((p == k) & (y != k)).sum()); fn = int(((p != k) & (y == k)).sum())
        res[nom] = dict(precision=vp / (vp + fp) if vp + fp else None, rappel=vp / (vp + fn) if vp + fn else None,
                        n=int((y == k).sum()))
    normaux = y == 0
    res["fausses_alertes"] = float((p[normaux] != 0).mean()) if normaux.any() else None
    res["n_normaux"] = int(normaux.sum())
    f1s = []
    for k in range(3):
        if (y == k).any():
            vp = ((p == k) & (y == k)).sum(); prec = vp / max((p == k).sum(), 1); rap = vp / (y == k).sum()
            f1s.append(0 if prec + rap == 0 else 2 * prec * rap / (prec + rap))
    res["f1_macro"] = float(np.mean(f1s))
    res["exactitude"] = float((y == p).mean())
    res["confusion"] = [[int(((y == a) & (p == b)).sum()) for b in range(3)] for a in range(3)]
    return res


def predire(modele, X, lot=64):
    modele.eval()
    sorties = []
    with torch.no_grad():
        for i in range(0, len(X), lot):
            sorties.append(modele(torch.from_numpy(X[i:i + lot])).numpy())
    return np.concatenate(sorties) if sorties else np.zeros((0, 3))


# ---------------------------------------------------------------- entraînement
def entrainer(nom, graine=0):
    cfg = CONFIGS[nom]
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    stats = charger_stats()
    index = pd.read_csv(DATA / "index.csv").fillna("")
    Xtr, Ytr, _ = charger_ensemble(index, "train", cfg["mode"], stats)
    Xva, Yva, _ = charger_ensemble(index, "val", cfg["mode"], stats)
    print(f"[{nom}] train {np.bincount(Ytr, minlength=3)}  val {np.bincount(Yva, minlength=3)}", flush=True)

    modele = cfg["modele"]()
    poids = torch.tensor(len(Ytr) / (3 * np.maximum(np.bincount(Ytr, minlength=3), 1)), dtype=torch.float32)
    perte = nn.CrossEntropyLoss(weight=poids)
    opt = torch.optim.AdamW(modele.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epoques"])
    meilleur, etat, historique = -1, None, []
    t0 = time.time()
    for ep in range(cfg["epoques"]):
        modele.train()
        ordre = rng.permutation(len(Xtr))
        for i in range(0, len(ordre), 32):
            lot = []
            for j in ordre[i:i + 32]:
                x = augmenter(Xtr[j], rng)
                if cfg["dropout_capteur"] and rng.random() < cfg["dropout_capteur"]:
                    x = x.copy()
                    if rng.random() < 0.5:
                        x[:2] = 0          # radar absent
                    else:
                        x[2:] = 0          # optique absente
                lot.append(x)
            xb = torch.from_numpy(np.stack(lot)); yb = torch.from_numpy(Ytr[ordre[i:i + 32]]).long()
            opt.zero_grad(); l = perte(modele(xb), yb); l.backward(); opt.step()
        sched.step()
        m = metriques(Yva, predire(modele, Xva).argmax(1))
        historique.append(dict(epoque=ep + 1, f1_val=m["f1_macro"], fausses_alertes_val=m["fausses_alertes"]))
        if m["f1_macro"] > meilleur:
            meilleur, etat = m["f1_macro"], {k: v.clone() for k, v in modele.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"[{nom}] époque {ep + 1:2d}  F1 val {m['f1_macro']:.3f}  fausses alertes val {m['fausses_alertes']:.2f}", flush=True)
    modele.load_state_dict(etat)
    MODELES.mkdir(exist_ok=True)
    torch.save(dict(etat=etat, nom=nom, mode=cfg["mode"], classes=CLASSES), MODELES / f"{nom}.pt")
    print(f"[{nom}] meilleur F1 val {meilleur:.3f}, {time.time() - t0:.0f} s", flush=True)
    return modele, historique


class Enveloppe(nn.Module):
    """Renvoie les logits ET les caractéristiques internes (pour détecter l'inconnu)."""
    def __init__(self, modele):
        super().__init__()
        self.m = modele

    def forward(self, x):
        if isinstance(self.m, PetitCNN):
            f = self.m.tete[1](self.m.tete[0](self.m.corps(x)))
            return self.m.tete[3](f), f
        f = torch.flatten(self.m.avgpool(self.m.features(x)), 1)
        return self.m.classifier(f), f


def predire_avec_caract(modele, X, lot=64):
    env = Enveloppe(modele).eval()
    L, F = [], []
    with torch.no_grad():
        for i in range(0, len(X), lot):
            l, f = env(torch.from_numpy(X[i:i + lot]))
            L.append(l.numpy()); F.append(f.numpy())
    return np.concatenate(L), np.concatenate(F)


def charger_modele(nom):
    sauv = torch.load(MODELES / f"{nom}.pt", weights_only=False)
    modele = CONFIGS[nom]["modele"]()
    modele.load_state_dict(sauv["etat"])
    modele.eval()
    return modele


if __name__ == "__main__":
    noms = sys.argv[1:] or list(CONFIGS)
    fichier = DOCS / "historique_entrainement.json"
    tout = json.loads(fichier.read_text(encoding="utf-8")) if fichier.exists() else {}
    for nom in noms:
        _, hist = entrainer(nom)
        tout[nom] = hist
    fichier.write_text(json.dumps(tout, indent=1, ensure_ascii=False), encoding="utf-8")
