"""Étape 3 : référence simple « RGB contre image complète » pour les inondations.

Même méthode pour chaque jeu de bandes, seule l'entrée change :
1. un détecteur d'eau par pixel (régression logistique), appris sur les pixels
   annotés des tuiles train de Sen1Floods11 ;
2. pour chaque tuile, part des pixels détectés comme eau ;
3. alerte INONDATION si cette part dépasse un seuil, choisi sur le train
   (meilleur équilibre entre rappel et fausses alertes).
Évaluation sur l'ensemble val (pays jamais vus). Le test reste intact pour la fin.

Limite assumée : le détecteur voit toute l'eau, y compris rivières et lacs
normaux. C'est justement ce que les modèles suivants doivent apprendre à éviter.

Usage : python src/baseline_pixels.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402

CACHE = DATA / "cache"
B2, B3, B4 = 1, 2, 3
JEUX = {
    "RGB (3 bandes)": ("s2", [B4, B3, B2]),
    "S2 complet (13 bandes)": ("s2", list(range(13))),
    "S1 radar (VV, VH)": ("s1", [0, 1]),
    "Complet (S2 + S1, 15)": ("both", None),
}


def caracteristiques(d, jeu):
    """(n_pixels, n_caract.) et masque des pixels lisibles, pour une tuile du cache."""
    source, bandes = JEUX[jeu]
    s2 = d["s2"] / 10000.0 if "s2" in d else None
    s1 = d["s1"] if "s1" in d else None
    if source == "s2":
        if s2 is None:
            return None, None
        x, ok = s2[bandes], (s2 != 0).any(0)
    elif source == "s1":
        if s1 is None:
            return None, None
        x, ok = s1, np.isfinite(s1).all(0)
    else:
        if s2 is None or s1 is None:
            return None, None
        x = np.concatenate([s2, s1])
        ok = (s2 != 0).any(0) & np.isfinite(s1).all(0)
    return np.nan_to_num(x.reshape(x.shape[0], -1).T), ok.ravel()


def charger_tuiles():
    index = pd.read_csv(DATA / "index.csv").fillna("")
    garde = index["classe"].isin(["INONDATION", "NORMAL"]) & index["ensemble"].isin(["train", "val"])
    return index[garde & (index["source"] != "sen1floods11_perm_water")].reset_index(drop=True)


def entrainer_pixels(tuiles, jeu, rng, par_tuile=1500):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, Y = [], []
    for _, l in tuiles[(tuiles["ensemble"] == "train") & (tuiles["source"] == "sen1floods11")].iterrows():
        d = np.load(CACHE / f"{l['id']}.npz")
        x, ok = caracteristiques(d, jeu)
        if x is None:
            continue
        m = d["masque"].ravel()
        idx = np.flatnonzero(ok & (m != -1))
        if idx.size == 0:
            continue
        idx = rng.choice(idx, min(par_tuile, idx.size), replace=False)
        X.append(x[idx]); Y.append(m[idx] == 1)
    X, Y = np.concatenate(X), np.concatenate(Y)
    modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    modele.fit(X, Y)
    return modele


def part_eau(modele, tuiles, jeu):
    parts = []
    for _, l in tuiles.iterrows():
        x, ok = caracteristiques(np.load(CACHE / f"{l['id']}.npz"), jeu)
        if x is None or ok.mean() < 0.2:
            parts.append(np.nan)
            continue
        parts.append(100 * modele.predict(x[ok]).mean())
    return np.array(parts)


def metriques(y, alerte):
    vp = int(((y == 1) & alerte).sum()); fp = int(((y == 0) & alerte).sum())
    fn = int(((y == 1) & ~alerte).sum()); vn = int(((y == 0) & ~alerte).sum())
    prec = vp / (vp + fp) if vp + fp else float("nan")
    rap = vp / (vp + fn) if vp + fn else float("nan")
    return dict(precision=prec, rappel=rap, fausses_alertes=fp / (fp + vn) if fp + vn else float("nan"),
                f1=2 * prec * rap / (prec + rap) if prec + rap else 0.0, vp=vp, fp=fp, fn=fn, vn=vn)


def choisir_seuil(parts, y):
    """Seuil qui maximise l'équilibre rappel / (1 - fausses alertes) sur le train."""
    meilleur, score_max = 0.0, -1
    for s in np.unique(np.round(parts[~np.isnan(parts)], 2)):
        m = metriques(y, parts >= s)
        score = m["rappel"] + (1 - m["fausses_alertes"])
        if score > score_max:
            meilleur, score_max = s, score
    return meilleur


def main():
    rng = np.random.default_rng(0)
    tuiles = charger_tuiles()
    y = (tuiles["classe"] == "INONDATION").astype(int).to_numpy()
    resultats = []
    for jeu in JEUX:
        modele = entrainer_pixels(tuiles, jeu, rng)
        parts = part_eau(modele, tuiles, jeu)
        disponible = ~np.isnan(parts)
        tr = (tuiles["ensemble"] == "train").to_numpy() & disponible
        seuil = choisir_seuil(parts[tr], y[tr])
        # vitesse : une tuile pleine résolution (512 x 512) simulée
        x = np.random.default_rng(1).normal(size=(512 * 512, modele[-1].coef_.shape[1])).astype("float32")
        t0 = time.perf_counter(); modele.predict(x); ms = 1000 * (time.perf_counter() - t0)
        for nom, sel in [("val", tuiles["ensemble"] == "val"),
                         ("val, scènes Copernicus seules", (tuiles["ensemble"] == "val") & (tuiles["source"] != "sen1floods11"))]:
            sel = sel.to_numpy() & disponible
            if not sel.any():
                continue
            m = metriques(y[sel], parts[sel] >= seuil)
            resultats.append(dict(jeu=jeu, evaluation=nom, seuil_pct_eau=seuil, n=int(sel.sum()),
                                  **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
                                  parametres=int(modele[-1].coef_.size + 1), ms_par_tuile=round(ms, 1)))
        print(f"{jeu} : seuil {seuil} % d'eau", flush=True)
    df = pd.DataFrame(resultats)
    df.to_csv(DOCS / "resultats_baseline.csv", index=False)
    print(df.drop(columns=["vp", "fp", "fn", "vn"]).to_string(index=False))
    print(df[["jeu", "evaluation", "vp", "fp", "fn", "vn"]].to_string(index=False))


if __name__ == "__main__":
    main()
