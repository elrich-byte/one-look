"""Harmoniser CaBuAr (S2 L2A) vers l'apparence des images L1C du projet.

CaBuAr est corrigé de l'atmosphère (L2A) : dans le visible, ses images sont bien plus
sombres que nos images L1C (bleu médian ~400 contre ~950). Sans correction, un modèle
peut apprendre « image sombre = feu » au lieu de reconnaître le brûlé.
Correction simple, par bande : on aligne moyenne et écart-type des images CaBuAr
« avant incendie » (normales) sur ceux des tuiles normales L1C d'entraînement.
La même transformation est appliquée aux images « après » : le contraste brûlé /
intact est conservé. Ne modifie que le cache (data/cache/cabuar_*.npz).

Usage : python src/harmoniser_cabuar.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402

CACHE = DATA / "cache"


def stats_bandes(ids):
    somme, somme2, n = np.zeros(13), np.zeros(13), 0
    for i in ids:
        s2 = np.load(CACHE / f"{i}.npz")["s2"].reshape(13, -1).astype("float64")
        ok = (s2 != 0).any(0)
        somme += s2[:, ok].sum(1); somme2 += (s2[:, ok] ** 2).sum(1); n += ok.sum()
    m = somme / n
    return m, np.sqrt(somme2 / n - m ** 2)


def main():
    index = pd.read_csv(DATA / "index.csv").fillna("")
    ref = index[(index["ensemble"] == "train") & (index["classe"] == "NORMAL") & (index["source"] != "cabuar") & (index["s2"] != "")]
    avant = index[(index["source"] == "cabuar") & (index["classe"] == "NORMAL")]
    m_ref, e_ref = stats_bandes(ref["id"])
    m_cab, e_cab = stats_bandes(avant["id"])
    a = np.where(e_cab > 0, e_ref / np.maximum(e_cab, 1e-6), 1.0)
    b = m_ref - a * m_cab
    a[10], b[10] = 0.0, 0.0                                        # B10 absente de CaBuAr
    n = 0
    for i in index[index["source"] == "cabuar"]["id"]:
        f = CACHE / f"{i}.npz"
        d = dict(np.load(f))
        if d.get("harmonise", False):
            continue
        s2 = d["s2"].astype("float32")
        ok = (s2 != 0).any(0)
        s2 = np.where(ok, np.clip(a[:, None, None] * s2 + b[:, None, None], 1, 20000), 0).astype("float32")
        d["s2"], d["harmonise"] = s2, np.array(True)
        np.savez_compressed(f.with_suffix(".tmp.npz"), **d)
        f.with_suffix(".tmp.npz").replace(f)
        n += 1
    (DOCS / "harmonisation_cabuar.json").write_text(json.dumps(
        {"pente": a.round(4).tolist(), "origine": b.round(1).tolist(),
         "moyenne_L1C": m_ref.round(1).tolist(), "moyenne_CaBuAr_avant": m_cab.round(1).tolist()}, indent=1),
        encoding="utf-8")
    print(f"{n} tuiles CaBuAr harmonisées")
    print("pente   ", a.round(2)); print("origine ", b.round(0))


if __name__ == "__main__":
    main()
