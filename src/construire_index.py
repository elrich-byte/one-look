"""Étapes 2.1 à 2.4 : rassembler toutes les sources dans data/index.csv.

Une ligne par tuile, avec : source, catégorie, classe (ALERTE / NORMAL / GRIS /
ECARTEE / INCONNU), ensemble (train / val / test / inconnu) et chemins S1 / S2.

Règles de découpage (aucune fuite géographique) :
- Sen1Floods11 : par pays (config.py).
- Scènes hors crue sur les mêmes lieux : même ensemble que leur pays.
- Scènes mondiales (eau permanente, négatifs difficiles) : par blocs de 10° x 10°,
  70 % train / 15 % val / 15 % test, tirage déterministe.
- Scènes inconnues : ensemble « inconnu », jamais utilisé à l'entraînement.

Les sources non encore téléchargées sont simplement ignorées.
Usage : python src/construire_index.py
"""
import hashlib
import math
import re

import pandas as pd

from config import DATA, DOCS, MAILLE_DEG, PAYS_TEST, PAYS_VAL


def ensemble_pays(pays):
    return "test" if pays in PAYS_TEST else "val" if pays in PAYS_VAL else "train"


def ensemble_maille(lon, lat):
    cle = f"{math.floor(lon / MAILLE_DEG)}_{math.floor(lat / MAILLE_DEG)}"
    u = int(hashlib.md5(cle.encode()).hexdigest(), 16) % 100
    return "train" if u < 70 else "val" if u < 85 else "test"


def sen1floods11():
    df = pd.read_csv(DATA / "index_sen1floods11.csv")
    stats = pd.read_csv(DOCS / "stats_tuiles.csv")[["tile", "perm_pct"]]
    df = df.merge(stats, on="tile")
    cat = "crue_evenement"
    df["categorie"] = cat
    normal = df["classe"] == "NORMAL"
    df.loc[normal, "categorie"] = "normal_pendant_crue"
    df.loc[normal & (df["perm_pct"] > 1), "categorie"] = "normal_eau_permanente"
    df.loc[normal & (df["nuage_pct"] > 30), "categorie"] = "normal_nuageux"
    df = df.rename(columns={"tile": "id", "label": "masque"})
    return df[["id", "source", "categorie", "pays", "ensemble", "classe", "s1", "s2", "masque"]]


def perm_water():
    """814 tuiles d'eau permanente (Sen1Floods11), si téléchargées."""
    dossier = DATA / "sen1floods11" / "perm_water"
    if not dossier.exists():
        return pd.DataFrame()
    import rasterio
    lignes = []
    for p in sorted(dossier.rglob("*.tif")):
        if not p.name.startswith("sentinel_"):
            continue
        nombres = re.findall(r"-?\d+\.\d+", p.stem)
        if len(nombres) < 2:
            continue
        lon, lat = float(nombres[-2]), float(nombres[-1])
        with rasterio.open(p) as src:
            nb = src.count
        rel = str(p.relative_to(DATA)).replace("\\", "/")
        masque = next(iter(dossier.rglob(p.name.replace("sentinel_", "water_"))), None)
        lignes.append(dict(
            id=p.stem, source="sen1floods11_perm_water", categorie="eau_permanente_mondiale",
            pays="", ensemble=ensemble_maille(lon, lat), classe="NORMAL",
            s1=rel if nb == 2 else "", s2=rel if nb >= 12 else "",
            masque=str(masque.relative_to(DATA)).replace("\\", "/") if masque else ""))
    return pd.DataFrame(lignes)


def scenes():
    """Scènes openEO (hors crue, négatifs, inconnues) et CaBuAr, si téléchargées."""
    dossier = DATA / "scenes"
    lignes = []
    if (dossier / "manifest.csv").exists():
        for _, r in pd.read_csv(dossier / "manifest.csv").iterrows():
            s1, s2 = dossier / f"{r['id']}_S1.tif", dossier / f"{r['id']}_S2.tif"
            if not (s1.exists() or s2.exists()):
                continue
            if r["classe"] == "INCONNU":
                ens = "inconnu"
            elif r["groupe"] == "pays":
                ens = ensemble_pays(r["pays"])
            else:
                ens = ensemble_maille(r["lon"], r["lat"])
            lignes.append(dict(id=r["id"], source="copernicus_openeo", categorie=r["categorie"],
                               pays=r["pays"] if isinstance(r["pays"], str) else "", ensemble=ens,
                               classe=r["classe"],
                               s1=f"scenes/{s1.name}" if s1.exists() else "",
                               s2=f"scenes/{s2.name}" if s2.exists() else "", masque=""))
    if (dossier / "manifest_cabuar.csv").exists():
        for _, r in pd.read_csv(dossier / "manifest_cabuar.csv").iterrows():
            lignes.append(dict(id=r["id"], source="cabuar", categorie=r["categorie"], pays="",
                               ensemble="inconnu", classe="INCONNU", s1="",
                               s2=f"scenes/{r['id']}_S2.tif", masque=""))
    return pd.DataFrame(lignes)


def main():
    parties = [sen1floods11(), perm_water(), scenes()]
    df = pd.concat([p for p in parties if len(p)], ignore_index=True).fillna("")
    df.to_csv(DATA / "index.csv", index=False)
    print(f"{len(df)} tuiles -> {DATA / 'index.csv'}\n")
    print(pd.crosstab([df["source"], df["ensemble"]], df["classe"], margins=True).to_string())
    utiles = df[df["classe"].isin(["ALERTE", "NORMAL"])]
    print("\nPour l'entraînement et l'évaluation (hors zone grise et écartées) :")
    print(pd.crosstab(utiles["ensemble"], utiles["classe"], margins=True).to_string())


if __name__ == "__main__":
    main()
